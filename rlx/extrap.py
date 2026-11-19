"""Extrapolate the GRPO weight trajectory from the observed prefix (steps 1..T_cut).

Per weight tensor: stack prefix deltas D [T, P]; top-r trajectory modes via the T x T Gram matrix;
coefficient time series C [T, r] (V [P, r] orthonormal).
  relex   RELEX (2026), faithful: rank-1, least-squares line in t over 1..T, delta(T*) = c(T*) v1
  line3   anchored, rank 3: delta(T*) = delta_real(T) + V (slope * (T* - T))         (constant LS-fit velocity)
  linerec3 anchored, rank 3, constant RECENT (smoothed) velocity: isolates Koopman dynamics from velocity estimation
  koop3   anchored, rank 3, Koopman on velocities: Hankel state s_t = [v_t, v_{t-1}, v_{t-2}], s_{t+1} = K s_t,
          K = K0 + E with ridge on E (K0 = constant-velocity operator, i.e. RELEX's dynamics), spectral radius <= 1.
          Reduces to line3 when the data carry no curvature; captures saturating/decaying modes when they do.
Modes:
  --mode oneshot : predict T* in --targets for relex/line3/koop3, evaluate GSM8K, weight-space error vs reference
  --mode cc      : course correction: extrapolate k steps, run r REAL GRPO steps (re-anchor the position on the
                   observed weights), repeat until T_final; evaluate.  -> <out>/<mode>[_<method>].json"""
import argparse, json, os, random, time

import torch

from .common import evaluate, grpo_step, load_gsm, load_model, params, set_weights


def wait(path, poll=20):
    while not os.path.exists(path):
        time.sleep(poll)


def smooth(C, w=5):
    out = torch.empty_like(C)
    for t in range(len(C)):
        out[t] = C[max(0, t - w + 1): t + 1].mean(0)
    return out


class Traj:
    """Per-tensor modes + fitted dynamics."""

    def __init__(self, D, r, d=3, alpha=1.0):
        T = D.shape[0]
        G = D @ D.T
        ev, U = torch.linalg.eigh(G)
        ev, U = ev.flip(0)[:r].clamp(min=1e-12), U.flip(1)[:, :r]
        s = ev.sqrt()
        self.V = (D.T @ U) / s                                         # [P, r] orthonormal
        self.C = U * s                                                 # [T, r]
        t = torch.arange(1, T + 1, dtype=D.dtype, device=D.device)
        X = torch.stack([torch.ones_like(t), t], 1)
        coef = torch.linalg.lstsq(X, self.C).solution                  # [2, r]
        self.icpt, self.slope = coef[0], coef[1]
        self.T, self.r, self.d = T, r, d
        Cs = smooth(self.C)
        self.c_anchor_s = Cs[-1]
        v = Cs[1:] - Cs[:-1]                                           # [T-1, r]
        H = torch.stack([torch.cat([v[i - j] for j in range(d)]) for i in range(d - 1, len(v))])   # [n, r*d]
        Xh, Yh = H[:-1], H[1:]
        n = r * d
        K0 = torch.zeros(n, n, dtype=D.dtype, device=D.device)
        K0[:r, :r] = torch.eye(r); K0[r:, :-r] = torch.eye(n - r)     # v persists; shift history
        R = Yh - Xh @ K0.T
        lam = alpha * (Xh ** 2).sum(0).mean().clamp(min=1e-20)
        E = torch.linalg.solve(Xh.T @ Xh + lam * torch.eye(n, dtype=D.dtype, device=D.device), Xh.T @ R).T
        K = K0 + E
        w, Q = torch.linalg.eig(K)
        mag = w.abs()
        w = torch.where(mag > 1, w / mag, w)
        self.K = (Q @ torch.diag(w) @ torch.linalg.inv(Q)).real
        self.s0 = H[-1]

    def relex(self, Tstar):
        return (self.icpt[0] + self.slope[0] * Tstar) * self.V[:, 0]

    def line_shift(self, steps):
        return self.V @ (self.slope * steps)

    def koop_roll(self, s, steps):
        """roll the velocity state; returns (coefficient displacement [r], new state)."""
        disp = torch.zeros(self.r, dtype=s.dtype, device=s.device)
        for _ in range(steps):
            s = self.K @ s
            disp = disp + s[: self.r]
        return disp, s


def build(prefix, base_names, r, dev):
    trajs = {}
    for n in base_names:
        D = torch.stack([p[n].to(dev, torch.float32).reshape(-1) for p in prefix])
        trajs[n] = Traj(D, r)
        del D
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--mode", choices=["oneshot", "cc"], required=True)
    ap.add_argument("--method", default="koop3", choices=["koop3", "line3"])
    ap.add_argument("--t_cut", type=int, default=40)
    ap.add_argument("--targets", default="100,150")
    ap.add_argument("--k", type=int, default=17, help="cc: extrapolated steps per cycle")
    ap.add_argument("--r_real", type=int, default=5, help="cc: real GRPO steps per cycle")
    ap.add_argument("--t_final", type=int, default=150)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--train", default="data/gsm8k_train.jsonl")
    ap.add_argument("--test", default="data/gsm8k_test.jsonl")
    ap.add_argument("--n_test", type=int, default=1319)
    ap.add_argument("--ckpt", default="ckpt_rlx")
    ap.add_argument("--out", default="results_rlx")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    wait(os.path.join(a.ckpt, "PREFIX_DONE"))
    t0 = time.time()
    tok, model = load_model(a.model, dev)
    base = {n: p.detach().clone() for n, p in params(model).items()}
    names = list(base)
    prefix = [torch.load(os.path.join(a.ckpt, "deltas", f"step{t}.pt")) for t in range(1, a.t_cut + 1)]
    last = prefix[-1]
    test = load_gsm(a.test, a.n_test)
    trajs = build(prefix, names, 3, dev)
    del prefix
    print(f"[{a.mode}] fitted {len(trajs)} tensors ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "oneshot":
        res = {"t_cut": a.t_cut, "acc": {}, "werr": {}}
        preds = {}
        for T in [int(x) for x in a.targets.split(",")]:
            for m in ["relex", "line3", "linerec3", "koop3"]:
                delta = {}
                for n in names:
                    tr, lastn = trajs[n], last[n].to(dev, torch.float32).reshape(-1)
                    if m == "relex":
                        d = tr.relex(T)
                    elif m == "line3":
                        d = lastn + tr.line_shift(T - a.t_cut)
                    elif m == "linerec3":
                        d = lastn + tr.V @ (tr.s0[: tr.r] * (T - a.t_cut))
                    else:
                        disp, _ = tr.koop_roll(tr.s0, T - a.t_cut)
                        d = lastn + tr.V @ disp
                    delta[n] = d.view(base[n].shape).to(torch.bfloat16).cpu()
                set_weights(model, base, delta)
                acc = evaluate(model, tok, test, dev)
                res["acc"][f"{m}@{T}"] = acc
                preds[(m, T)] = delta
                print(f"[oneshot] {m}@{T} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
                json.dump(res, open(os.path.join(a.out, "oneshot.json"), "w"), indent=1)
        wait(os.path.join(a.ckpt, "REF_DONE"))
        for T in [int(x) for x in a.targets.split(",")]:
            true = torch.load(os.path.join(a.ckpt, "deltas", f"step{T}.pt"))
            den = sum(float(((true[n].float() - last[n].float()) ** 2).sum()) for n in names) ** 0.5
            for m in ["relex", "line3", "linerec3", "koop3"]:
                num = sum(float(((preds[(m, T)][n].float() - true[n].float()) ** 2).sum()) for n in names) ** 0.5
                res["werr"][f"{m}@{T}"] = num / den
            res["werr"][f"stay@{T}"] = 1.0
            print(f"[oneshot] weight error @{T}: " + " ".join(f"{m} {res['werr'][f'{m}@{T}']:.3f}" for m in ["relex", "line3", "linerec3", "koop3"]), flush=True)
        json.dump(res, open(os.path.join(a.out, "oneshot.json"), "w"), indent=1)
        return
    # ---- course correction
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    train = load_gsm(a.train)
    set_weights(model, base, last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt.load_state_dict(torch.load(os.path.join(a.ckpt, "opt_Tcut.pt"), map_location=dev))
    state = {n: trajs[n].s0.clone() for n in names}
    tau, real_steps, log = a.t_cut, a.t_cut, []
    while tau < a.t_final:
        k = min(a.k, a.t_final - tau)
        with torch.no_grad():
            for n, p in params(model).items():
                tr = trajs[n]
                if a.method == "koop3":
                    disp, state[n] = tr.koop_roll(state[n], k)
                else:
                    disp = tr.slope * k
                p.add_((tr.V @ disp).view(p.shape).to(p.dtype))       # re-anchored: shift from the CURRENT real weights
        tau += k
        rr = min(a.r_real, a.t_final - tau)
        rw = [grpo_step(model, tok, opt, rng.sample(train, a.prompts), a.samples, a.max_new, a.temp, dev) for _ in range(rr)]
        if a.method == "koop3":
            for n in names:
                _, state[n] = trajs[n].koop_roll(state[n], rr)          # keep the velocity state in step with time
        tau += rr; real_steps += rr
        log.append({"tau": tau, "real_steps": real_steps, "reward": sum(rw) / max(1, len(rw))})
        print(f"[cc/{a.method}] tau {tau} real steps {real_steps} reward {log[-1]['reward']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    acc = evaluate(model, tok, test, dev)
    res = {"method": a.method, "k": a.k, "r_real": a.r_real, "t_final": a.t_final, "real_steps": real_steps, "acc": acc, "log": log}
    print(f"[cc/{a.method}] final acc {acc:.3f} with {real_steps} real steps ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(a.out, f"cc_{a.method}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

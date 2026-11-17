"""Stage 2: is TRM's recursion a dynamical system worth modelling with Koopman operators?

Per token, state s_t = [z_H(t), z_L(t)] (PCA), control u = input embedding. Models of s_t -> s_{t+1}:
  persistence  s_{t+1} = s_t                                    (floor)
  DMDc         s_{t+1} = A s_t + B u + c   (ridge least squares) (linear dynamics on raw states)
  MLP          s_{t+1} = s_t + f(s_t, u)                         (nonlinear one-step, ~ceiling)
  Koopman AE   z = [s, g(s)],  z_{t+1} = K z_t + B u,  multi-step rollout loss, read s off z
               rolled out pure-latent, or with periodic re-encoding g <- g(s_hat) every r steps
               (the "course correction" of Fathi et al. 2024)
Skill_k = 1 - MSE(model, k steps) / MSE(persistence, k steps), on held-out puzzles.

Also: accuracy vs ACT step (incl. past trained depth), step sizes, DMD spectrum, and whether
spectral features at the end of ACT step 1 predict final correctness.
Writes <out>/diagnose.json, <out>/REPORT.md, <out>/diagnose.png.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

KS = (1, 3, 6, 12)


def auroc(y, s):
    from scipy.stats import rankdata
    y, s = np.asarray(y).astype(bool), np.asarray(s, float)
    if y.all() or (~y).all() or not np.isfinite(s).all():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def split_by_puzzle(base, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(base)
    rng.shuffle(uniq)
    tr = set(uniq[: len(uniq) // 2])
    m = np.array([b in tr for b in base])
    return np.where(m)[0], np.where(~m)[0]


def pairs(S, U, ex, t_max, cap, rng):
    """(s_t, u, s_{t+1}) over examples ex, t < t_max, all tokens; subsampled to cap."""
    N, T, P, d = S.shape
    t_max = min(t_max, T - 1)
    e = rng.choice(ex, size=cap, replace=True)
    t = rng.integers(0, t_max, size=cap)
    p = rng.integers(0, P, size=cap)
    return S[e, t, p], U[e, p], S[e, t + 1, p]


def fit_dmdc(X, Uc, Y, dev, ridge=1e-3):
    X, Uc, Y = (torch.as_tensor(v, dtype=torch.float64, device=dev) for v in (X, Uc, Y))
    Phi = torch.cat([X, Uc, torch.ones(len(X), 1, device=dev, dtype=torch.float64)], 1)
    G = Phi.T @ Phi
    lam = ridge * torch.trace(G) / G.shape[0]
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev, dtype=torch.float64), Phi.T @ Y)
    d = X.shape[1]
    return {"A": W[:d].T.float(), "B": W[d:-1].T.float(), "c": W[-1].float()}


class MLPStep(nn.Module):
    def __init__(self, d, du, h=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + du, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, s, u):
        return s + self.f(torch.cat([s, u], -1))


class KoopmanAE(nn.Module):
    """State-inclusive Koopman model: z = [s, g(s)], linear K on z with control B u."""

    def __init__(self, d, du, m=128, h=512):
        super().__init__()
        self.d = d
        self.g = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, m))
        self.K = nn.Parameter(torch.eye(d + m) + 0.001 * torch.randn(d + m, d + m))
        self.B = nn.Parameter(torch.zeros(d + m, du))
        self.c = nn.Parameter(torch.zeros(d + m))

    def lift(self, s):
        return torch.cat([s, self.g(s)], -1)

    def step(self, z, u):
        return z @ self.K.T + u @ self.B.T + self.c


def train_mlp(Xs, Us, Ys, dev, epochs, log):
    d, du = Xs.shape[1], Us.shape[1]
    net = MLPStep(d, du).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    X, U, Y = (torch.as_tensor(v, device=dev) for v in (Xs, Us, Ys))
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        tot = 0
        for b in perm.split(4096):
            loss = ((net(X[b], U[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        log(f"    [mlp] ep {ep} mse {tot / len(X):.5f}")
    return net.eval()


def windows(S, U, ex, W, n, t_max, rng):
    N, T, P, d = S.shape
    t_max = min(t_max, T) - W
    e = rng.choice(ex, size=n)
    t0 = rng.integers(0, max(1, t_max + 1), size=n)
    p = rng.integers(0, P, size=n)
    ix_t = t0[:, None] + np.arange(W)[None]
    return S[e[:, None], ix_t, p[:, None]], U[e, p]


def train_kae(S, U, ex, t_max, dev, epochs, horizon, log, seed):
    rng = np.random.default_rng(seed)
    d, du = S.shape[-1], U.shape[-1]
    net = KoopmanAE(d, du).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    W = horizon + 1
    for ep in range(epochs):
        tot = 0
        for _ in range(200):
            sw, uw = windows(S, U, ex, W, 1024, t_max, rng)
            s = torch.as_tensor(sw, device=dev)
            u = torch.as_tensor(uw, device=dev)
            z = net.lift(s[:, 0])
            target_z = net.lift(s[:, 1:])
            lp, ll = 0.0, 0.0
            for k in range(horizon):
                z = net.step(z, u)
                lp = lp + ((z[:, :d] - s[:, k + 1]) ** 2).mean()
                ll = ll + ((z[:, d:] - target_z[:, k, d:]) ** 2).mean()
            gz = net.g(s.flatten(0, 1))
            var = torch.relu(1.0 - gz.std(0)).mean()
            loss = (lp + 0.5 * ll) / horizon + 0.1 * var
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        log(f"    [koopman-ae] ep {ep} loss {tot / 200:.5f}")
    return net.eval()


@torch.no_grad()
def skill_table(S, U, ex, starts, models, dev, max_k):
    """For held-out examples ex and start times in `starts`, MSE of k-step predictions vs persistence."""
    N, T, P, d = S.shape
    res = {name: {k: [0.0, 0.0] for k in KS if k <= max_k} for name in models}
    for e0 in range(0, len(ex), 64):
        eb = ex[e0: e0 + 64]
        s_all = torch.as_tensor(S[eb], device=dev)  # [b,T,P,d]
        u = torch.as_tensor(U[eb], device=dev)      # [b,P,du]
        for t in starts:
            if t + 1 >= T:
                continue
            s0 = s_all[:, t]
            for name, fn in models.items():
                preds = fn(s0, u, min(max_k, T - 1 - t))
                for k in res[name]:
                    if t + k >= T:
                        continue
                    tgt = s_all[:, t + k]
                    res[name][k][0] += ((preds[k - 1] - tgt) ** 2).sum().item()
                    res[name][k][1] += ((s0 - tgt) ** 2).sum().item()
    return {name: {k: (1 - v[0] / v[1]) if v[1] > 0 else float("nan") for k, v in r.items()} for name, r in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/arc1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pairs", type=int, default=1_000_000)
    ap.add_argument("--mlp_epochs", type=int, default=4)
    ap.add_argument("--kae_epochs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device
    logf = open(os.path.join(a.out, "diagnose.log"), "w")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")

    z = np.load(os.path.join(a.out, "traj.npz"), allow_pickle=True)
    summ = json.load(open(os.path.join(a.out, "rollout_summary.json")))
    S = np.concatenate([z["ZH"], z["ZL"]], -1).astype(np.float32)  # [N,T,P,2d]
    U = z["U"].astype(np.float32)
    correct, qhalt, relstep, pixacc = z["correct"], z["qhalt"], z["relstep"], z["pixacc"]
    N, T, P, d = S.shape
    Hc, Htr = summ["H_cycles"], summ["halt_max_steps"]
    A = correct.shape[1]
    T_tr = min(Htr, A) * Hc  # states inside the trained depth
    tr, te = split_by_puzzle(z["base"], a.seed)
    log(f"[diag] N={N} T={T} P={P} d={d}  trained depth {Htr} ACT steps ({T_tr} states)  train/test puzzles split {len(tr)}/{len(te)} examples")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # ---------- fit on train puzzles, within trained depth ----------
    X, Uc, Y = pairs(S, U, tr, T_tr, a.pairs, rng)
    dmd = fit_dmdc(X, Uc, Y, dev)
    mlp = train_mlp(X, Uc, Y, dev, a.mlp_epochs, log)
    kae = train_kae(S, U, tr, T_tr, dev, a.kae_epochs, a.horizon, log, a.seed)

    def roll_dmd(s0, u, K):
        out, s = [], s0
        for _ in range(K):
            s = s @ dmd["A"].T + u @ dmd["B"].T + dmd["c"]
            out.append(s)
        return out

    def roll_mlp(s0, u, K):
        out, s = [], s0
        for _ in range(K):
            s = mlp(s, u)
            out.append(s)
        return out

    def roll_kae(period):
        def f(s0, u, K):
            out, zz = [], kae.lift(s0)
            for k in range(K):
                zz = kae.step(zz, u)
                out.append(zz[..., :d])
                if period and (k + 1) % period == 0:
                    zz = kae.lift(zz[..., :d])  # periodic re-encoding (course correction)
            return out
        return f

    models = {"DMDc": roll_dmd, "MLP": roll_mlp, "KoopmanAE_latent": roll_kae(None),
              "KoopmanAE_reenc3": roll_kae(3), "KoopmanAE_reenc1": roll_kae(1)}
    buckets = {"act_step_1": list(range(0, Hc)),
               "trained_depth": list(range(Hc, T_tr - 1, 2)),
               "past_trained_depth": list(range(T_tr, T - 1, 2))}
    skills = {}
    for bname, starts in buckets.items():
        if not starts:
            continue
        skills[bname] = skill_table(S, U, te, starts, models, dev, max(KS))
        log(f"[skill vs persistence] {bname}: " + "  ".join(
            f"{m}=" + "/".join(f"{v:.3f}" for v in r.values()) for m, r in skills[bname].items()) + f"   (k={list(KS)})")

    # ---------- spectrum ----------
    Aop = dmd["A"].double().cpu()
    ev = torch.linalg.eigvals(Aop).numpy()
    Kev = torch.linalg.eigvals(kae.K.detach().double().cpu()).numpy()
    spec = {"dmd_spectral_radius": float(np.abs(ev).max()),
            "dmd_frac_abs_gt_0.9": float((np.abs(ev) > 0.9).mean()),
            "dmd_frac_abs_gt_0.99": float((np.abs(ev) > 0.99).mean()),
            "kae_spectral_radius": float(np.abs(Kev).max()),
            "kae_frac_abs_gt_0.99": float((np.abs(Kev) > 0.99).mean())}
    log("[spectrum]", json.dumps({k: round(v, 4) for k, v in spec.items()}))

    # slow subspace (|lambda| > 0.95) for the feature below and for explore.py
    evals, evecs = np.linalg.eig(Aop.numpy())
    order = np.argsort(-np.abs(evals))
    np.savez(os.path.join(a.out, "dmd.npz"), A=dmd["A"].cpu().numpy(), B=dmd["B"].cpu().numpy(),
             c=dmd["c"].cpu().numpy(), evals=evals[order], evecs=evecs[:, order])
    slow = evecs[:, order][:, np.abs(evals[order]) > 0.95]
    Q = np.linalg.qr(np.concatenate([slow.real, slow.imag], 1))[0] if slow.shape[1] else np.zeros((d, 0))

    # ---------- does early dynamics predict final correctness? ----------
    y = correct[:, min(Htr, A) - 1]
    early = {}
    for step in (1, 2):
        t = step * Hc - 1
        if t >= T:
            continue
        feats = {"q_halt (model's own)": qhalt[:, step - 1],
                 "rel step size": -relstep[:, t],
                 "DMD one-step residual": None,
                 "slow-mode share of update": None}
        with torch.no_grad():
            s_t = torch.as_tensor(S[:, t], device=dev)
            s_n = torch.as_tensor(S[:, t + 1], device=dev) if t + 1 < T else None
            u = torch.as_tensor(U, device=dev)
            if s_n is not None:
                pred = s_t @ dmd["A"].T + u @ dmd["B"].T + dmd["c"]
                feats["DMD one-step residual"] = -((pred - s_n) ** 2).sum(-1).mean(-1).cpu().numpy()
            if Q.shape[1] and t >= 1:
                delta = (s_t - torch.as_tensor(S[:, t - 1], device=dev)).cpu().numpy()
                proj = delta @ Q
                feats["slow-mode share of update"] = ((proj ** 2).sum(-1) / ((delta ** 2).sum(-1) + 1e-9)).mean(-1)
        from sklearn.linear_model import LogisticRegression
        pooled = S[:, t].mean(1)
        mu, sd = pooled[tr].mean(0), pooled[tr].std(0) + 1e-6
        clf = LogisticRegression(C=0.1, max_iter=2000).fit((pooled[tr] - mu) / sd, y[tr]) if 0 < y[tr].mean() < 1 else None
        res = {k: auroc(y[te], v[te]) for k, v in feats.items() if v is not None}
        res["linear probe on pooled state"] = auroc(y[te], clf.decision_function((pooled[te] - mu) / sd)) if clf else float("nan")
        early[f"end_of_act_step_{step}"] = res
        log(f"[early signal] end of ACT step {step}: " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))

    acc = correct.mean(0)
    out = {"n": N, "T": T, "H_cycles": Hc, "trained_depth": Htr, "exact_acc_by_act_step": acc.tolist(),
           "pixacc_by_act_step": pixacc.mean(0).tolist(), "relstep_by_t": relstep.mean(0).tolist(),
           "skill_vs_persistence": skills, "spectrum": spec, "early_signal": early, "base_rate_correct": float(y.mean())}
    json.dump(out, open(os.path.join(a.out, "diagnose.json"), "w"), indent=1)
    write_report(a.out, out, ev, Kev)


def verdict(o):
    lines = []
    sk = o["skill_vs_persistence"].get("trained_depth") or o["skill_vs_persistence"].get("act_step_1")
    rs = np.array(o["relstep_by_t"])
    Hc = o["H_cycles"]
    late_step = float(rs[Hc * 2:].mean()) if len(rs) > Hc * 2 else float("nan")
    lines.append(f"- Mean relative update of z_H after ACT step 2: **{late_step:.4f}** per H-cycle "
                 f"({'near-frozen: recursion is shallow, little dynamics left to model' if late_step < 0.01 else 'still moving'}).")
    if sk:
        d1, m1 = sk["DMDc"][1], sk["MLP"][1]
        kl, kr = sk["KoopmanAE_latent"][6], sk["KoopmanAE_reenc3"][6]
        d6 = sk["DMDc"][6]
        lines.append(f"- One-step skill over persistence (trained depth): DMDc **{d1:.3f}**, MLP **{m1:.3f}** "
                     f"→ {'linear dynamics capture most of what is predictable' if m1 - d1 < 0.05 else 'a nonlinear gap of %.3f exists — room for a learned lift' % (m1 - d1)}.")
        lines.append(f"- 6-step skill: DMDc {d6:.3f}, Koopman-AE pure latent {kl:.3f}, with re-encoding every 3 steps {kr:.3f} "
                     f"→ {'re-encoding helps (course correction is live)' if kr - kl > 0.02 else 're-encoding adds little here'}.")
        go = (d1 > 0.3 or (m1 - d1 > 0.1 and max(kl, kr) > d6 + 0.05)) and late_step >= 0.01
        lines.append(f"\n**Gate: {'GO — dynamics are rich and predictable enough for Koopman-guided exploration / reweighting' if go else 'CAUTION — see bullets; Koopman methods have little to act on unless the recursion is still moving and predictable'}.**")
    return lines


def write_report(out_dir, o, ev, Kev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    acc = np.array(o["exact_acc_by_act_step"])
    ax[0].plot(np.arange(1, len(acc) + 1), acc, marker="o", ms=3)
    ax[0].axvline(o["trained_depth"], ls="--", c="gray")
    ax[0].set(title="Exact accuracy vs ACT step", xlabel="ACT step (dashed = trained depth)", ylabel="exact match")
    rs = np.array(o["relstep_by_t"])
    ax[1].semilogy(np.arange(1, len(rs)), np.maximum(rs[1:], 1e-6))
    ax[1].axvline(o["trained_depth"] * o["H_cycles"], ls="--", c="gray")
    ax[1].set(title="Relative z_H update per H-cycle", xlabel="H-cycle t")
    sk = o["skill_vs_persistence"].get("trained_depth") or next(iter(o["skill_vs_persistence"].values()))
    for m, r in sk.items():
        ax[2].plot(list(map(int, r.keys())), list(r.values()), marker="o", label=m)
    ax[2].axhline(0, c="k", lw=0.8)
    ax[2].set(title="k-step skill over persistence (held-out)", xlabel="k", ylabel="1 - MSE/MSE_persist")
    ax[2].legend(fontsize=7)
    th = np.linspace(0, 2 * np.pi, 200)
    ax[3].plot(np.cos(th), np.sin(th), c="gray", lw=0.8)
    ax[3].scatter(ev.real, ev.imag, s=6, label="DMDc A")
    ax[3].scatter(Kev.real, Kev.imag, s=6, alpha=0.5, label="Koopman-AE K")
    ax[3].set(title="Spectra", aspect="equal")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diagnose.png"), dpi=130)

    L = ["# TRM recursion dynamics — viability report", "",
         f"N={o['n']} test examples, T={o['T']} H-cycle states, trained depth {o['trained_depth']} ACT steps. "
         f"Base rate correct at trained depth: {o['base_rate_correct']:.3f}.", "", "![](diagnose.png)", "", "## Verdict", ""]
    L += verdict(o)
    L += ["", "## k-step skill over persistence (held-out puzzles)", "", "| bucket | model | " + " | ".join(f"k={k}" for k in KS) + " |",
          "|---|---|" + "---|" * len(KS)]
    for b, r in o["skill_vs_persistence"].items():
        for m, v in r.items():
            L.append(f"| {b} | {m} | " + " | ".join(f"{v.get(k, v.get(str(k), float('nan'))):.3f}" for k in KS) + " |")
    L += ["", "## Spectrum", "", "```", json.dumps(o["spectrum"], indent=1), "```", "",
          "## Does early dynamics predict final correctness? (AUROC, held-out)", ""]
    for st, r in o["early_signal"].items():
        L.append(f"- {st}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    L += ["", "Reference: Karakus et al. (NeurIPS 2026 wkshp) report AUROC 0.895 from the first outer-step latent on ARC-AGI-1.",
          "", "## Accuracy by ACT step", "", ", ".join(f"{i + 1}:{v:.3f}" for i, v in enumerate(o["exact_acc_by_act_step"]))]
    open(os.path.join(out_dir, "REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

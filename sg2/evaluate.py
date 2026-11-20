"""Evaluate one seed: sample every arm on the same test trajectories; physics metrics by collision bucket and horizon.
Arms
  block48 (N=10/50/200 denoising steps), block8, block1      the paper's three factorisations (+ step sweep)
  koopdiff_1                                                  one parallel pass conditioned on the Koopman rollout
  koopdiff_per{P}, koopdiff_evt{P}                            P serial passes: periodic / event-triggered re-encoding
  featdiff_1, featdiff_per{P}                                 capacity-matched control without linear dynamics
Event trigger: after each pass, residual r_t = ||phi(generated window ending at t) - Lambda^(t-s+1) z_s|| / ||.||
for all t in parallel; re-encode at the first t with r_t > tau and regenerate the frames after it. tau is calibrated
on validation trajectories (binary search) so that the mean number of passes hits P.
Serial cost reported as passes (sequential parallel-generation rounds) and as network evaluations (passes x N).
Metrics: local error (paper's Delta x^(5), true simulator from the generated state), energy error, overlap rate
(balls interpenetrating), global error.   -> <out>/eval_s<seed>.json"""
import argparse, json, math, os, time

import torch

from sg.sim import simulate
from .nets import (CTX, FD, H, Denoiser, FeatControl, Koop, block_bounds, build_mask, denorm_pos, norm_pos,
                   prefix_bounds)
from .train import windows_ending

BUCKETS = [(0, 2), (3, 4), (5, 6), (7, 99)]
QUARTERS = [(6, 17), (18, 29), (30, 41), (42, 47)]
R = 0.07


def local_error_t(full, k=5):
    """full [B, 53, 5, 2] positions -> [B, len(ts)] local error at future frames t = k+1 .. 47 (future index)."""
    B = full.shape[0]
    ts = list(range(CTX + k + 1, CTX + H))
    p0 = torch.stack([full[:, t - k] for t in ts], 1).reshape(-1, 5, 2).clamp(R, 1 - R)
    v0 = torch.stack([full[:, t - k] - full[:, t - k - 1] for t in ts], 1).reshape(-1, 5, 2)
    sim, _ = simulate(p0.shape[0], k + 1, init=(p0, v0), device=full.device)
    tgt = torch.stack([full[:, t] for t in ts], 1).reshape(-1, 5, 2)
    return (sim[:, k] - tgt).norm(dim=-1).mean(-1).view(B, len(ts))


def energy_error(full):
    v = full[:, CTX - 1:] - full[:, CTX - 2:-1]
    ke = (v ** 2).sum((-1, -2))
    return ((ke[:, 1:] - ke[:, :1]).abs() / ke[:, :1].clamp(min=1e-9)).mean(1)


def overlap_rate(full):
    iu = torch.triu_indices(5, 5, 1, device=full.device)
    d = (full[:, CTX:, iu[0]] - full[:, CTX:, iu[1]]).norm(dim=-1)            # [B, 48, 10]
    return (d < 0.8 * 2 * R).any(-1).float().mean(1)


# ------------------------------------------------------------------ samplers (positions are normalised inside)
@torch.no_grad()
def sample_block(model, ctx, k, N):
    B, dev = ctx.shape[0], ctx.device
    gen = torch.zeros(B, H, FD, device=dev)
    mask = build_mask(*block_bounds(k, B, dev))
    for b0 in range(0, H, k):
        b1 = min(H, b0 + k)
        x = torch.randn(B, b1 - b0, FD, device=dev)
        for i in range(N):
            noisy = gen.clone(); noisy[:, b0:b1] = x
            tau = torch.full((B, H), i / N, device=dev)
            x = x + model(ctx, gen, noisy, tau, mask)[:, b0:b1] / N
        gen[:, b0:b1] = x
    return gen, torch.full((B,), float(math.ceil(H / k)), device=dev)


@torch.no_grad()
def gen_after(model, ctx, gen, s, cond, N):
    """regenerate frames t >= s[b] in one parallel pass (prefix of generated frames given)."""
    B, dev = ctx.shape[0], ctx.device
    mask = build_mask(*prefix_bounds(s))
    keep = torch.arange(H, device=dev)[None] < s[:, None]
    x = torch.randn(B, H, FD, device=dev)
    for i in range(N):
        noisy = torch.where(keep[..., None], gen, x)
        tau = torch.full((B, H), i / N, device=dev)
        x = x + model(ctx, gen, noisy, tau, mask, cond) / N
    return torch.where(keep[..., None], gen, x)


def make_cond(kind, aux, ctx, gen, s):
    full = torch.cat([ctx, gen], 1)
    win = windows_ending(full, CTX + s - 1)
    stp = (torch.arange(H, device=ctx.device)[None] - s[:, None] + 1).clamp(min=0)
    if kind == "koop":
        z = aux.encode(win)
        return aux.roll(z, stp), z
    return aux.features(win, stp), None


@torch.no_grad()
def sample_periodic(model, aux, kind, ctx, P, N):
    B, dev = ctx.shape[0], ctx.device
    gen = torch.zeros(B, H, FD, device=dev)
    for i in range(P):
        s = torch.full((B,), int(round(i * H / P)), dtype=torch.long, device=dev)
        cond, _ = make_cond(kind, aux, ctx, gen, s)
        gen = gen_after(model, ctx, gen, s, cond, N)
    return gen, torch.full((B,), float(P), device=dev)


@torch.no_grad()
def sample_event(model, koop, ctx, tau_thr, N, max_passes=24):
    B, dev = ctx.shape[0], ctx.device
    tH = torch.arange(H, device=dev)
    gen = torch.zeros(B, H, FD, device=dev)
    s = torch.zeros(B, dtype=torch.long, device=dev)
    cond, z = make_cond("koop", koop, ctx, gen, s)
    gen = gen_after(model, ctx, gen, s, cond, N)
    passes = torch.ones(B, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    for _ in range(max_passes - 1):
        full = torch.cat([ctx, gen], 1)
        wins = full.unfold(1, CTX, 1)[:, 1:H + 1].permute(0, 1, 3, 2)             # windows ending at future t
        zw = koop.encode(wins.reshape(-1, CTX, FD)).view(B, H, -1)
        zl = koop.roll(z, (tH[None] - s[:, None] + 1).clamp(min=0))
        r = (zw - zl).norm(dim=-1) / zl.norm(dim=-1).clamp(min=1e-6)
        valid = (tH[None] >= s[:, None] + 1) & (tH[None] <= H - 2) & active[:, None]
        exc = (r > tau_thr) & valid
        hit = exc.any(1)
        if not hit.any():
            break
        t_hit = exc.float().argmax(1)
        s_new = torch.where(hit, t_hit + 1, s)
        cond_new, z_new = make_cond("koop", koop, ctx, gen, s_new)
        regen = gen_after(model, ctx, gen, s_new, cond_new, N)
        gen = torch.where(hit[:, None, None], regen, gen)
        z = torch.where(hit[:, None], z_new, z)
        s = s_new
        passes = passes + hit.float()
        active = active & hit
    return gen, passes


def calibrate(model, koop, ctx_val, target, N, iters=9):
    lo, hi = math.log(1e-4), math.log(10.0)
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        _, p = sample_event(model, koop, ctx_val, math.exp(mid), N)
        mp = float(p.mean())
        if best is None or abs(mp - target) < abs(best[1] - target):
            best = (math.exp(mid), mp)
        if mp > target:
            lo = mid          # too many passes -> raise threshold
        else:
            hi = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="ckpt_sg2")
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--N_sweep", default="10,200")
    ap.add_argument("--budgets", default="2,4,8,16")
    ap.add_argument("--bs", type=int, default=500)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", default="results_sg2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev, t0 = a.device, time.time()
    torch.manual_seed(100 + a.seed)
    X, C = simulate(a.n_test, CTX + H, seed=777, device=dev)
    Xv, _ = simulate(a.n_val, CTX + H, seed=778, device=dev)
    ctx_all = norm_pos(X.reshape(a.n_test, CTX + H, FD))[:, :CTX]
    ctx_val = norm_pos(Xv.reshape(a.n_val, CTX + H, FD))[:, :CTX]
    ncoll = C[:, CTX:].sum(1)
    out = {"seed": a.seed, "N": a.N, "buckets": BUCKETS, "quarters": QUARTERS,
           "bucket_counts": [int(((ncoll >= lo) & (ncoll <= hi)).sum()) for lo, hi in BUCKETS], "arms": {}, "calib": {}}
    gt = X
    out["floor"] = {"local": float(local_error_t(gt).mean()), "energy": float(energy_error(gt).mean()),
                    "overlap": float(overlap_rate(gt).mean())}

    def load(name, cls):
        p = os.path.join(a.ckpt, f"{name}_s{a.seed}.pt")
        if not os.path.exists(p):
            print(f"  (missing {p})"); return None
        m = cls(); m.load_state_dict(torch.load(p, map_location=dev)); return m.to(dev).eval()

    def record(name, fn):
        gens, passes = [], []
        for i in range(0, a.n_test, a.bs):
            g, p = fn(ctx_all[i:i + a.bs]); gens.append(g); passes.append(p)
        gen, ps = torch.cat(gens), torch.cat(passes)
        full = torch.cat([X[:, :CTX], denorm_pos(gen).view(-1, H, 5, 2)], 1)
        lt = local_error_t(full)
        le, ke, ov = lt.mean(1), energy_error(full), overlap_rate(full)
        ge = (full[:, CTX:] - X[:, CTX:]).norm(dim=-1).mean((1, 2))
        e = {"passes": float(ps.mean()), "local": float(le.mean()), "energy": float(ke.mean()), "overlap": float(ov.mean()),
             "global": float(ge.mean()), "by_bucket": [], "by_quarter": []}
        for lo, hi in BUCKETS:
            m = (ncoll >= lo) & (ncoll <= hi)
            e["by_bucket"].append({k: float(v[m].mean()) for k, v in (("local", le), ("energy", ke), ("overlap", ov))})
        for q0, q1 in QUARTERS:
            e["by_quarter"].append(float(lt[:, q0 - 6: q1 - 6 + 1].mean()))
        out["arms"][name] = e
        print(f"[s{a.seed}] {name:16s} passes {e['passes']:5.1f}  local {e['local']:.4f}  energy {e['energy']:.3f}  "
              f"overlap {e['overlap']:.3f}  local by bucket " + " ".join(f"{b['local']:.4f}" for b in e["by_bucket"]) +
              f"  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(a.out, f"eval_s{a.seed}.json"), "w"), indent=1)

    os.makedirs(a.out, exist_ok=True)
    D = lambda cond=0: (lambda: Denoiser(a.d, a.layers, cond_dim=cond))
    for k in (48, 8, 1):
        m = load(f"block{k}", D())
        if m is not None:
            record(f"block{k}", lambda c, m=m, k=k: sample_block(m, c, k, a.N))
            if k == 48:
                for Ns in [int(x) for x in a.N_sweep.split(",") if x]:
                    record(f"block48_N{Ns}", lambda c, m=m, Ns=Ns: sample_block(m, c, 48, Ns))
    budgets = [int(x) for x in a.budgets.split(",")]
    koop = load("koopbase", Koop)
    kd = load("koopdiff", D(128))
    if kd is not None and koop is not None:
        record("koopdiff_1", lambda c: sample_periodic(kd, koop, "koop", c, 1, a.N))
        for P in budgets:
            record(f"koopdiff_per{P}", lambda c, P=P: sample_periodic(kd, koop, "koop", c, P, a.N))
        for P in budgets:
            thr, mp = calibrate(kd, koop, ctx_val, P, a.N)
            out["calib"][str(P)] = {"tau": thr, "val_passes": mp}
            print(f"[s{a.seed}] calibrated event threshold for {P} passes: tau={thr:.4g} (val passes {mp:.2f})", flush=True)
            record(f"koopdiff_evt{P}", lambda c, thr=thr: sample_event(kd, koop, c, thr, a.N))

    def make_feat():
        m = Denoiser(a.d, a.layers, cond_dim=128); m.feat = FeatControl(); return m
    fd = load("featdiff", make_feat)
    if fd is not None:
        record("featdiff_1", lambda c: sample_periodic(fd, fd.feat, "feat", c, 1, a.N))
        for P in budgets:
            record(f"featdiff_per{P}", lambda c, P=P: sample_periodic(fd, fd.feat, "feat", c, P, a.N))
    print(f"[s{a.seed}] done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

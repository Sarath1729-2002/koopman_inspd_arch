"""Evaluate all arms on fresh trajectories. Metrics, bucketed by # ball-ball collisions in the true 48-frame horizon:
  local error  (the paper's Delta x^(5)): from the PREDICTED state at t-5 (velocity by finite difference of predicted
               frames t-6, t-5), run the true simulator 5 frames and compare with the predicted frame t. Measures
               physical consistency without penalising chaotic divergence.
  global error mean distance to the true positions.
  passes       serial passes (parallel 1, ar 48, koop 1 + #re-encodings).   -> <out>/eval.json"""
import argparse, json, os

import torch

from .models import AR, CTX, FD, H, Koop, Parallel
from .sim import simulate

BUCKETS = [(0, 2), (3, 4), (5, 6), (7, 99)]


def local_error(pred_full, k=5):
    """pred_full [B, 53, 5, 2] (context + predicted). Returns per-sample mean local error over t = CTX+k+1 .. end."""
    B = pred_full.shape[0]
    ts = list(range(CTX + k + 1, CTX + H))
    p0 = torch.stack([pred_full[:, t - k] for t in ts], 1).reshape(-1, 5, 2).clamp(0.07, 0.93)
    v0 = torch.stack([pred_full[:, t - k] - pred_full[:, t - k - 1] for t in ts], 1).reshape(-1, 5, 2)
    sim, _ = simulate(p0.shape[0], k + 1, init=(p0, v0), device=pred_full.device)
    tgt = torch.stack([pred_full[:, t] for t in ts], 1).reshape(-1, 5, 2)
    return (sim[:, k] - tgt).norm(dim=-1).mean(-1).view(B, len(ts)).mean(1)


def energy_error(pred_full):
    """relative error of total kinetic energy (finite-difference velocities) over the horizon vs the context's.
    Elastic collisions conserve it; slowing/blurring predictions (regression to the mean) are penalised."""
    v = pred_full[:, CTX - 1:] - pred_full[:, CTX - 2:-1]                 # [B, 49, 5, 2], first = context velocity
    ke = (v ** 2).sum((-1, -2))
    return ((ke[:, 1:] - ke[:, :1]).abs() / ke[:, :1].clamp(min=1e-9)).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_sg")
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--taus", default="0.05,0.1,0.2,0.35,0.5,0.8")
    ap.add_argument("--out", default="results_sg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    X, C = simulate(a.n_test, CTX + H, seed=777, device=dev)
    Xf = X.reshape(a.n_test, CTX + H, FD)
    ncoll = C[:, CTX:].sum(1)
    ctx = Xf[:, :CTX]
    gt_full = X.view(a.n_test, CTX + H, 5, 2)
    floor = {"local": float(local_error(gt_full).mean()), "energy": float(energy_error(gt_full).mean())}
    print(f"metric floor (true trajectories): local {floor['local']:.5f} energy {floor['energy']:.3f}", flush=True)
    res = {"floor": floor, "buckets": BUCKETS, "bucket_counts": [int(((ncoll >= lo) & (ncoll <= hi)).sum()) for lo, hi in BUCKETS], "arms": {}}

    def record(name, pred, passes):
        full = torch.cat([ctx, pred], 1).view(-1, CTX + H, 5, 2)
        le = local_error(full)
        ge = (full[:, CTX:] - X[:, CTX:]).norm(dim=-1).mean((1, 2))
        ke = energy_error(full)
        entry = {"passes": float(passes.float().mean()), "local": float(le.mean()), "energy": float(ke.mean()),
                 "global": float(ge.mean()), "by_bucket": []}
        for lo, hi in BUCKETS:
            sel = (ncoll >= lo) & (ncoll <= hi)
            entry["by_bucket"].append({"local": float(le[sel].mean()), "energy": float(ke[sel].mean()),
                                       "global": float(ge[sel].mean()), "passes": float(passes.float()[sel].mean())})
        res["arms"][name] = entry
        print(f"{name:14s} passes {entry['passes']:5.1f}  local {entry['local']:.5f}  energy {entry['energy']:.3f}  global {entry['global']:.4f}  "
              "local by bucket " + " ".join(f"{b['local']:.5f}" for b in entry["by_bucket"]), flush=True)

    load = lambda cls, n: (lambda m: (m.load_state_dict(torch.load(os.path.join(a.ckpt, n + ".pt"), map_location=dev)), m.eval())[1])(cls().to(dev))
    with torch.no_grad():
        bs = 500
        def batched(f):
            outs = [f(ctx[i:i + bs]) for i in range(0, len(ctx), bs)]
            return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
        if os.path.exists(os.path.join(a.ckpt, "parallel.pt")):
            m = load(Parallel, "parallel"); record("parallel", *batched(lambda c: (m(c), torch.ones(len(c), device=dev))))
        if os.path.exists(os.path.join(a.ckpt, "ar.pt")):
            m = load(AR, "ar"); record("ar", *batched(lambda c: (m.rollout(c), torch.full((len(c),), float(H), device=dev))))
        if os.path.exists(os.path.join(a.ckpt, "koop1.pt")):
            m = load(Koop, "koop1"); record("koop_oneshot", *batched(lambda c: m.rollout(c)))
        if os.path.exists(os.path.join(a.ckpt, "koop.pt")):
            m = load(Koop, "koop")
            record("koop_none", *batched(lambda c: m.rollout(c)))
            for k in (16, 8, 4, 2):
                record(f"koop_every{k}", *batched(lambda c, k=k: m.rollout(c, reenc_at=list(range(k, H, k)))))
            for tau in [float(x) for x in a.taus.split(",")]:
                record(f"koop_event{tau}", *batched(lambda c, tau=tau: m.rollout(c, trigger_tau=tau)))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

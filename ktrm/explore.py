"""Stage 3: PTRM-style stochastic exploration, with the noise DIRECTION as the only variable.

PTRM (Sghaier, Parviz, Jolicoeur-Martineau 2026) injects isotropic Gaussian noise into the latent
at every recursion step, runs R rollouts, and keeps the one whose Q-head halt logit is highest.
Here every variant injects noise of the SAME per-token norm (sigma * ||z_token||) after every
H-cycle into z_H and z_L; only its direction differs:

  iso_full    isotropic in the full hidden space                       (= PTRM)
  iso_pca     isotropic inside the PCA subspace used for the dynamics  (control for the subspace)
  koop_slow   inside the span of DMD modes with |lambda| > tau          (hypothesis: perturb the
              slow, still-uncommitted directions)
  rand_sub    inside a random subspace of the same rank as koop_slow    (control for rank)

Reports, per variant and sigma: pass@1 with Q-head selection (PTRM's rule), pass@R (oracle any),
and the deterministic baseline. Needs runs/<name>/{pca.npz,dmd.npz} from rollout + diagnose.
"""
import json
import os
import time

import numpy as np
import torch

from .common import base_parser, setup
from .model import exact_correct, rollout


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n_examples", type=int, default=512)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--sigmas", default="0.05,0.1,0.2")
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--max_rank", type=int, default=32)
    ap.add_argument("--n_act", type=int, default=None, help="default: trained halt_max_steps")
    ap.add_argument("--variants", default="iso_full,iso_pca,koop_slow,rand_sub")
    ap.add_argument("--canonical_only", action="store_true")
    a = ap.parse_args()
    model, data, info = setup(a)
    dev = a.device
    A = a.n_act or model.config.halt_max_steps
    pca = np.load(os.path.join(a.out, "pca.npz"))
    dmd = np.load(os.path.join(a.out, "dmd.npz"))
    WH = torch.as_tensor(pca["WH"], device=dev).float()  # [D, dH]
    WL = torch.as_tensor(pca["WL"], device=dev).float()
    dH = WH.shape[1]
    ev, V = dmd["evals"], dmd["evecs"]  # sorted by |lambda| desc, in [zH_pca, zL_pca] coordinates
    slow = V[:, np.abs(ev) > a.tau]
    Qs = np.linalg.qr(np.concatenate([slow.real, slow.imag], 1))[0][:, : a.max_rank] if slow.shape[1] else None
    rank = 0 if Qs is None else Qs.shape[1]
    print(f"[explore] {int((np.abs(ev) > a.tau).sum())} DMD modes with |lambda|>{a.tau} -> slow subspace rank {rank}")
    rng = np.random.default_rng(a.seed)
    Qr = np.linalg.qr(rng.normal(size=(V.shape[0], max(rank, 1))))[0]
    bases = {"iso_pca": None, "koop_slow": Qs, "rand_sub": Qr}
    bases = {k: (torch.as_tensor(v, device=dev).float() if v is not None else None) for k, v in bases.items()}

    def make_noise(variant, sigma):
        def fn(t, zH, zL):
            B, L, D = zH.shape
            if variant == "iso_full":
                nH, nL = torch.randn_like(zH, dtype=torch.float32), torch.randn_like(zL, dtype=torch.float32)
            else:
                if variant == "iso_pca":
                    e = torch.randn(B, L, 2 * dH, device=dev)
                else:
                    Q = bases[variant]
                    e = torch.randn(B, L, Q.shape[1], device=dev) @ Q.T
                nH, nL = e[..., :dH] @ WH.T, e[..., dH:] @ WL.T
            scale = lambda n, z: n * (sigma * z.float().norm(dim=-1, keepdim=True) / n.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            return (zH.float() + scale(nH, zH)).to(zH.dtype), (zL.float() + scale(nL, zL)).to(zL.dtype)
        return fn

    idx = data.sample(a.n_examples, a.seed + 1, a.canonical_only)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    variants = [v for v in a.variants.split(",") if not (v in ("koop_slow", "rand_sub") and rank == 0)]
    res = {"deterministic": {"pass1": 0.0}}
    counts = {}
    t0 = time.time()
    for s in range(0, len(idx), a.batch_size):
        batch = data.batch(idx[s: s + a.batch_size], dev)
        _, outs = rollout(model, batch, A)
        c, _ = exact_correct(outs[-1][0], batch["labels"])
        res["deterministic"]["pass1"] += c.sum().item()
        for v in variants:
            for sg in sigmas:
                key = f"{v}@{sg}"
                cs, qs = [], []
                for r in range(a.rollouts):
                    torch.manual_seed(a.seed * 1000 + r)
                    _, outs = rollout(model, batch, A, noise=make_noise(v, sg))
                    cr, _ = exact_correct(outs[-1][0], batch["labels"])
                    cs.append(cr)
                    qs.append(outs[-1][1])
                cs, qs = torch.stack(cs, 1), torch.stack(qs, 1)  # [B, R]
                pick = cs[torch.arange(len(cs)), qs.argmax(1)]
                acc = counts.setdefault(key, [0, 0, 0])
                acc[0] += pick.sum().item()
                acc[1] += cs.any(1).sum().item()
                acc[2] += cs.float().mean(1).sum().item()
        print(f"  {min(len(idx), s + a.batch_size)}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    n = len(idx)
    res["deterministic"]["pass1"] /= n
    for key, (p1, pR, mean) in counts.items():
        res[key] = {"pass1_qselect": p1 / n, f"pass@{a.rollouts}_oracle": pR / n, "mean_single_rollout": mean / n}
    out = {"n_examples": n, "rollouts": a.rollouts, "n_act": A, "slow_rank": rank, "tau": a.tau, "results": res}
    json.dump(out, open(os.path.join(a.out, "explore.json"), "w"), indent=1)
    det = res["deterministic"]["pass1"]
    lines = [f"# Exploration: noise direction (n={n}, R={a.rollouts}, slow rank {rank})", "",
             f"Deterministic pass@1: **{det:.4f}**", "",
             "| variant | sigma | pass@1 (Q-select) | Δ vs det | pass@R oracle | mean single rollout |", "|---|---|---|---|---|---|"]
    for key, r in sorted(res.items()):
        if key == "deterministic":
            continue
        v, sg = key.split("@")
        lines.append(f"| {v} | {sg} | {r['pass1_qselect']:.4f} | {r['pass1_qselect'] - det:+.4f} | "
                     f"{r[f'pass@{a.rollouts}_oracle']:.4f} | {r['mean_single_rollout']:.4f} |")
    lines += ["", "Read: koop_slow must beat iso_full (PTRM) AND rand_sub at matched sigma for the Koopman-mode idea to survive."]
    open(os.path.join(a.out, "EXPLORE.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

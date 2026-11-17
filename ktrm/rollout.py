"""Stage 1: run a trained TRM on ARC test examples for MORE ACT steps than it was trained with,
and record its recursion trajectory.

Saved to <out>/traj.npz:
  ZH, ZL   [N, T, P, d]  z_H / z_L after every H-cycle (T = n_act * H_cycles), PCA-projected,
                         at P token positions: position 0 (read by the Q head) + n_tok output cells
  U        [N, P, du]    input-embedding (control) at the same positions, PCA-projected
  correct  [N, A]        exact-match of the decoded grid after every ACT step
  pixacc   [N, A]        cell accuracy after every ACT step
  qhalt    [N, A]        Q-head halt logit after every ACT step
  relstep  [N, T]        ||z_H(t) - z_H(t-1)|| / ||z_H(t-1)|| over ALL tokens (full 512-d, not PCA)
  base     [N]           original puzzle id (for leakage-free splits)
PCA bases (fit on the first batch, all t) in <out>/pca.npz.
"""
import json
import os
import time

import numpy as np
import torch

from .common import base_parser, setup
from .model import exact_correct, rollout


def pick_positions(labels, n_tok, emb_len, gen):
    """Token 0 (Q-head position) + n_tok random cells inside the target grid."""
    B = labels.shape[0]
    pos = torch.zeros(B, n_tok + 1, dtype=torch.long, device=labels.device)
    for b in range(B):
        valid = torch.nonzero(labels[b] != -100).flatten()
        if len(valid) == 0:
            valid = torch.arange(labels.shape[1], device=labels.device)
        sel = valid[torch.randint(len(valid), (n_tok,), generator=gen, device="cpu").to(valid.device)]
        pos[b, 1:] = sel + emb_len
    return pos


def gather(z, pos):
    return torch.gather(z, 1, pos[..., None].expand(-1, -1, z.shape[-1]))


def fit_pca(S, d):
    S = S.float()
    mu = S.mean(0)
    _, sv, V = torch.pca_lowrank(S - mu, q=min(d + 8, S.shape[1]), center=False, niter=6)
    evr = float((sv[:d] ** 2).sum() / ((S - mu) ** 2).sum())
    return mu, V[:, :d].contiguous(), evr


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n_examples", type=int, default=1024)
    ap.add_argument("--n_act", type=int, default=32, help="ACT steps to run (trained with halt_max_steps, usually 16)")
    ap.add_argument("--n_tok", type=int, default=24)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--u_dim", type=int, default=64)
    ap.add_argument("--canonical_only", action="store_true")
    a = ap.parse_args()
    model, data, info = setup(a)
    dev = a.device
    idx = data.sample(a.n_examples, a.seed, a.canonical_only)
    N, A, Hc = len(idx), a.n_act, model.config.H_cycles
    T, P = A * Hc, a.n_tok + 1
    gen = torch.Generator().manual_seed(a.seed)
    emb_len = int(model.inner.puzzle_emb_len)
    print(f"[rollout] {N} examples ({int(data.canonical[idx].sum())} canonical), {A} ACT steps x {Hc} H-cycles = {T} states")

    # ---- PCA calibration on the first batch (all t) ----
    b0 = idx[: a.batch_size]
    batch = data.batch(b0, dev)
    pos = pick_positions(batch["labels"], a.n_tok, emb_len, gen)
    H_s, L_s = [], []
    x, _ = rollout(model, batch, A, on_state=lambda t, zH, zL: (H_s.append(gather(zH, pos).float().flatten(0, 1).cpu()),
                                                               L_s.append(gather(zL, pos).float().flatten(0, 1).cpu())))
    muH, WH, evH = fit_pca(torch.cat(H_s), a.pca_dim)
    muL, WL, evL = fit_pca(torch.cat(L_s), a.pca_dim)
    muU, WU, evU = fit_pca(gather(x, pos).float().flatten(0, 1).cpu(), a.u_dim)
    np.savez(os.path.join(a.out, "pca.npz"), muH=muH, WH=WH, muL=muL, WL=WL, muU=muU, WU=WU)
    print(f"[pca] explained var  zH {evH:.3f}  zL {evL:.3f}  u {evU:.3f}")
    del H_s, L_s
    P_ = lambda z, mu, W: ((z.float() - mu.to(z.device)) @ W.to(z.device)).half().cpu().numpy()

    ZH = np.zeros((N, T, P, a.pca_dim), np.float16)
    ZL = np.zeros_like(ZH)
    U = np.zeros((N, P, a.u_dim), np.float16)
    correct = np.zeros((N, A), bool)
    pixacc = np.zeros((N, A), np.float32)
    qhalt = np.zeros((N, A), np.float32)
    relstep = np.zeros((N, T), np.float32)
    t0 = time.time()
    for s in range(0, N, a.batch_size):
        sl = slice(s, min(N, s + a.batch_size))
        batch = data.batch(idx[sl], dev)
        pos = pick_positions(batch["labels"], a.n_tok, emb_len, gen)
        prev = {}

        def on_state(t, zH, zL):
            ZH[sl, t] = P_(gather(zH, pos), muH, WH)
            ZL[sl, t] = P_(gather(zL, pos), muL, WL)
            if "zH" in prev:
                d = (zH.float() - prev["zH"]).flatten(1).norm(dim=1) / prev["zH"].flatten(1).norm(dim=1).clamp_min(1e-6)
                relstep[sl, t] = d.cpu().numpy()
            prev["zH"] = zH.float()

        x, outs = rollout(model, batch, A, on_state=on_state)
        U[sl] = P_(gather(x, pos), muU, WU)
        for k, (preds, q) in enumerate(outs):
            c, pa = exact_correct(preds, batch["labels"])
            correct[sl, k], pixacc[sl, k], qhalt[sl, k] = c.cpu().numpy(), pa.cpu().numpy(), q.cpu().numpy()
        print(f"  {sl.stop}/{N}  ({time.time() - t0:.0f}s)", flush=True)

    base = np.array([data.base[i] for i in idx])
    np.savez(os.path.join(a.out, "traj.npz"), ZH=ZH, ZL=ZL, U=U, correct=correct, pixacc=pixacc, qhalt=qhalt,
             relstep=relstep, base=base, idx=idx, canonical=data.canonical[idx])
    H = model.config.halt_max_steps
    acc = correct.mean(0)
    summary = {**info, "n_examples": N, "n_act": A, "exact_acc_by_act_step": [round(float(v), 4) for v in acc],
               "exact_acc_at_trained_depth": float(acc[min(H, A) - 1]), "pca_evr": {"zH": evH, "zL": evL, "u": evU}}
    json.dump(summary, open(os.path.join(a.out, "rollout_summary.json"), "w"), indent=1)
    print(f"[rollout] exact acc (per augmented example, no voting) at step 1 {acc[0]:.4f}, "
          f"step {min(H, A)} {acc[min(H, A) - 1]:.4f}, step {A} {acc[-1]:.4f}")
    if a.ckpt and acc[min(H, A) - 1] < 0.02:
        print("[WARN] near-zero accuracy: the dataset build probably doesn't match the checkpoint's puzzle IDs "
              "(TRM conditions on puzzle_identifiers). Rebuild the data with the exact command/seed used for training.")


if __name__ == "__main__":
    main()

"""Train one model.  --kind block --k {48,8,1} | koopbase | koopdiff | featdiff ; --seed s.
Flow matching (rectified flow): x_tau = (1-tau) x0 + tau x1, target v = x1 - x0, x0 ~ N(0, I).
  block-k   tau sampled per block (frames in a block are denoised jointly at inference)
  koopdiff  parallel denoiser after a random given prefix s (50%: s=0), conditioned on the frozen Koopman rollout
            Lambda^(t-s+1) phi(true window ending at s-1) for every frame t >= s
  featdiff  identical, but conditioned on an MLP of (same window, steps ahead): no linear-dynamics structure
  koopbase  the Koopman model itself (one-shot prediction + latent linearity + reconstruction)
Writes <ckpt>/<name>_s<seed>.pt (skips if it exists)."""
import argparse, math, os, random, time

import torch
import torch.nn.functional as F

from sg.sim import simulate
from .nets import (CTX, FD, H, Denoiser, FeatControl, Koop, block_bounds, build_mask, norm_pos, prefix_bounds)


def data(n, seed, dev):
    X, _ = simulate(n, CTX + H, seed=seed, device=dev)
    return norm_pos(X.reshape(n, CTX + H, FD))


def windows_ending(full, idx):
    """full [B, 53, 10]; idx [B] = full-frame index of the window's last frame -> [B, 5, 10]"""
    ar = torch.arange(CTX, device=full.device)
    return full[torch.arange(full.shape[0], device=full.device)[:, None], (idx[:, None] - CTX + 1 + ar[None])]


def name_of(a):
    return f"block{a.k}" if a.kind == "block" else a.kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["block", "koopbase", "koopdiff", "featdiff"])
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ckpt", default="ckpt_sg2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    name = name_of(a)
    path = os.path.join(a.ckpt, f"{name}_s{a.seed}.pt")
    if os.path.exists(path):
        print(f"[{name} s{a.seed}] exists, skipping"); return
    steps = a.steps or (20000 if a.kind == "koopbase" else 100000)
    torch.manual_seed(a.seed); random.seed(a.seed)
    t0 = time.time()
    X = data(a.n_train, 1000 + a.seed, dev)
    print(f"[{name} s{a.seed}] data {tuple(X.shape)} ({time.time() - t0:.0f}s)", flush=True)
    koop = None
    if a.kind == "koopbase":
        model = Koop().to(dev)
    else:
        model = Denoiser(a.d, a.layers, cond_dim=128 if a.kind in ("koopdiff", "featdiff") else 0).to(dev)
        if a.kind == "featdiff":
            model.feat = FeatControl().to(dev)
        if a.kind == "koopdiff":
            kp = os.path.join(a.ckpt, f"koopbase_s{a.seed}.pt")
            while not os.path.exists(kp):
                time.sleep(30)
            koop = Koop().to(dev); koop.load_state_dict(torch.load(kp, map_location=dev)); koop.eval()
            for p in koop.parameters():
                p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = 1000
    lr_at = lambda s: a.lr * min(1.0, s / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    tH = torch.arange(H, device=dev)
    if a.kind == "block":
        g, e = block_bounds(a.k, 1, dev)
        mask_block = build_mask(g, e)
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        B = X[torch.randint(0, len(X), (a.batch,), device=dev)]
        ctx, fut = B[:, :CTX], B[:, CTX:]
        if a.kind == "koopbase":
            z0 = model.encode(ctx)
            pred = model.dec(model.roll(z0, (tH + 1)[None].expand(a.batch, -1)))
            wins = B.unfold(1, CTX, 1)[:, 1:H + 1].permute(0, 1, 3, 2)              # true windows ending at future t
            zw = model.encode(wins.reshape(-1, CTX, FD)).view(a.batch, H, -1)
            zl = model.roll(z0, (tH + 1)[None].expand(a.batch, -1))
            w = torch.exp(-(tH + 1).float() / 16)[None, :, None]
            lin = (((zw - zl) ** 2) * w).sum() / ((zw ** 2) * w).sum().clamp(min=1e-6)
            loss = F.mse_loss(pred, fut) + 0.1 * lin + F.mse_loss(model.dec(zw), fut)
        else:
            x0 = torch.randn_like(fut)
            if a.kind == "block":
                nb = math.ceil(H / a.k)
                tb = torch.rand(a.batch, nb, device=dev)
                tau = tb[:, tH // a.k]
                mask = mask_block.expand(a.batch, -1, -1, -1)
                lmask = torch.ones(a.batch, H, device=dev)
                cond = None
            else:
                s = torch.where(torch.rand(a.batch, device=dev) < 0.5, torch.zeros(a.batch, dtype=torch.long, device=dev),
                                torch.randint(1, H, (a.batch,), device=dev))
                tau = torch.rand(a.batch, 1, device=dev).expand(-1, H)
                mask = build_mask(*prefix_bounds(s))
                lmask = (tH[None] >= s[:, None]).float()
                win = windows_ending(B, CTX + s - 1)                                    # true window ending at s-1
                stp = (tH[None] - s[:, None] + 1).clamp(min=0)
                if a.kind == "koopdiff":
                    with torch.no_grad():
                        cond = koop.roll(koop.encode(win), stp)
                else:
                    cond = model.feat.features(win, stp)
            noisy = (1 - tau[..., None]) * x0 + tau[..., None] * fut
            v = model(ctx, fut, noisy, tau, mask, cond)
            loss = (((v - (fut - x0)) ** 2).mean(-1) * lmask).sum() / lmask.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 2000 == 0 or step == 1:
            print(f"[{name} s{a.seed}] step {step}/{steps} loss {loss.item():.5f} ({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(loss.item()):
                raise RuntimeError("non-finite loss")
    os.makedirs(a.ckpt, exist_ok=True)
    torch.save(model.state_dict(), path + ".tmp"); os.replace(path + ".tmp", path)
    print(f"[{name} s{a.seed}] saved ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

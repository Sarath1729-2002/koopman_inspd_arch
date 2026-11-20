"""Train one arm on simulated hard-sphere trajectories (5 context + 48 predicted frames).
  --arm parallel | ar | koop (trained with random re-encoding schedules) | koop1 (one-shot only)
Koopman losses: prediction MSE under the sampled schedule + latent linearity ||phi(true window_t) - Lambda^t z0||^2
(relative) + reconstruction ||psi(phi(window_t)) - frame_t||^2.   -> <ckpt>/<arm>.pt"""
import argparse, os, random, time

import torch
import torch.nn.functional as F

from .models import AR, CTX, FD, H, Koop, Parallel
from .sim import simulate


def data(n, seed, dev):
    X, C = simulate(n, CTX + H, seed=seed, device=dev)
    return X.reshape(n, CTX + H, FD), C


def koop_loss(m, X, schedule):
    ctx, tgt = X[:, :CTX], X[:, CTX:]
    reenc = [] if schedule is None else list(range(schedule, H, schedule))
    pred, _ = m.rollout(ctx, reenc_at=reenc)
    lp = F.mse_loss(pred, tgt)
    ts = torch.arange(1, H + 1, device=X.device)
    wins = X.unfold(1, CTX, 1)[:, 1:H + 1].permute(0, 1, 3, 2)          # true windows ending at horizon t
    zw = m.encode(wins.reshape(-1, CTX, FD)).view(X.shape[0], H, -1)
    zl = m.powers(m.encode(ctx), ts)
    w = torch.exp(-ts.float() / 16)[None, :, None]                        # linearity matters most near t=0
    ll = (((zw - zl) ** 2) * w).sum() / ((zw ** 2) * w).sum().clamp(min=1e-6)
    lr_ = F.mse_loss(m.dec(zw), tgt)
    return lp + 0.1 * ll + lr_, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["parallel", "ar", "koop", "koop1"])
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="ckpt_sg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = a.device
    t0 = time.time()
    X, _ = data(a.n_train, 1000 + a.seed, dev)
    print(f"[{a.arm}] data {tuple(X.shape)} ({time.time() - t0:.0f}s)", flush=True)
    m = {"parallel": Parallel, "ar": AR, "koop": Koop, "koop1": Koop}[a.arm]().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(X), (a.batch,), device=dev)
        B = X[idx]
        if a.arm == "parallel":
            loss = lp = F.mse_loss(m(B[:, :CTX]), B[:, CTX:])
        elif a.arm == "ar":
            loss = lp = F.mse_loss(m.step_all(B[:, :-1])[:, CTX - 1:], B[:, CTX:])
        else:
            s = None if (a.arm == "koop1" or rng.random() < 0.25) else rng.choice([2, 4, 8, 16])
            loss, lp = koop_loss(m, B, s)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == 1:
            print(f"[{a.arm}] step {step} loss {loss.item():.6f} pred-mse {lp.item():.6f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.ckpt, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(a.ckpt, f"{a.arm}.pt"))
    print(f"[{a.arm}] saved ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

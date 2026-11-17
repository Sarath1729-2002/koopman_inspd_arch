"""Train one (task, arm, seed) and evaluate length generalization. Writes runs/<task>/<arm>_s<seed>.json.

Training follows Fan et al.: data sampled online, curriculum on max length, loss only on the answer
after exactly T(n) loops, AdamW, cosine decay after the curriculum ends, batch 64.
Evaluation at every eval length (512 samples each):
  oracle     exact match after T(n) loops
  overloop   exact match after 2*T(n) loops       (does running longer break it?)
  confstop   loops chosen per sample by max answer confidence within 1..3*T(n)  (no oracle)
  curve      exact match at every loop 1..3*T(n), for the train-max and the longest eval length
  norm_ratio ||h after 2T|| / ||h after T||       (state drift)
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .model import Looped
from .tasks import IGNORE, TASKS, VOCAB, make_batch


def exact(logits, Y):
    m = Y != IGNORE
    return ((logits.argmax(-1) == Y) | ~m).all(-1)


@torch.no_grad()
def evaluate(model, task, lengths, n_eval, dev, seed, train_max):
    model.eval()
    T = TASKS[task][1]
    res = {}
    for L in lengths:
        X, Y, ts = make_batch(task, [L] * n_eval, seed=seed * 7919 + L)
        X, Y = X.to(dev), Y.to(dev)
        Tn = int(ts[0])
        allt, _ = model(X, 3 * Tn, keep_all=True)               # [B, 3T, L, V]
        acc_t = torch.stack([exact(allt[:, t], Y).float().mean() for t in range(3 * Tn)]).cpu()
        m = (Y != IGNORE)
        conf = (allt.softmax(-1).max(-1).values * m[:, None]).sum(-1) / m.sum(-1)[:, None]   # [B, 3T]
        pick = conf.argmax(1)
        cs = exact(allt[torch.arange(len(X)), pick], Y).float().mean()
        _, hT = model(X, Tn)
        _, h2T = model(X, 2 * Tn)
        nr = float((h2T.float().norm(dim=-1).mean() / hT.float().norm(dim=-1).mean()))
        res[L] = {"oracle": float(acc_t[Tn - 1]), "overloop": float(acc_t[2 * Tn - 1]), "confstop": float(cs),
                  "norm_ratio": nr, "T": Tn,
                  "curve": acc_t.tolist() if L in (train_max, max(lengths)) else None}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--arm", required=True, choices=["plain", "gainctl", "koopman", "koopman_re"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default per task: parity 20k, copy 30k, addition 40k")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--train_max", type=int, default=None)
    ap.add_argument("--eval_lengths", default=None)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--reenc_every", type=int, default=4)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    gen, Tfn, tmax_default, evl_default, depth, heads = TASKS[a.task]
    steps = a.steps or {"parity": 20000, "copy": 30000, "addition": 40000}[a.task]
    tmax = a.train_max or tmax_default
    evl = [int(x) for x in a.eval_lengths.split(",")] if a.eval_lengths else evl_default
    if tmax not in evl:
        evl = sorted(set(evl + [tmax]))
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = a.device
    model = Looped(len(VOCAB), a.arm, d=a.d, heads=heads, depth=depth, reenc_every=a.reenc_every).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cur_end = int(0.4 * steps)
    interval = max(1, cur_end // tmax)
    warm = 500
    rng = random.Random(a.seed)
    path = os.path.join(a.out, a.task, f"{a.arm}_s{a.seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0, log = time.time(), []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    for step in range(1, steps + 1):
        cur = min(tmax, 1 + step // interval)
        lens = [rng.randint(1, cur) for _ in range(a.batch)]
        X, Y, ts = make_batch(a.task, lens, rng=rng)
        X, Y, ts = X.to(dev), Y.to(dev), ts.to(dev)
        lr = a.lr * min(1.0, step / warm) if step <= cur_end else a.lr * 0.5 * (1 + math.cos(math.pi * (step - cur_end) / (steps - cur_end)))
        for g in opt.param_groups:
            g["lr"] = lr
        with amp:
            logits, _ = model(X, int(ts.max()), read_at=ts)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                acc = exact(logits, Y).float().mean().item()
            log.append({"step": step, "loss": loss.item(), "train_exact": acc, "cur_max": cur, "lr": lr, "gnorm": float(gn)})
            print(f"[{a.task}/{a.arm}/s{a.seed}] step {step:6d} cur {cur:2d} loss {loss.item():.4f} exact {acc:.3f} "
                  f"gnorm {float(gn):.2f} ({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(loss.item()):
                print("  non-finite loss, stopping early"); break
    with amp:
        ev = evaluate(model, a.task, evl, a.n_eval, dev, a.seed, tmax)
    extra = {}
    if hasattr(model, "lam"):
        r = model.lam.radius().detach().cpu()
        extra = {"lambda_radius_mean": float(r.mean()), "lambda_radius_min": float(r.min()), "lambda_radius_max": float(r.max())}
    out = {"task": a.task, "arm": a.arm, "seed": a.seed, "steps": steps, "train_max": tmax, "eval": ev, "log": log,
           "params": sum(p.numel() for p in model.parameters()), "seconds": time.time() - t0, **extra}
    json.dump(out, open(path, "w"), indent=1)
    ood = [v["oracle"] for L, v in ev.items() if L > tmax]
    print(f"[{a.task}/{a.arm}/s{a.seed}] done: in-dist(train max) {ev[tmax]['oracle']:.3f}  OOD mean {sum(ood) / max(1, len(ood)):.3f} -> {path}")


if __name__ == "__main__":
    main()

"""Train one (arm, seed) on S5 at length <= train_len, evaluate per-token accuracy at longer lengths.
Writes <out>/<arm>_s<seed>.json."""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import group, make_batch
from .models import build


@torch.no_grad()
def evaluate(model, arm, lengths, n_eval, dev, seed, gname):
    model.eval()
    res = {}
    for L in lengths:
        g = torch.Generator().manual_seed(10_000 + seed * 101 + L)
        x, y = make_batch(n_eval, L, g, gname)
        x, y = x.to(dev), y.to(dev)
        accs = []
        for i in range(0, n_eval, 256):
            accs.append((model(x[i:i + 256]).argmax(-1) == y[i:i + 256]).float())
        pos = torch.cat(accs).mean(0).cpu()                       # [L] accuracy per position
        r = {"acc": float(pos.mean()), "acc_last_quarter": float(pos[3 * L // 4:].mean()),
             "curve": pos[:: max(1, L // 128)].tolist()}
        if arm.startswith("bnd"):
            k = int(arm[3:])
            isb = torch.tensor([(t + 1) % k == 0 for t in range(L)])
            r["acc_at_boundary"] = float(pos[isb].mean())
            r["acc_mid_chunk"] = float(pos[~isb].mean())
        res[L] = r
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", default="s5", choices=["s5", "s3"])
    ap.add_argument("--steps", type=int, default=50000, help="max steps (stops early once solved)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--train_len", type=int, default=64)
    ap.add_argument("--start_len", type=int, default=2, help="curriculum start length")
    ap.add_argument("--eval_lengths", default="64,128,256,512")
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    model = build(a.arm, group(a.group)[2], a.d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(a.seed)
    cur_end, warm, start_len = int(0.4 * a.steps), 500, min(a.start_len, a.train_len)
    t0, log, solved = time.time(), [], 0
    for step in range(1, a.steps + 1):
        cur = min(a.train_len, start_len + int((a.train_len - start_len) * step / max(1, cur_end)))
        x, y = make_batch(a.batch, cur, gen, a.group)
        x, y = x.to(dev), y.to(dev)
        lr = a.lr * min(1.0, step / warm) if step <= cur_end else \
            a.lr * 0.5 * (1 + math.cos(math.pi * (step - cur_end) / max(1, a.steps - cur_end)))
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            log.append({"step": step, "loss": loss.item(), "acc": acc, "len": cur, "gnorm": float(gn)})
            print(f"[{a.arm}/s{a.seed}] step {step:6d} len {cur:3d} loss {loss.item():.4f} tok-acc {acc:.4f} "
                  f"gnorm {float(gn):.2f} ({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(loss.item()):
                print("  non-finite loss, stopping"); break
            solved = solved + 1 if (cur == a.train_len and acc >= 0.999) else 0
            if solved >= 3:
                print("  solved at train length (3 checks >= 0.999), stopping early"); break
    lengths = sorted(set([a.train_len] + [int(v) for v in a.eval_lengths.split(",")]))
    ev = evaluate(model, a.arm, lengths, a.n_eval, dev, a.seed, a.group)
    out = {"group": a.group, "arm": a.arm, "seed": a.seed, "train_len": a.train_len, "steps_run": step, "eval": ev, "log": log,
           "params": sum(p.numel() for p in model.parameters()), "seconds": time.time() - t0}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.arm}_s{a.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{a.arm}/s{a.seed}] done: " + "  ".join(f"L{L} {r['acc']:.3f}" for L, r in ev.items()) + f" -> {path}")


if __name__ == "__main__":
    main()

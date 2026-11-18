"""Train one (task, arm, seed) at length <= train_len; per-token accuracy at long lengths. -> <out>/<task>/<arm>_s<seed>.json"""
import argparse, json, math, os, time

import torch
import torch.nn.functional as F

from .core import TASKS, build, make_batch


@torch.no_grad()
def evaluate(model, task, lengths, n_eval, dev, seed):
    model.eval()
    res = {}
    for L in lengths:
        x, y = make_batch(task, n_eval, L, torch.Generator().manual_seed(10_000 + seed * 101 + L))
        pos = torch.cat([(model(x[i:i + 256].to(dev)).argmax(-1) == y[i:i + 256].to(dev)).float()
                         for i in range(0, n_eval, 256)]).mean(0).cpu()
        res[L] = {"acc": float(pos.mean()), "acc_last_quarter": float(pos[3 * L // 4:].mean()),
                  "curve": pos[:: max(1, L // 128)].tolist()}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train_len", type=int, default=64)
    ap.add_argument("--eval_lengths", default="64,256,1024,4096")
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev, V = a.device, TASKS[a.task]
    model = build(a.arm, V).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(a.seed)
    cur_end, warm = int(0.3 * a.steps), 300
    t0, log, solved = time.time(), [], 0
    for step in range(1, a.steps + 1):
        cur = min(a.train_len, 2 + int((a.train_len - 2) * step / max(1, cur_end)))
        x, y = make_batch(a.task, a.batch, cur, gen)
        x, y = x.to(dev), y.to(dev)
        lr = a.lr * min(1.0, step / warm) if step <= cur_end else \
            a.lr * 0.5 * (1 + math.cos(math.pi * (step - cur_end) / max(1, a.steps - cur_end)))
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            log.append({"step": step, "loss": loss.item(), "acc": acc, "len": cur})
            print(f"[{a.task}/{a.arm}/s{a.seed}] step {step:6d} len {cur:3d} loss {loss.item():.5f} acc {acc:.4f} "
                  f"gnorm {float(gn):.2f} ({time.time() - t0:.0f}s)", flush=True)
            solved = solved + 1 if (cur == a.train_len and acc >= 0.9995) else 0
            if solved >= 4:
                print("  solved at train length, stopping early"); break
    lengths = sorted(set([a.train_len] + [int(v) for v in a.eval_lengths.split(",")]))
    ev = evaluate(model, a.task, lengths, a.n_eval, dev, a.seed)
    out = {"task": a.task, "arm": a.arm, "seed": a.seed, "train_len": a.train_len, "steps_run": step, "eval": ev,
           "log": log, "params": sum(p.numel() for p in model.parameters()), "seconds": time.time() - t0}
    if hasattr(model, "rho"):
        r = torch.sigmoid(model.rho.weight).detach()
        out["radius_min"], out["radius_mean"] = float(r.min()), float(r.mean())
    path = os.path.join(a.out, a.task, f"{a.arm}_s{a.seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{a.task}/{a.arm}/s{a.seed}] done: " + "  ".join(f"L{L} {r['acc']:.3f}" for L, r in ev.items()) + f" -> {path}")


if __name__ == "__main__":
    main()

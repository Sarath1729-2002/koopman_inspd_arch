"""Train one byte LM at context 512 on wikitext-103. -> ckpt/<core>_<arm>.pt"""
import argparse, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ByteLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="rglru", choices=["rglru", "mamba"])
    ap.add_argument("--reenc_every", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    arm = f"re{a.reenc_every}" if a.reenc_every else "base"
    data = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint8, mode="r")
    rng = np.random.default_rng(a.seed)
    dev = a.device
    model = ByteLM(a.core, a.d, a.layers, a.reenc_every).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ix = rng.integers(0, len(data) - a.ctx - 1, a.batch)
        xy = torch.from_numpy(np.stack([data[i:i + a.ctx + 1] for i in ix]).astype(np.int64)).to(dev)
        lr = a.lr * min(1, step / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        with amp:
            logits = model(xy[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, 256), xy[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0 or step == 1:
            print(f"[{a.core}/{arm}] step {step:5d} bpb {loss.item() / math.log(2):.3f} gnorm {float(gn):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(loss.item()):
                print("non-finite loss"); break
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.core}_{arm}.pt")
    torch.save({"args": vars(a), "arm": arm, "state": model.state_dict()}, path)
    print(f"[{a.core}/{arm}] saved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

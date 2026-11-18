"""Evaluate checkpoints at 16x the training context; per-position bits-per-byte in buckets.
For base checkpoints also run the test-time arms (all act only beyond the training context, every 64 tokens):
  none      as trained
  reset     zero the recurrent state every 512 tokens          (= a 512-window model; the 'forget' heuristic)
  clamp     clip each state channel to its in-distribution mean +- 3 std          (norm/outlier heuristic)
  ae        post-hoc course correction: per-layer denoising autoencoder trained on in-distribution states
            (positions 256..512 of training-length windows), applied as h <- AE(h)     (Koopman re-encoding, training-free)
Writes results/<core>_<arm>.json"""
import argparse, glob, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ByteLM

BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]


def load(path, dev):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    m = ByteLM(a["core"], a["d"], a["layers"], a["reenc_every"]); m.load_state_dict(ck["state"])
    return m.to(dev).eval(), a, ck["arm"]


@torch.no_grad()
def per_pos_loss(model, X, dev, bs=8):
    out, rms = [], {}
    def rec(i, hs):
        rms.setdefault(i, []).append(hs.float().pow(2).mean(-1).sqrt().mean(0).cpu())
    model.rec_hook = rec
    for l in model.rec_layers():
        l.rec = rec
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev)
        lg = model(x[:, :-1]).float()
        out.append(F.cross_entropy(lg.transpose(1, 2), x[:, 1:], reduction="none").cpu())
    for l in model.rec_layers():
        l.rec = None
    loss = torch.cat(out).mean(0) / math.log(2)
    r = torch.stack([torch.stack(v).mean(0) for v in rms.values()]).mean(0)   # [L] mean state rms over layers
    return loss, r


@torch.no_grad()
def collect_states(model, data, ctx, n, dev, rng):
    S = {}
    def rec(i, hs):
        S.setdefault(i, []).append(hs[:, ctx // 2::8].reshape(-1, hs.shape[-1]).float().cpu())
    for l in model.rec_layers():
        l.rec = rec
    for _ in range(n // 16):
        ix = rng.integers(0, len(data) - ctx - 1, 16)
        model(torch.from_numpy(np.stack([data[i:i + ctx] for i in ix]).astype(np.int64)).to(dev))
    for l in model.rec_layers():
        l.rec = None
    return {i: torch.cat(v) for i, v in S.items()}


def train_ae(H, dev, steps=1500):
    mu, sd = H.mean(0).to(dev), H.std(0).to(dev) + 1e-6
    D = H.shape[1]
    ae = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, D)).to(dev)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    Hd = ((H.to(dev) - mu) / sd)
    for s in range(steps):
        h = Hd[torch.randint(0, len(Hd), (1024,), device=dev)]
        scale = torch.exp(torch.empty(len(h), 1, device=dev).uniform_(math.log(0.5), math.log(2.0)))
        hin = h * scale + 0.3 * torch.randn_like(h)                          # drift: rescaling + noise
        loss = F.mse_loss(ae(hin), h)
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    return lambda h: ae((h - mu) / sd) * sd + mu, float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--only", default="*", help="glob on checkpoint name, e.g. rglru_*")
    ap.add_argument("--data", default="data")
    ap.add_argument("--eval_len", type=int, default=8192)
    ap.add_argument("--n_seq", type=int, default=64)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    ev = np.fromfile(os.path.join(a.data, "eval.bin"), dtype=np.uint8)
    tr = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint8, mode="r")
    L = a.eval_len + 1
    n = min(a.n_seq, len(ev) // L)
    X = torch.from_numpy(ev[: n * L].reshape(n, L).astype(np.int64))
    buckets = [(lo, hi) for lo, hi in BUCKETS if hi <= a.eval_len]
    for path in sorted(glob.glob(os.path.join(a.ckpt, a.only + ".pt"))):
        model, args, arm = load(path, dev)
        ctx = args["ctx"]
        arms = {arm: None}
        if arm == "base":
            rng = np.random.default_rng(0)
            S = collect_states(model, tr, ctx, 256, dev, rng)
            lo = {i: (v.mean(0) - 3 * v.std(0)).to(dev) for i, v in S.items()}
            hi = {i: (v.mean(0) + 3 * v.std(0)).to(dev) for i, v in S.items()}
            aes = {}
            for i, v in S.items():
                aes[i], fl = train_ae(v, dev)
                print(f"  AE layer {i}: final denoise mse {fl:.3f}")
            beyond = lambda pos: pos >= ctx and pos % 64 == 0
            arms["reset"] = lambda i, h, pos: torch.zeros_like(h) if pos % ctx == 0 else h
            arms["clamp"] = lambda i, h, pos: torch.maximum(torch.minimum(h, hi[i]), lo[i]) if beyond(pos) else h
            arms["ae"] = lambda i, h, pos: aes[i](h.float()).to(h.dtype) if beyond(pos) else h
        for name, ctrl in arms.items():
            t0 = time.time()
            model.set_ctrl(ctrl if name != arm or arm != "base" else None)
            if name == arm:
                model.set_ctrl(None)
            loss, rms = per_pos_loss(model, X, dev)
            model.set_ctrl(None)
            res = {"core": args["core"], "arm": name if name != arm else arm, "trained": arm, "ctx": ctx,
                   "buckets": {f"{lo}-{hi}": float(loss[lo:hi].mean()) for lo, hi in buckets},
                   "state_rms": {f"{lo}-{hi}": float(rms[lo:hi].mean()) for lo, hi in buckets},
                   "curve": loss.reshape(-1, 64).mean(1).tolist()}
            tag = f"{args['core']}_{res['arm']}"
            json.dump(res, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
            print(f"{tag:16s} " + "  ".join(f"{k}:{v:.3f}" for k, v in res["buckets"].items()) +
                  f"  | state rms in-dist {res['state_rms'][f'{256}-{512}']:.3f} far {list(res['state_rms'].values())[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

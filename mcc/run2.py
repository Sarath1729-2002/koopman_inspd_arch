"""Iteration 2: correct WITHOUT forgetting, and measure where long memory matters.

Arms (each edits its own long state every --every tokens once past W; shadows as in iteration 1):
  none | window (long <- oldest shadow) | clamp (clip to mu +- 3sd)
  obs3      drifted elements (|z|>3)  -> shadow                     (iteration-1 obs_elem)
  obs5      drifted elements (|z|>5)  -> shadow                     (touch fewer elements)
  blend50   drifted elements (|z|>3)  -> shadow + 0.50 (long - shadow)   (pull back, keep half the long value)
  blend25   drifted elements (|z|>3)  -> shadow + 0.25 (long - shadow)
Tasks:
  ppl     natural text, L tokens (as iteration 1)
  recall  [prefix filler 4096][passage P 1024][gap G][P again 1024], G in {512, 3072}:
          recall gain = NLL(first copy of P) - NLL(second copy). Window can only recall when G + 1024 <= W.
Writes <out>/<model>.json"""
import argparse, json, os, time

import numpy as np
import torch
import torch.nn.functional as F

from .run import calib_stats, load, make_cache, step

ARMS = [("none", None), ("window", None), ("clamp", 3.0), ("obs3", (3.0, 0.0)), ("obs5", (5.0, 0.0)),
        ("blend50", (3.0, 0.5)), ("blend25", (3.0, 0.25))]


def tokens(tok, path, n_tok, offset_frac):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    off = int(len(data) * offset_frac)
    text = bytes(data[off: off + min(len(data) - off, n_tok * 6)]).decode("utf-8", errors="ignore")
    return tok(text, add_special_tokens=False)["input_ids"][:n_tok]


@torch.no_grad()
def run_pass(model, X, W, M, every, mu, sd, dev, dtype):
    """X [S, T+1] -> loss [S, R, T] for every arm."""
    nl = model.config.num_hidden_layers
    S, T = X.shape[0], X.shape[1] - 1
    R = len(ARMS)
    cache = make_cache(model, S * (R + M), dev, dtype)
    base = torch.arange(S, device=dev) * (R + M)
    arm_rows = (base[:, None] + torch.arange(R, device=dev)[None]).reshape(-1)
    loss = torch.zeros(S, R, T)
    Xd = X.to(dev)
    for t in range(T):
        for j in range(M):
            if t >= j * W // M and (t - j * W // M) % W == 0:
                for l in range(nl):
                    cache.ssm_states[l][base + R + j] = 0
        lg = step(model, cache, Xd[:, t].repeat_interleave(R + M))[arm_rows].float()
        loss[:, :, t] = F.cross_entropy(lg, Xd[:, t + 1].repeat_interleave(R), reduction="none").view(S, R).cpu()
        if t + 1 >= W and (t + 1) % every == 0:
            ages = [((t + 1 - j * W // M) % W) if t + 1 >= j * W // M else -1 for j in range(M)]
            shr = base + R + int(np.argmax(ages))
            for l in range(nl):
                st = cache.ssm_states[l]
                shadow = st[shr].float()
                for i, (name, p) in enumerate(ARMS):
                    if name == "none":
                        continue
                    if name == "window":
                        st[base + i] = shadow.to(st.dtype); continue
                    cur = st[base + i].float()
                    if name == "clamp":
                        st[base + i] = torch.maximum(torch.minimum(cur, mu[l] + p * sd[l]), mu[l] - p * sd[l]).to(st.dtype); continue
                    tau, keep = p
                    bad = (cur - mu[l]).abs() > tau * sd[l]
                    st[base + i] = torch.where(bad, shadow + keep * (cur - shadow), cur).to(st.dtype)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--W", type=int, default=2048)
    ap.add_argument("--L", type=int, default=12288)
    ap.add_argument("--n_seq", type=int, default=12)
    ap.add_argument("--n_recall", type=int, default=16)
    ap.add_argument("--gaps", default="512,3072")
    ap.add_argument("--n_calib", type=int, default=64)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--every", type=int, default=64)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_mcc2")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, dtype = a.device, getattr(torch, a.dtype)
    t0 = time.time()
    tok, model = load(a.model, dev, dtype)
    mu, sd = calib_stats(model, tok, os.path.join(a.data, "train.bin"), a.W, a.n_calib, dev, dtype)
    names = [n for n, _ in ARMS]
    res = {"model": a.model, "W": a.W, "arms": names, "every": a.every}
    # natural ppl
    ev = tokens(tok, os.path.join(a.data, "eval.bin"), a.n_seq * (a.L + 1), 0.0)
    k = len(ev) // (a.L + 1)
    X = torch.tensor(ev[: k * (a.L + 1)]).view(k, a.L + 1)
    loss = run_pass(model, X, a.W, a.M, a.every, mu, sd, dev, dtype)
    res["ppl"] = {"L": a.L, "n_seq": k, "loss_curve": {n: loss[:, i].mean(0).reshape(-1, 64).mean(1).tolist() for i, n in enumerate(names)}}
    print(f"[{a.model}] ppl pass done ({time.time() - t0:.0f}s)", flush=True)
    # recall
    P_all = tokens(tok, os.path.join(a.data, "eval.bin"), a.n_recall * 1024, 0.55)
    F_all = tokens(tok, os.path.join(a.data, "train.bin"), a.n_recall * (4096 + 3072 + 64), 0.3)
    res["recall"] = {}
    for G in [int(g) for g in a.gaps.split(",")]:
        seqs = []
        for i in range(a.n_recall):
            P = P_all[i * 1024:(i + 1) * 1024]
            Fl = F_all[i * (4096 + 3072): (i + 1) * (4096 + 3072)]
            if len(P) < 1024 or len(Fl) < 4096 + G:
                break
            seqs.append(Fl[:4096] + P + Fl[4096:4096 + G] + P + [P[0]])
        X = torch.tensor(seqs)
        loss = run_pass(model, X, a.W, a.M, a.every, mu, sd, dev, dtype)
        s1, s2 = 4096, 4096 + 1024 + G
        first = loss[:, :, s1 + 16: s1 + 1023].mean((0, 2)); second = loss[:, :, s2 + 16: s2 + 1023].mean((0, 2))
        res["recall"][str(G)] = {n: {"first": float(first[i]), "second": float(second[i]), "gain": float(first[i] - second[i])} for i, n in enumerate(names)}
        print(f"[{a.model}] recall G={G}: " + " ".join(f"{n}:{float(first[i] - second[i]):+.3f}" for i, n in enumerate(names)), flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, a.model.split("/")[-1] + ".json"), "w"), indent=1)
    print(f"[{a.model}] done ({res['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()

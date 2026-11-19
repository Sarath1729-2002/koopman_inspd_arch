"""Course-correcting a pretrained Mamba past its training length, by re-encoding from observations.

One batched pass per model. For each evaluation sequence the batch holds, side by side:
  - one LONG state per arm (each arm edits its own long state), and
  - M SHADOW states restarted from zero at staggered times, so at any step there is a fresh encoding of the last
    <= W tokens (W = training length): an in-distribution state by construction (Fathi et al.: re-encode from the
    observable). All rows read the same tokens; conv states coincide after 4 tokens.
Every --every tokens beyond W the arms act on their long state:
  none        nothing
  window      long <- oldest shadow                     (= a W-token sliding-window model; the 'forget' heuristic)
  clamp       long <- clip(long, mu +- tau*sd)          (per element, stats from positions W/2..W)
  obs_elem    long <- where(|long-mu| > tau*sd, shadow, long)          (re-encode only drifted ELEMENTS)
  obs_chan    per (layer, channel): if the channel's state norm is a > tau outlier, take the shadow's channel
Writes <out>/<model>.json with per-position loss per arm and drift diagnostics."""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

ARMS = ["none", "window", "clamp", "obs_elem", "obs_chan"]


def load(name, dev, dtype):
    from transformers import AutoTokenizer, MambaForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    m = MambaForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return tok, m


def make_cache(model, B, dev, dtype):
    from transformers.models.mamba.modeling_mamba import MambaCache
    return MambaCache(model.config, max_batch_size=B, dtype=dtype, device=dev)


@torch.no_grad()
def step(model, cache, ids):
    """one token for every row; cache_position = conv_kernel-1 makes HF roll the conv window correctly from t=0."""
    cp = torch.tensor([model.config.conv_kernel - 1], device=ids.device)
    return model(input_ids=ids[:, None], cache_params=cache, use_cache=True, cache_position=cp).logits[:, -1]


def token_seqs(tok, path, n, L, offset_bytes=0):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    offset_bytes = min(offset_bytes, len(data) // 2)
    text = bytes(data[offset_bytes: offset_bytes + min(len(data) - offset_bytes, n * L * 6)]).decode("utf-8", errors="ignore")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    k = min(n, len(ids) // L)
    return torch.tensor(ids[: k * L]).view(k, L)


@torch.no_grad()
def calib_stats(model, tok, path, W, n, dev, dtype, bs=32):
    """per-layer elementwise mean/std of the ssm state over positions W/2..W (in-distribution)."""
    X = token_seqs(tok, path, n, W, offset_bytes=10_000_000)
    assert len(X) > 0, "no calibration sequences"
    nl = model.config.num_hidden_layers
    s1 = s2 = None; cnt = 0
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev); B = len(x)
        cache = make_cache(model, B, dev, dtype)
        for t in range(W):
            step(model, cache, x[:, t])
            if t >= W // 2 and t % 16 == 0:
                S = torch.stack([cache.ssm_states[l].float() for l in range(nl)])      # [nl, B, D, N]
                a, b = S.sum(1), (S ** 2).sum(1)
                s1 = a if s1 is None else s1 + a; s2 = b if s2 is None else s2 + b; cnt += B
    mu = s1 / cnt
    sd = (s2 / cnt - mu ** 2).clamp(min=1e-12).sqrt()
    # channel-norm stats
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--W", type=int, default=2048, help="training context length")
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--n_calib", type=int, default=64)
    ap.add_argument("--M", type=int, default=4, help="number of staggered shadow states")
    ap.add_argument("--every", type=int, default=64)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_mcc")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, dtype = a.device, getattr(torch, a.dtype)
    t0 = time.time()
    tok, model = load(a.model, dev, dtype)
    nl = model.config.num_hidden_layers
    mu, sd = calib_stats(model, tok, os.path.join(a.data, "train.bin"), a.W, a.n_calib, dev, dtype)
    chan_mu = mu.norm(dim=-1)                                        # [nl, D] (norm of mean as a reference scale)
    print(f"[{a.model}] calib done ({time.time() - t0:.0f}s)", flush=True)
    X = token_seqs(tok, os.path.join(a.data, "eval.bin"), a.n_seq, a.L + 1)
    S, R, M = len(X), len(ARMS), a.M
    rows = S * (R + M)                                               # row = s*(R+M) + j ; j<R arm, j>=R shadow
    arm_rows = torch.tensor([s * (R + M) + j for s in range(S) for j in range(R)], device=dev)
    sh = lambda s, j: s * (R + M) + R + j
    cache = make_cache(model, rows, dev, dtype)
    loss = torch.zeros(S, R, a.L)
    drift = {b: [] for b in range(a.L // a.W + 1)}                  # fraction of drifted elements in 'none', per W-block
    Xd = X.to(dev)
    for t in range(a.L):
        ids = Xd[:, t].repeat_interleave(R + M)
        # shadows: restart shadow j whenever (t - j*W/M) % W == 0
        for j in range(M):
            if t >= j * a.W // M and (t - j * a.W // M) % a.W == 0:
                r = torch.tensor([sh(s, j) for s in range(S)], device=dev)
                for l in range(nl):
                    cache.ssm_states[l][r] = 0
        lg = step(model, cache, ids)[arm_rows].float()
        loss[:, :, t] = F.cross_entropy(lg, Xd[:, t + 1].repeat_interleave(R), reduction="none").view(S, R).cpu()
        if t + 1 >= a.W and (t + 1) % a.every == 0:
            ages = [((t + 1 - j * a.W // M) % a.W) if t + 1 >= j * a.W // M else -1 for j in range(M)]
            jo = int(np.argmax(ages))                                 # oldest shadow (< W tokens of context)
            base = torch.arange(S, device=dev) * (R + M)
            shr = base + R + jo
            for l in range(nl):
                st = cache.ssm_states[l]
                lo, hi = mu[l] - a.tau * sd[l], mu[l] + a.tau * sd[l]
                shadow = st[shr].float()
                st[base + 1] = shadow.to(st.dtype)                                             # window
                st[base + 2] = torch.maximum(torch.minimum(st[base + 2].float(), hi), lo).to(st.dtype)   # clamp
                cur = st[base + 3].float(); bad = (cur < lo) | (cur > hi)
                st[base + 3] = torch.where(bad, shadow, cur).to(st.dtype)                       # obs_elem
                cur = st[base + 4].float()
                badc = ((cur - mu[l]).abs() / sd[l]).mean(-1) > a.tau                          # [S, D]
                st[base + 4] = torch.where(badc[..., None], shadow, cur).to(st.dtype)           # obs_chan
                if l == nl // 2:
                    cur = st[base].float()
                    drift[(t + 1) // a.W].append(float(((cur < lo) | (cur > hi)).float().mean()))
        if t % 2048 == 0:
            print(f"[{a.model}] t={t}/{a.L} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    res = {"model": a.model, "W": a.W, "L": a.L, "n_seq": S, "M": M, "every": a.every, "tau": a.tau, "arms": ARMS,
           "loss_curve": {arm: loss[:, i].mean(0).reshape(-1, 64).mean(1).tolist() for i, arm in enumerate(ARMS)},
           "drift_frac_by_block": {str(k): (float(np.mean(v)) if v else None) for k, v in drift.items()},
           "seconds": time.time() - t0}
    tag = a.model.split("/")[-1]
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    print(f"[{a.model}] done ({res['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()

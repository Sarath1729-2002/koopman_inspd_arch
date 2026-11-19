"""Iteration 3: a LEARNED course-correction gate on a frozen pretrained Mamba.

Every --every tokens past W, per layer and per channel:   h_long <- g * h_long + (1 - g) * h_shadow
  h_shadow = oldest staggered shadow state (fresh in-distribution encoding of the last <= W tokens; re-encoding from
  the observable). g = sigmoid(b[l, c] + MLP_l(features)), features per channel: mean |z| of long, mean |z| of shadow,
  log norm ratio, cosine(long, shadow). Initialised at g ~ 0.05 (~ the 'window' heuristic), MLP output zero-init.
Training: Mamba frozen; gate trained on next-token loss on train text, truncated backprop over --seg tokens after
each gate application. Evaluation exactly as iteration 2 (natural PPL + recall at G=512/3072), arms:
  none | window | obs3 | learned.   Writes <out>/<model>.json"""
import argparse, json, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .run import calib_stats, load, make_cache
from .run2 import tokens


class Gate(nn.Module):
    def __init__(self, nl, D, init=-3.0):
        super().__init__()
        self.b = nn.Parameter(torch.full((nl, D), init))
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1)) for _ in range(nl)])
        for m in self.mlp:
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)

    def forward(self, l, hL, hS, mu, sd):
        zL = ((hL - mu) / sd).abs().mean(-1); zS = ((hS - mu) / sd).abs().mean(-1)
        nr = torch.log((hL.norm(dim=-1) + 1e-6) / (hS.norm(dim=-1) + 1e-6))
        cs = F.cosine_similarity(hL, hS, dim=-1)
        f = torch.stack([zL.clamp(max=20) / 5, zS.clamp(max=20) / 5, nr.clamp(-5, 5), cs], -1)
        g = torch.sigmoid(self.b[l] + self.mlp[l](f).squeeze(-1))       # [B, D]
        return g[..., None] * hL + (1 - g[..., None]) * hS, g


def fwd(model, cache, ids):
    cp = torch.tensor([model.config.conv_kernel - 1], device=ids.device)
    return model(input_ids=ids[:, None], cache_params=cache, use_cache=True, cache_position=cp).logits[:, -1]


def shadow_reset(scache, nl, S, M, W, t):
    for j in range(M):
        if t >= j * W // M and (t - j * W // M) % W == 0:
            rows = torch.arange(S, device=scache.ssm_states[0].device) * M + j
            for l in range(nl):
                scache.ssm_states[l][rows] = 0


def oldest(S, M, W, t1, dev):
    ages = [((t1 - j * W // M) % W) if t1 >= j * W // M else -1 for j in range(M)]
    return torch.arange(S, device=dev) * M + int(np.argmax(ages))


def train_gate(model, gate, X, W, M, every, seg, mu, sd, lr, dev, dtype, log):
    nl = model.config.num_hidden_layers
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    S, T = X.shape[0], X.shape[1] - 1
    Xd = X.to(dev)
    lc, sc = make_cache(model, S, dev, dtype), make_cache(model, S * M, dev, dtype)
    t = 0
    while t < T:
        if t + 1 >= W and (t + 1) % every == 0 and t + seg < T:
            with torch.no_grad():
                fwd(model, sc, Xd[:, t].repeat_interleave(M)); shadow_reset(sc, nl, S, M, W, t + 1)
            # apply gate (with grad) to the long state after consuming token t
            with torch.no_grad():
                fwd(model, lc, Xd[:, t])
            sh = oldest(S, M, W, t + 1, dev)
            gs = []
            for l in range(nl):
                new, g = gate(l, lc.ssm_states[l].detach().float(), sc.ssm_states[l][sh].float(), mu[l], sd[l])
                lc.ssm_states[l] = new.to(dtype); gs.append(g.mean().item())
            loss = 0
            for u in range(1, seg + 1):
                with torch.no_grad():
                    fwd(model, sc, Xd[:, t + u].repeat_interleave(M)); shadow_reset(sc, nl, S, M, W, t + u + 1)
                loss = loss + F.cross_entropy(fwd(model, lc, Xd[:, t + u]).float(), Xd[:, t + u + 1])
            loss = loss / seg
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0); opt.step()
            for l in range(nl):
                lc.ssm_states[l] = lc.ssm_states[l].detach(); lc.conv_states[l] = lc.conv_states[l].detach()
            log.append({"t": t, "loss": loss.item(), "g_mean": float(np.mean(gs))})
            t += seg + 1
        else:
            with torch.no_grad():
                fwd(model, sc, Xd[:, t].repeat_interleave(M)); shadow_reset(sc, nl, S, M, W, t + 1)
                fwd(model, lc, Xd[:, t])
            t += 1


@torch.no_grad()
def eval_pass(model, gate, X, W, M, every, mu, sd, dev, dtype, arms=("none", "window", "obs3", "learned")):
    nl = model.config.num_hidden_layers
    S, T, R = X.shape[0], X.shape[1] - 1, len(arms)
    Xd = X.to(dev)
    lc, sc = make_cache(model, S * R, dev, dtype), make_cache(model, S * M, dev, dtype)
    base = torch.arange(S, device=dev) * R
    loss = torch.zeros(S, R, T)
    for t in range(T):
        fwd(model, sc, Xd[:, t].repeat_interleave(M)); shadow_reset(sc, nl, S, M, W, t + 1)
        lg = fwd(model, lc, Xd[:, t].repeat_interleave(R)).float()
        loss[:, :, t] = F.cross_entropy(lg, Xd[:, t + 1].repeat_interleave(R), reduction="none").view(S, R).cpu()
        if t + 1 >= W and (t + 1) % every == 0:
            sh = oldest(S, M, W, t + 1, dev)
            for l in range(nl):
                st = lc.ssm_states[l]; shadow = sc.ssm_states[l][sh].float()
                for i, arm in enumerate(arms):
                    cur = st[base + i].float()
                    if arm == "window":
                        st[base + i] = shadow.to(st.dtype)
                    elif arm == "obs3":
                        bad = (cur - mu[l]).abs() > 3 * sd[l]
                        st[base + i] = torch.where(bad, shadow, cur).to(st.dtype)
                    elif arm == "learned":
                        st[base + i] = gate(l, cur, shadow, mu[l], sd[l])[0].to(st.dtype)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--W", type=int, default=2048)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--every", type=int, default=64)
    ap.add_argument("--seg", type=int, default=32, help="truncated-backprop tokens after each gate application")
    ap.add_argument("--train_len", type=int, default=8192)
    ap.add_argument("--train_seqs", type=int, default=8)
    ap.add_argument("--train_batches", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--L", type=int, default=12288)
    ap.add_argument("--n_seq", type=int, default=12)
    ap.add_argument("--n_recall", type=int, default=16)
    ap.add_argument("--gaps", default="512,3072")
    ap.add_argument("--n_calib", type=int, default=64)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_mcc3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, dtype = a.device, torch.float32
    t0 = time.time()
    tok, model = load(a.model, dev, dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    nl, D = model.config.num_hidden_layers, model.config.intermediate_size
    mu, sd = calib_stats(model, tok, os.path.join(a.data, "train.bin"), a.W, a.n_calib, dev, dtype)
    gate = Gate(nl, D).to(dev)
    log = []
    for b in range(a.train_batches):
        tr = tokens(tok, os.path.join(a.data, "train.bin"), a.train_seqs * (a.train_len + 1), 0.05 + 0.12 * b)
        k = len(tr) // (a.train_len + 1)
        train_gate(model, gate, torch.tensor(tr[: k * (a.train_len + 1)]).view(k, a.train_len + 1),
                   a.W, a.M, a.every, a.seg, mu, sd, a.lr, dev, dtype, log)
        print(f"[{a.model}] train batch {b + 1}/{a.train_batches}: last loss {np.mean([x['loss'] for x in log[-20:]]):.3f} "
              f"g_mean {log[-1]['g_mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    gate.eval()
    arms = ["none", "window", "obs3", "learned"]
    res = {"model": a.model, "W": a.W, "arms": arms, "train_log": log, "gate_bias_mean": float(torch.sigmoid(gate.b.detach()).mean())}
    ev = tokens(tok, os.path.join(a.data, "eval.bin"), a.n_seq * (a.L + 1), 0.0)
    k = len(ev) // (a.L + 1)
    loss = eval_pass(model, gate, torch.tensor(ev[: k * (a.L + 1)]).view(k, a.L + 1), a.W, a.M, a.every, mu, sd, dev, dtype)
    res["ppl"] = {"L": a.L, "loss_curve": {n: loss[:, i].mean(0).reshape(-1, 64).mean(1).tolist() for i, n in enumerate(arms)}}
    print(f"[{a.model}] ppl pass done ({time.time() - t0:.0f}s)", flush=True)
    P_all = tokens(tok, os.path.join(a.data, "eval.bin"), a.n_recall * 1024, 0.55)
    F_all = tokens(tok, os.path.join(a.data, "train.bin"), a.n_recall * (4096 + 3072 + 64), 0.3)
    res["recall"] = {}
    for G in [int(g) for g in a.gaps.split(",")]:
        seqs = []
        for i in range(a.n_recall):
            P = P_all[i * 1024:(i + 1) * 1024]; Fl = F_all[i * 7168:(i + 1) * 7168]
            if len(P) < 1024 or len(Fl) < 4096 + G:
                break
            seqs.append(Fl[:4096] + P + Fl[4096:4096 + G] + P + [P[0]])
        loss = eval_pass(model, gate, torch.tensor(seqs), a.W, a.M, a.every, mu, sd, dev, dtype)
        s1, s2 = 4096, 4096 + 1024 + G
        first, second = loss[:, :, s1 + 16:s1 + 1023].mean((0, 2)), loss[:, :, s2 + 16:s2 + 1023].mean((0, 2))
        res["recall"][str(G)] = {n: {"first": float(first[i]), "second": float(second[i]), "gain": float(first[i] - second[i])} for i, n in enumerate(arms)}
        print(f"[{a.model}] recall G={G}: " + " ".join(f"{n}:{float(first[i] - second[i]):+.3f}" for i, n in enumerate(arms)), flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, a.model.split("/")[-1] + ".json"), "w"), indent=1)
    torch.save(gate.state_dict(), os.path.join(a.out, a.model.split("/")[-1] + "_gate.pt"))
    print(f"[{a.model}] done ({res['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()

"""Stage 1: learn the loop dynamics INSIDE the trained depth (depths 1..T_train), on GSM8K train
(question + gold solution, teacher-forced). Nothing past T_train is ever used for fitting.

  PCA      residual-stream state (D=2048) -> r dims, fit on states at all trained depths
  DMD      z_{d+1} = A z_d + c on transitions d = fit_from .. T_train-1      (linear Koopman, identity observables)
  KAE      Koopman autoencoder: lifted y = [z, g(z)], y_{d+1} = K y_d + c, multi-step rollout over the
           trained depths; z read off y                                       (Fathi et al. 2024 style)
Spectra are capped below 1 so that rolled-out predictions cannot blow up.
Saves <out>/fit.pt.
"""
import os

import torch
import torch.nn as nn

from .common import load, parser, save_json
from .data import gsm8k


class KAE(nn.Module):
    def __init__(self, r, m=128, h=1024):
        super().__init__()
        self.r = r
        self.g = nn.Sequential(nn.Linear(r, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, m))
        self.K = nn.Parameter(torch.eye(r + m) * 0.95)
        self.c = nn.Parameter(torch.zeros(r + m))

    def lift(self, z):
        return torch.cat([z, self.g(z)], -1)

    def step(self, y):
        return y @ self.K.T + self.c


def cap_spectrum(A, cap=0.99):
    rho = torch.linalg.eigvals(A.double().cpu()).abs().max().item()
    return (A * (cap / rho) if rho > cap else A), rho


@torch.no_grad()
def collect(a, tok, model, ctl, rows, n_tok_per_seq):
    """Z[d] = [N_tok, D] states at depth d (1..t_train) for the same sampled token positions."""
    g = torch.Generator().manual_seed(a.seed)
    buf = {d: [] for d in range(1, a.t_train + 1)}
    texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows]
    for i in range(0, len(texts), a.batch_size):
        enc = tok(texts[i: i + a.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to(a.device)
        model(**enc, use_cache=False)
        mask = enc["attention_mask"].bool()
        for b in range(mask.shape[0]):
            valid = torch.nonzero(mask[b]).flatten().cpu()
            pick = valid[torch.randint(len(valid), (min(n_tok_per_seq, len(valid)),), generator=g)]
            for d in range(1, a.t_train + 1):
                buf[d].append(ctl.states[d - 1][b, pick.to(a.device)].float().cpu())
    return {d: torch.cat(v) for d, v in buf.items()}


def main():
    ap = parser(__doc__)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--tok_per_seq", type=int, default=64)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--fit_from", type=int, default=2, help="first depth whose transition d->d+1 is used")
    ap.add_argument("--kae_epochs", type=int, default=30)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok, model, ctl = load(a, loops=a.t_train)
    rows = gsm8k(a.data_dir, "train")[: a.n_train]
    Z = collect(a, tok, model, ctl, rows, a.tok_per_seq)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    dev = a.device
    allS = torch.cat([Z[d] for d in Z]).to(dev)
    mu = allS.mean(0)
    _, sv, V = torch.pca_lowrank(allS - mu, q=a.rank + 16, center=False, niter=6)
    W = V[:, : a.rank].contiguous()
    evr = float((sv[: a.rank] ** 2).sum() / ((allS - mu) ** 2).sum())
    P = {d: (Z[d].to(dev) - mu) @ W for d in Z}
    print(f"[fit] {allS.shape[0] // a.t_train} tokens x {a.t_train} depths, PCA {a.rank}: explained var {evr:.3f}")

    # DMD on transitions fit_from..t_train-1
    X = torch.cat([P[d] for d in range(a.fit_from, a.t_train)]).double()
    Y = torch.cat([P[d + 1] for d in range(a.fit_from, a.t_train)]).double()
    Phi = torch.cat([X, torch.ones(len(X), 1, device=dev, dtype=torch.float64)], 1)
    G = Phi.T @ Phi
    Wt = torch.linalg.solve(G + 1e-3 * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], device=dev, dtype=torch.float64), Phi.T @ Y)
    A, c = Wt[:-1].T.float(), Wt[-1].float()
    resid = ((Phi @ Wt - Y) ** 2).sum() / ((Y - X) ** 2).sum()
    A, rho = cap_spectrum(A)
    print(f"[fit] DMD: in-sample MSE / persistence MSE = {float(resid):.3f}, spectral radius {rho:.4f} (capped 0.99)")

    # KAE on full trained trajectories (depth 1..t_train), rollout from fit_from
    traj = torch.stack([P[d] for d in range(1, a.t_train + 1)], 1)  # [N, T, r]
    sd = traj.reshape(-1, a.rank).std(0) + 1e-6
    kae = KAE(a.rank).to(dev)
    opt = torch.optim.AdamW(kae.parameters(), 1e-3, weight_decay=1e-4)
    start = a.fit_from - 1
    for ep in range(a.kae_epochs):
        perm = torch.randperm(len(traj), device=dev)
        tot = 0.0
        for b in perm.split(1024):
            t = traj[b] / sd
            y = kae.lift(t[:, start])
            loss = 0.0
            for k in range(start + 1, a.t_train):
                y = kae.step(y)
                tgt = kae.lift(t[:, k])
                loss = loss + ((y[:, : a.rank] - t[:, k]) ** 2).mean() + 0.5 * ((y[:, a.rank:] - tgt[:, a.rank:].detach()) ** 2).mean()
            loss = loss + 0.1 * torch.relu(1 - kae.g(t[:, start]).std(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == a.kae_epochs - 1:
            print(f"    [kae] ep {ep} loss {tot / len(traj):.4f}")
    with torch.no_grad():
        kae.K.data, krho = cap_spectrum(kae.K.data)
    print(f"[fit] KAE spectral radius {krho:.4f} (capped 0.99)")
    os.makedirs(a.out, exist_ok=True)
    torch.save({"mu": mu.cpu(), "W": W.cpu(), "A": A.cpu(), "c": c.cpu(), "sd": sd.cpu(), "kae": kae.state_dict(),
                "rank": a.rank, "t_train": a.t_train, "fit_from": a.fit_from}, os.path.join(a.out, "fit.pt"))
    save_json({"pca_evr": evr, "dmd_resid_over_persist": float(resid), "dmd_rho": rho, "kae_rho": krho,
               "n_tokens": int(allS.shape[0] // a.t_train)}, a, "fit.json")


if __name__ == "__main__":
    main()

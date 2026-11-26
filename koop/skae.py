"""Sparse Koopman autoencoder (after Li, Tadipatri, Fathi, Chandar, Goroshin 2026, arXiv:2608.29057).
Encoder -> top-k sparse latent z (only k of m coefficients active), one linear Koopman map K on z, decoder back.
The SUPPORT (which k coefficients are active) is a label-free regime variable. A state whose support overlaps
poorly with every support seen in training is outside the known regimes ("support novelty").
Inputs are whitened PCA coordinates of frozen model states (fit inside this module)."""
import numpy as np, torch, torch.nn as nn


class SKAE(nn.Module):
    def __init__(s, d, m=256, k=16, h=512):
        super().__init__()
        s.k = k
        s.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, m))
        s.dec = nn.Sequential(nn.Linear(m, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))
        s.K = nn.Linear(m, m, bias=False)
        nn.init.eye_(s.K.weight)

    def code(s, x):
        a = s.enc(x)
        idx = a.abs().topk(s.k, -1).indices
        return torch.zeros_like(a).scatter(-1, idx, a.gather(-1, idx))

    def step(s, z):   # the Koopman step keeps the latent sparse (top-k again)
        a = s.K(z)
        idx = a.abs().topk(s.k, -1).indices
        return torch.zeros_like(a).scatter(-1, idx, a.gather(-1, idx))


class Space:
    """PCA(r) + whitening of raw states, fitted once; maps [N, D] torch -> [N, r]"""
    def __init__(s, X, r=256):
        X = torch.as_tensor(X, dtype=torch.float64)
        s.mu = X.mean(0)
        U, S, V = torch.linalg.svd(X - s.mu, full_matrices=False)
        s.V = V[:r].T.contiguous(); s.sd = (S[:r] / np.sqrt(len(X) - 1)).clamp_min(1e-8)

    def __call__(s, X):
        return (((torch.as_tensor(X, dtype=torch.float64) - s.mu) @ s.V) / s.sd).float()

    def state(s):
        return {'mu': s.mu, 'V': s.V, 'sd': s.sd}

    @classmethod
    def load(cls, st):
        o = cls.__new__(cls); o.mu, o.V, o.sd = st['mu'], st['V'], st['sd']; return o


def train_skae(seqs, d, m=256, k=16, steps=4000, L=4, bs=256, lr=1e-3, dev='cpu', seed=0, log=print):
    """seqs: float tensor [N, T, d] (whitened); trains on windows of length L+1.
    loss = recon + latent prediction (L steps) + decoded prediction. Returns the model (eval mode)."""
    torch.manual_seed(seed)
    net = SKAE(d, m, k).to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr)
    N, T, _ = seqs.shape; L = min(L, T - 1)
    seqs = seqs.to(dev)
    for it in range(steps):
        i = torch.randint(0, N, (bs,), device=dev); t0 = torch.randint(0, T - L, (bs,), device=dev)
        W = torch.stack([seqs[i, t0 + j] for j in range(L + 1)], 1)      # [bs, L+1, d]
        Z = net.code(W.flatten(0, 1)).view(bs, L + 1, -1)
        rec = (net.dec(Z) - W).pow(2).mean()
        z, lp, xp = Z[:, 0], 0.0, 0.0
        for j in range(1, L + 1):
            z = net.step(z)
            lp = lp + (z - Z[:, j].detach()).pow(2).mean() + (z.detach() - Z[:, j]).pow(2).mean() * 0.1
            xp = xp + (net.dec(z) - W[:, j]).pow(2).mean()
        loss = rec + (lp + xp) / L
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == steps - 1:
            log(f'[skae] step {it} loss {loss.item():.4f} recon {rec.item():.4f} pred {float(xp.detach() if torch.is_tensor(xp) else xp) / L:.4f}')
    return net.eval()


@torch.no_grad()
def supports(net, X, bs=8192):
    """binary support matrix [N, m] (uint8 on CPU)"""
    out = []
    for i in range(0, len(X), bs):
        out.append((net.code(X[i:i + bs]) != 0).to(torch.uint8).cpu())
    return torch.cat(out)


@torch.no_grad()
def novelty(Sq, Sbank, k, bs=4096):
    """1 - max Jaccard overlap of each query support with any bank support. Sq [n, m], Sbank [M, m] uint8"""
    B = Sbank.float(); out = []
    for i in range(0, len(Sq), bs):
        c = Sq[i:i + bs].float() @ B.T                  # overlaps
        out.append(1 - (c / (2 * k - c)).max(1).values)
    return torch.cat(out)


@torch.no_grad()
def pred_residual(net, X0, X1):
    """relative one-step Koopman prediction error in input space, per row"""
    xp = net.dec(net.step(net.code(X0)))
    return (xp - X1).pow(2).sum(-1) / X1.pow(2).sum(-1).clamp_min(1e-8)


def fit_regimes(S, R=8, seed=0):
    """k-means on binary supports -> centroids [R, m] (float). On a two-basin toy system this recovers the basins
    with 98% purity, where the largest-coefficient rule does not."""
    from sklearn.cluster import KMeans
    km = KMeans(R, n_init=5, random_state=seed).fit(S.float().numpy())
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)


def assign(S, C):
    """nearest centroid for each binary support row"""
    return torch.cdist(S.float().to(C.device), C).argmin(-1)


def load_skae(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = SKAE(ck['d'], ck['m'], ck['k']); net.load_state_dict(ck['net']); net = net.to(dev).eval()
    sp = Space.load({k: v.to(dev) for k, v in ck['space'].items()})
    return net, sp, ck

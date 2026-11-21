"""Faithful Fathi-style Koopman autoencoder world model with control (no recurrent filter).
State h = phi(o) in R^m (overcomplete), dynamics h' = Lambda h + B a + W((U h)*(V a)), o_hat = psi(h).
Course correction = h <- phi(psi(h)) (decode -> re-encode), exactly as in Fathi et al. 2024.
Same interface as RSSM (z is an empty tensor) so evaluate/diag/plan treat both uniformly."""
import torch, torch.nn as nn
from wm.model import mlp, KoopTrans


class KAE(nn.Module):
    def __init__(s, obs_dim, act_dim, m=64, hid=256):
        super().__init__()
        s.m = m
        s.phi = mlp(obs_dim, m, hid)
        s.psi = mlp(m, obs_dim, hid)
        s.trans = KoopTrans(m, act_dim)
        s.latent = 'none'; s.C = 1

    def _z(s, h):
        return h.new_zeros(*h.shape[:-1], 0)

    # interface used by evaluate / diag / plan
    def enc(s, o):
        return s.phi(o)

    def encode_state(s, e, mode=True):
        return e, s._z(e)

    def decode(s, h, z=None):
        return s.psi(h)

    def step(s, h, z, a):
        return s.trans(h, a)

    def observe(s, obs, act, n_ctx=None, mode=True):
        L = obs.shape[1] if n_ctx is None else n_ctx + 1
        hs = s.phi(obs[:, :L])
        zero = obs.new_zeros(obs.shape[0], max(L - 1, 1))
        return hs, s._z(hs), zero, zero, zero

    def imagine(s, h, z, acts, reenc_mask_fn=None, kind='full', score_fn=None):
        preds, n_re = [], torch.zeros(h.shape[0], device=h.device)
        for t in range(acts.shape[1]):
            h = s.trans(h, acts[:, t])
            o = s.psi(h)
            preds.append(o)
            if reenc_mask_fn is not None:
                h_new = s.phi(o)
                resid = (h_new - h).norm(dim=-1) / (h.norm(dim=-1) + 1e-6) if score_fn is None else score_fn(h)
                msk = reenc_mask_fn(t, resid)
                if msk is not None and msk.any():
                    mf = msk.float().unsqueeze(-1)
                    h = mf * h_new + (1 - mf) * h
                    n_re = n_re + msk.float()
        return torch.stack(preds, 1), n_re

    def losses(s, o, a, K):
        """Lusch/Fathi objective: reconstruction + K-step prediction in obs space + latent linearity."""
        B, L1, _ = o.shape
        z_all = s.phi(o)
        rec = (s.psi(z_all) - o).pow(2).mean()
        st = torch.randint(0, L1 - K, (1,)).item()
        h = z_all[:, st]
        pred, lin = 0.0, 0.0
        for k in range(1, K + 1):
            h = s.trans(h, a[:, st + k - 1])
            if getattr(s, 'noise', 0) > 0:   # noise injection baseline
                h = h + s.noise * h.detach().std() * torch.randn_like(h)
            pred = pred + (s.psi(h) - o[:, st + k]).pow(2).mean()
            lin = lin + (h - z_all[:, st + k]).pow(2).mean()
        pred, lin = pred / K, lin / K
        loss = rec + pred + 0.1 * lin
        return loss, {'rec': rec.item(), 'pred': pred.item(), 'lin': lin.item()}

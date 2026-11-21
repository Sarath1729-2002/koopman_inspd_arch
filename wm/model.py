"""Dreamer-style recurrent state-space world model (RSSM) for proprioceptive MuJoCo.
State = (h, z): h deterministic (GRU or Koopman-style linear transition), z stochastic
(discrete: C categoricals x K classes, DreamerV3; or continuous Gaussian).
Course correction = decode the imagined state to an observation, re-encode it (h <- init(enc(o)),
z <- posterior(h, enc(o))), continue imagining."""
import torch, torch.nn as nn, torch.nn.functional as F


def mlp(i, o, hid=256, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.SiLU()]
        d = hid
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class KoopTrans(nn.Module):
    """h' = Lambda h + B u + W((U h) * (V u)); Lambda = 2x2 rotation-scaling blocks, radius in (0.5, 1)."""
    def __init__(s, D, u_dim, rank=16):
        super().__init__()
        assert D % 2 == 0
        s.rho = nn.Parameter(torch.full((D // 2,), 2.0))
        s.theta = nn.Parameter(torch.rand(D // 2) * 3.14159)
        s.B = nn.Linear(u_dim, D)
        s.U, s.V, s.W = nn.Linear(D, rank, bias=False), nn.Linear(u_dim, rank, bias=False), nn.Linear(rank, D, bias=False)

    def forward(s, h, u):
        r = 0.5 + 0.5 * torch.sigmoid(s.rho)
        c, si = r * torch.cos(s.theta), r * torch.sin(s.theta)
        h2 = h.view(*h.shape[:-1], -1, 2)
        x, y = h2[..., 0], h2[..., 1]
        lin = torch.stack([c * x - si * y, si * x + c * y], -1).view_as(h)
        return lin + s.B(u) + s.W(s.U(h) * s.V(u))


class GRUTrans(nn.Module):
    def __init__(s, D, u_dim, hid=256):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(u_dim, hid), nn.LayerNorm(hid), nn.SiLU())
        s.cell = nn.GRUCell(hid, D)

    def forward(s, h, u):
        return s.cell(s.inp(u), h)


class RSSM(nn.Module):
    def __init__(s, obs_dim, act_dim, trans='koop', latent='disc', D=256, C=16, K=16, G=32, hid=256, E=256):
        super().__init__()
        s.latent, s.C, s.K = latent, C, K
        s.zdim = C * K if latent == 'disc' else G
        s.enc = mlp(obs_dim, E, hid)
        s.init = nn.Sequential(mlp(E, D, hid), nn.Tanh()) if trans == 'gru' else mlp(E, D, hid)
        u_dim = s.zdim + act_dim
        s.trans = GRUTrans(D, u_dim, hid) if trans == 'gru' else KoopTrans(D, u_dim)
        out = s.zdim if latent == 'disc' else 2 * G
        s.prior = mlp(D, out, hid, 1)
        s.post = mlp(D + E, out, hid, 1)
        s.dec = mlp(D + s.zdim, obs_dim, hid)

    # ---- latent distributions -------------------------------------------------------------
    def dist(s, raw):
        if s.latent == 'disc':
            p = F.softmax(raw.view(*raw.shape[:-1], s.C, s.K), -1)
            return 0.99 * p + 0.01 / s.K
        mu, sd = raw.chunk(2, -1)
        return mu, F.softplus(sd) + 0.1

    def sample(s, d, mode=False):
        if s.latent == 'disc':
            idx = d.argmax(-1) if mode else torch.distributions.Categorical(probs=d).sample()
            oh = F.one_hot(idx, s.K).to(d.dtype)
            return (oh + d - d.detach()).flatten(-2)
        mu, sd = d
        return mu if mode else mu + sd * torch.randn_like(mu)

    def kl(s, q, p):  # KL(q || p), summed over latent dims
        if s.latent == 'disc':
            return (q * (q.log() - p.log())).sum((-1, -2))
        (mq, sq), (mp, sp) = q, p
        return (torch.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5).sum(-1)

    @staticmethod
    def sg(d):
        return tuple(x.detach() for x in d) if isinstance(d, tuple) else d.detach()

    # ---- core ops -------------------------------------------------------------------------
    def step(s, h, z, a):
        return s.trans(h, torch.cat([z, a], -1))

    def decode(s, h, z):
        return s.dec(torch.cat([h, z], -1))

    def encode_state(s, e, mode=False):
        """Fresh state from a single observation embedding (the fully-observed re-encoding map)."""
        h = s.init(e)
        return h, s.sample(s.dist(s.post(torch.cat([h, e], -1))), mode)

    def observe(s, obs, act, n_ctx=None, mode=False):
        """Filter obs[:, :n_ctx+1] (posterior). Returns per-step h, z, kl terms, init-consistency."""
        L = obs.shape[1] if n_ctx is None else n_ctx + 1
        e = s.enc(obs[:, :L])
        h, z = s.encode_state(e[:, 0], mode)
        hs, zs, dyn, rep, cons = [h], [z], [], [], []
        for t in range(1, L):
            h = s.step(h, z, act[:, t - 1])
            p = s.dist(s.prior(h))
            q = s.dist(s.post(torch.cat([h, e[:, t]], -1)))
            z = s.sample(q, mode)
            dyn.append(s.kl(s.sg(q), p)); rep.append(s.kl(q, s.sg(p)))
            if t >= 4:  # teach init(enc(o_t)) to land on the filtered manifold
                cons.append((s.init(e[:, t].detach()) - h.detach()).pow(2).mean(-1))
            hs.append(h); zs.append(z)
        return torch.stack(hs, 1), torch.stack(zs, 1), torch.stack(dyn, 1), torch.stack(rep, 1), \
            (torch.stack(cons, 1) if cons else torch.zeros(1, device=obs.device))

    def imagine(s, h, z, acts, reenc_mask_fn=None, kind='full', score_fn=None):
        """Open-loop prior rollout (mode latents) with true actions acts [B,H,A].
        reenc_mask_fn(t, h, z, o_hat, resid) -> bool [B] : where to re-encode after step t.
        Returns decoded predictions [B,H,O] and #re-encodes [B]."""
        preds, n_re = [], torch.zeros(h.shape[0], device=h.device)
        for t in range(acts.shape[1]):
            h = s.step(h, z, acts[:, t])
            if s.training and getattr(s, 'noise', 0) > 0:   # noise injection baseline (training only)
                h = h + s.noise * h.detach().std() * torch.randn_like(h)
            z = s.sample(s.dist(s.prior(h)), mode=True)
            o = s.decode(h, z)
            preds.append(o)
            if reenc_mask_fn is not None:
                e = s.enc(o)
                if kind == 'full':   # fresh state from the decoded observation
                    h_new, z_new = s.encode_state(e, mode=True)
                    resid = (h_new - h).norm(dim=-1) / (h.norm(dim=-1) + 1e-6)
                else:                # keep h, snap the discrete code to the posterior of the decoded obs
                    h_new = h
                    z_new = s.sample(s.dist(s.post(torch.cat([h, e], -1))), mode=True)
                    resid = (z_new - z).abs().sum(-1) / 2 / (s.C if s.latent == 'disc' else 1)
                if score_fn is not None:
                    resid = score_fn(h)
                m = reenc_mask_fn(t, resid)
                if m is not None and m.any():
                    mf = m.float().unsqueeze(-1)
                    h = mf * h_new + (1 - mf) * h
                    z = mf * z_new + (1 - mf) * z
                    n_re = n_re + m.float()
        return torch.stack(preds, 1), n_re

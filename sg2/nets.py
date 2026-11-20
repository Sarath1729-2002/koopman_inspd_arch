"""Flow-matching denoisers over 48 future frames of 5 balls, with block-causal factorisations (as in the Seriality Gap
paper: bidirectional / Block-k / autoregressive differ only in the attention mask), plus a Koopman latent model.

Token layout (101 tokens): [5 context frames (clean)] [48 clean future frames] [48 noisy future frames].
A noisy future frame t sees: the context, clean future frames j < g_t (already generated), and noisy frames
j in [g_t, e_t) (its own block, denoised jointly). Clean future frame j sees the context and clean frames <= j.
  block-k     g_t = k*floor(t/k), e_t = min(48, g_t + k)    k=48: parallel/bidirectional, k=1: autoregressive
  prefix-s    for t >= s: g_t = s, e_t = 48                 (parallel generation of the rest after a given prefix)
Optional per-noisy-frame conditioning vector (Koopman latent rollout, or the capacity-matched feature control)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BALL, CTX, H = 5, 5, 48
FD = 2 * N_BALL
L_TOK = CTX + 2 * H


def norm_pos(x):          # positions in [0.07, 0.93] -> roughly unit scale
    return (x - 0.5) / 0.25


def denorm_pos(x):
    return x * 0.25 + 0.5


def build_mask(g, e):
    """g, e: [B, 48] long. Returns bool [B, 1, L, L], True = may attend."""
    B, dev = g.shape[0], g.device
    L = L_TOK
    M = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
    M[:, :CTX, :CTX] = True                                               # context <-> context
    j = torch.arange(H, device=dev)
    # clean future j: context + clean <= j
    M[:, CTX:CTX + H, :CTX] = True
    M[:, CTX:CTX + H, CTX:CTX + H] = (j[None, :, None] >= j[None, None, :]).expand(B, -1, -1)
    # noisy t: context + clean j < g_t + noisy j in [g_t, e_t)
    nz = slice(CTX + H, L)
    M[:, nz, :CTX] = True
    M[:, nz, CTX:CTX + H] = j[None, None, :] < g[:, :, None]
    M[:, nz, nz] = (j[None, None, :] >= g[:, :, None]) & (j[None, None, :] < e[:, :, None])
    M[:, torch.arange(L), torch.arange(L)] = True
    return M[:, None]


def block_bounds(k, B, dev):
    t = torch.arange(H, device=dev)
    g = (t // k) * k
    return g.expand(B, -1).contiguous(), (g + k).clamp(max=H).expand(B, -1).contiguous()


def prefix_bounds(s):
    """s: [B] number of given future frames."""
    B, dev = s.shape[0], s.device
    t = torch.arange(H, device=dev)[None].expand(B, -1)
    g = torch.where(t >= s[:, None], s[:, None].expand(-1, H), t)          # given frames: only see themselves
    e = torch.where(t >= s[:, None], torch.full_like(t, H), t + 1)
    return g, e


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.l1, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.l1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.l2(x))


def tau_embed(tau, d):
    half = d // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / half)
    a = tau[..., None] * 1000.0 * f
    return torch.cat([a.sin(), a.cos()], -1)


class Denoiser(nn.Module):
    def __init__(self, d=256, L=8, h=8, cond_dim=0):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(FD, d)
        self.frame = nn.Embedding(CTX + H, d)
        self.kind = nn.Embedding(3, d)                     # 0 context, 1 clean future, 2 noisy future
        self.tau = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond = nn.Sequential(nn.Linear(cond_dim, d), nn.SiLU(), nn.Linear(d, d)) if cond_dim else None
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, FD)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, ctx, clean, noisy, tau, mask, cond=None):
        """ctx [B,5,10], clean [B,48,10], noisy [B,48,10], tau [B,48], mask [B,1,L,L], cond [B,48,c] -> v [B,48,10]"""
        dev = ctx.device
        fr = self.frame(torch.arange(CTX + H, device=dev))
        x_c = self.inp(ctx) + fr[:CTX] + self.kind.weight[0]
        x_f = self.inp(clean) + fr[CTX:] + self.kind.weight[1]
        x_n = self.inp(noisy) + fr[CTX:] + self.kind.weight[2] + self.tau(tau_embed(tau, self.d))
        if self.cond is not None:
            x_n = x_n + self.cond(cond)
        x = torch.cat([x_c, x_f, x_n], 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.out(self.ln(x[:, CTX + H:]))


class Koop(nn.Module):
    """Koopman latent dynamics on normalised positions: z = phi(5-frame window), z_{t+n} = Lambda^n z_t (closed form)."""

    def __init__(self, m=128, hid=512):
        super().__init__()
        self.m = m
        self.enc = nn.Sequential(nn.Linear(CTX * FD, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, m))
        self.dec = nn.Sequential(nn.Linear(m, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, FD))
        k = m // 2
        self.rho = nn.Parameter(torch.full((k,), 4.0))
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi / 2, k))

    def encode(self, win):
        return self.enc(win.reshape(win.shape[0], -1))

    def roll(self, z, steps):
        """z [B, m], steps [B, T] (>= 0) -> [B, T, m]"""
        r = torch.sigmoid(self.rho)
        k = self.m // 2
        a, b = z[:, None, :k], z[:, None, k:]
        tt = steps.float()[..., None]
        mag, ang = r ** tt, self.theta * tt
        c, s = mag * ang.cos(), mag * ang.sin()
        return torch.cat([c * a - s * b, s * a + c * b], -1)


class FeatControl(nn.Module):
    """Capacity-matched control: per-frame conditioning from an MLP of (window, steps ahead) with NO linear-dynamics
    structure. Same inputs and output size as the Koopman rollout."""

    def __init__(self, m=128, hid=512):
        super().__init__()
        self.m = m
        self.net = nn.Sequential(nn.Linear(CTX * FD + 32, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, m))

    def features(self, win, steps):
        B, T = steps.shape
        te = tau_embed(steps.float() / H, 32)                               # [B, T, 32]
        w = win.reshape(B, 1, -1).expand(-1, T, -1)
        return self.net(torch.cat([w, te], -1))

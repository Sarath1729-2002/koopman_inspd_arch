"""Looped decoder-only transformer (GPT-2 blocks, NoPE, causal attention, input injection) with four
loop updates. With u = h + e (e = input embedding) and f(u) = Block(u) - u (the block's residual update):

  A  plain      h <- u + f(u)                         = Block(h + e)            (Fan et al.)
  B  gainctl    h <- LN(Block(h + e))                                           (normalisation control)
  C  koopman    h <- Lambda h + e + f(u)              Lambda: block-diagonal 2x2 rotation-scaling,
                                                      radius in (0, 1]  -> spectrally bounded linear backbone
                                                      (Koopman Dreamer / LRU-style), nonlinear f on top
  D  koopman+re C, and every k loops re-encode through the output space (course correction):
                h <- g*h + (1-g) * softmax(head(LN h)) @ E_re      (decode -> re-embed, Fathi et al. 2024)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = heads

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class RotScale(nn.Module):
    """Block-diagonal real Koopman operator: d/2 complex modes r*e^{i theta}, 0 < r <= r_max."""

    def __init__(self, d, r_init=0.97, r_max=1.0):
        super().__init__()
        self.r_max = r_max
        p = math.log(r_init / (r_max - r_init))
        self.rho = nn.Parameter(torch.full((d // 2,), p) + 0.1 * torch.randn(d // 2))
        self.theta = nn.Parameter((torch.rand(d // 2) - 0.5) * (math.pi / 8))

    def radius(self):
        return self.r_max * torch.sigmoid(self.rho)

    def forward(self, h):
        r, th = self.radius(), self.theta
        x, y = h[..., 0::2], h[..., 1::2]
        c, s = r * torch.cos(th), r * torch.sin(th)
        out = torch.empty_like(h)
        out[..., 0::2] = c * x - s * y
        out[..., 1::2] = s * x + c * y
        return out


class Looped(nn.Module):
    def __init__(self, vocab, arm, d=256, heads=8, depth=1, reenc_every=4):
        super().__init__()
        self.arm, self.k = arm, reenc_every
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if arm == "gainctl":
            self.gain = nn.LayerNorm(d)
        if arm in ("koopman", "koopman_re"):
            self.lam = RotScale(d)
        if arm == "koopman_re":
            self.e_re = nn.Embedding(vocab, d)
            self.gate = nn.Parameter(torch.zeros(d))

    def stack(self, u):
        for b in self.blocks:
            u = b(u)
        return u

    def step(self, h, e, t):
        if self.arm == "plain":
            return self.stack(h + e)
        if self.arm == "gainctl":
            return self.gain(self.stack(h + e))
        u = h + e
        h = self.lam(h) + e + (self.stack(u) - u)
        if self.arm == "koopman_re" and (t + 1) % self.k == 0:
            p = F.softmax(self.head(self.ln_f(h)).float(), -1).to(h.dtype)
            g = torch.sigmoid(self.gate)
            h = g * h + (1 - g) * (p @ self.e_re.weight)
        return h

    def logits(self, h):
        return self.head(self.ln_f(h))

    def forward(self, x, n_loops, read_at=None, keep_all=False):
        """Runs n_loops; returns logits at read_at[i] loops for sample i (or at every loop if keep_all)."""
        e = self.emb(x)
        h = e
        out = torch.zeros(x.shape[0], x.shape[1], self.head.out_features, device=x.device) if read_at is not None else None
        all_ = []
        for t in range(n_loops):
            h = self.step(h, e, t)
            if read_at is not None:
                m = read_at == (t + 1)
                if m.any():
                    out[m] = self.logits(h[m]).float()
            if keep_all:
                all_.append(self.logits(h).float())
        return (torch.stack(all_, 1) if keep_all else out), h

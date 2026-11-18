"""Byte-level LM with diagonal gated linear recurrences (mainstream cores, simplified to one state per channel):
  rglru   Griffin / RecurrentGemma RG-LRU: a_t = sigmoid(L)^(c r_t),  h = a h + sqrt(1-a^2) (i_t * u)   (normalised input)
  mamba   Mamba-style selective decay:     a_t = exp(-dt_t A),         h = a h + dt_t * u               (unnormalised input)
Each layer: RMSNorm -> [conv4 -> recurrence] * gelu/silu gate -> proj, residual; then an MLP block.
reenc_every=k>0 adds the TRAINED-IN course correction at every k-th token, inside every layer:
  h <- s * LN( h + MLP(LN h) )        (decode -> normalise -> re-encode; MLP's last layer starts at zero)
A test-time controller `ctrl(layer, h, pos) -> h` can also be called at every chunk boundary (post-hoc corrections).
The scan is chunked (chunk 32) with the cumulative-log trick; log a is clamped to [-1, 0] so it is stable in fp32."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 32


def chunk_scan(loga, b, every, hook):
    """h_t = a_t h_{t-1} + b_t. hook(h, pos) is applied to the carried state after token pos-1 when pos % every == 0."""
    B, L, D = b.shape
    h = torch.zeros(B, D, device=b.device, dtype=b.dtype)
    out = []
    for i in range(0, L, CH):
        Lc = torch.cumsum(loga[:, i:i + CH], 1)
        hs = torch.exp(Lc) * (h[:, None] + torch.cumsum(torch.exp(-Lc) * b[:, i:i + CH], 1))
        out.append(hs)
        h = hs[:, -1]
        pos = i + hs.shape[1]
        if hook is not None and pos % every == 0:
            h = hook(h, pos)
    return torch.cat(out, 1)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class RecLayer(nn.Module):
    def __init__(self, d, di, core, reenc_every=0):
        super().__init__()
        self.core, self.k = core, reenc_every
        self.norm = RMSNorm(d)
        self.inp, self.gate = nn.Linear(d, di), nn.Linear(d, di)
        self.conv = nn.Conv1d(di, di, 4, groups=di, padding=3)
        self.out = nn.Linear(di, d)
        if core == "rglru":
            self.wa, self.wi = nn.Linear(di, di), nn.Linear(di, di)
            amin, amax = 0.9, 0.999                                   # a^c range at init (Griffin)
            ac = amin + (amax - amin) * torch.rand(di)
            self.lam = nn.Parameter(torch.logit(ac ** (1 / 8.0)))
        else:
            self.wdt = nn.Linear(di, di)
            dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            nn.init.constant_(self.wdt.weight, 0.0)
            with torch.no_grad():
                self.wdt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1(dt)
            self.A_log = nn.Parameter(torch.zeros(di))                   # A = 1
            self.Dskip = nn.Parameter(torch.ones(di))
        if reenc_every:
            self.re = nn.Sequential(nn.LayerNorm(di), nn.Linear(di, di), nn.GELU(), nn.Linear(di, di))
            nn.init.zeros_(self.re[-1].weight); nn.init.zeros_(self.re[-1].bias)
            self.re_ln = nn.LayerNorm(di, elementwise_affine=False)
            self.re_s = nn.Parameter(torch.full((di,), 0.1))
        self.ctrl, self.idx, self.rec = None, 0, None

    def reenc(self, h, pos):
        return self.re_s * self.re_ln(h + self.re(h))

    def forward(self, x):
        z = self.norm(x)
        u = self.conv(self.inp(z).transpose(1, 2))[..., : x.shape[1]].transpose(1, 2).float()
        if self.core == "rglru":
            r, i = torch.sigmoid(self.wa(u)), torch.sigmoid(self.wi(u))
            loga = (8.0 * r * F.logsigmoid(self.lam)).clamp(min=-1.0)
            b = torch.sqrt((1 - torch.exp(2 * loga)).clamp(min=1e-8)) * (i * u)
            g = F.gelu(self.gate(z).float())
        else:
            dt = F.softplus(self.wdt(u))
            loga = (-dt * torch.exp(self.A_log)).clamp(min=-1.0)
            b = dt * u
            g = F.silu(self.gate(z).float())

        def hook(h, pos):
            if self.k and pos % self.k == 0:
                h = self.reenc(h, pos)
            if self.ctrl is not None:
                h = self.ctrl(self.idx, h, pos)
            return h

        hs = chunk_scan(loga, b, CH, hook if (self.k or self.ctrl is not None) else None)
        if self.rec is not None:
            self.rec(self.idx, hs)
        y = hs * g if self.core == "rglru" else (hs + self.Dskip * u) * g
        return x + self.out(y.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.n, self.f = RMSNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.f(self.n(x))


class ByteLM(nn.Module):
    def __init__(self, core="rglru", d=384, layers=6, reenc_every=0):
        super().__init__()
        assert reenc_every % CH == 0, f"reenc_every must be a multiple of {CH}"
        self.emb = nn.Embedding(256, d)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            rl = RecLayer(d, 2 * d, core, reenc_every); rl.idx = i
            self.blocks += [rl, MLP(d)]
        self.norm, self.head = RMSNorm(d), nn.Linear(d, 256, bias=False)

    def rec_layers(self):
        return [b for b in self.blocks if isinstance(b, RecLayer)]

    def set_ctrl(self, ctrl=None, rec=None):
        for l in self.rec_layers():
            l.ctrl, l.rec = ctrl, rec

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

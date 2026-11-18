"""Re-encoding as ERROR CORRECTION, not computation.

S5 showed the boundary map cannot do composition (needs |G|^k cases). But cleanup only needs |G| cases:
snap a drifting state back to the nearest valid code. This is Fathi et al.'s actual use of re-encoding.

Tasks (per-token labels, train length <= 64, test to 4096):
  parity    running XOR of bits                       (diagonal linear RNN solves it with eigenvalue -1)
  mod5      running sum mod 5 of tokens in {0..4}     (solved by rotations by 2*pi*j/5)
Both are exactly solvable by the linear core, so any long-length failure is DRIFT (angle/magnitude error
accumulating over time), which is exactly what periodic re-encoding is supposed to fix.

Arms:
  lstm     reference
  rot      h_t = r(x_t) e^{i theta(x_t)} h_{t-1}, r in (0,1], learned init state, MLP readout (LRU-style core)
  unit     rot with r = 1 exactly (unitary core: no decay; any failure is pure angle drift)
  norm{k}  rot, and every k tokens rescale h to the norm of h0      (norm-only cleanup, no decoding)
  re{k}    rot, and every k tokens: h <- softmax(readout(h)) @ E   (decode -> re-encode, Fathi et al. 2024)
  ure{k}   unit + re-encode every k
Re-encoding only counts if it beats BOTH unit and norm{k}: otherwise "norm preservation" already explains it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = {"parity": 2, "mod5": 5}


def make_batch(task, B, L, gen):
    p = TASKS[task]
    x = torch.randint(0, p, (B, L), generator=gen)
    return x, torch.cumsum(x, 1) % p


class Readout(nn.Module):
    def __init__(self, d, V, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, V))

    def forward(self, h):
        return self.net(h)


class LSTMNet(nn.Module):
    def __init__(self, V, d=128):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(V, d), nn.LSTM(d, d, batch_first=True), Readout(d, V)

    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])


class Rot(nn.Module):
    def __init__(self, V, d=128, k=0, mode="none", unit=False):
        super().__init__()
        self.k, self.m, self.mode, self.unit = k, d // 2, mode, unit
        self.rho = nn.Embedding(V, self.m)          # magnitude logits (per input token)
        self.theta = nn.Embedding(V, self.m)        # angle per input token
        nn.init.constant_(self.rho.weight, 3.0)     # r ~ 0.95
        nn.init.uniform_(self.theta.weight, -math.pi, math.pi)
        self.h0 = nn.Parameter(torch.randn(d) / math.sqrt(d) * 4)
        self.out = Readout(d, V)
        if mode == "re":
            self.E = nn.Parameter(torch.randn(V, d) / math.sqrt(d) * 4)   # re-encoding codebook

    def forward(self, x):
        B, L = x.shape
        th = self.theta(x)
        r = torch.ones_like(th) if self.unit else torch.sigmoid(self.rho(x))                  # [B, L, m]
        c, s = r * torch.cos(th), r * torch.sin(th)
        h = self.h0.expand(B, -1)
        hs = []
        for t in range(L):
            a, b = h[:, : self.m], h[:, self.m:]
            h = torch.cat([c[:, t] * a - s[:, t] * b, s[:, t] * a + c[:, t] * b], -1)
            if self.k and (t + 1) % self.k == 0:
                if self.mode == "re":
                    h = F.softmax(self.out(h), -1) @ self.E
                else:
                    h = h * (self.h0.norm() / h.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            hs.append(h)
        return self.out(torch.stack(hs, 1))


def build(arm, V):
    if arm == "lstm":
        return LSTMNet(V)
    if arm == "rot":
        return Rot(V)
    if arm == "unit":
        return Rot(V, unit=True)
    for pre, mode, unit in (("ure", "re", True), ("re", "re", False), ("norm", "norm", False)):
        if arm.startswith(pre):
            return Rot(V, k=int(arm[len(pre):]), mode=mode, unit=unit)
    raise ValueError(arm)

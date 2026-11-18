"""Arms.
  lstm       nonlinear RNN = "re-encode every token"            (upper reference, sequential)
  deltaprod  DeltaProduct: per token, n_h generalized Householder steps S <- S + beta (v - S k) k^T, beta in [0,2]
             (eigenvalues in [-1,1]); fully parallelisable in the real thing. n_h=4 because every permutation of
             5 items is a product of <= 4 transpositions (= reflections)      (the bar to beat)
  lrnn       input-dependent diagonal linear RNN with eigenvalues in (-1,1) (Grazzi-style, gets parity, not S5)
  bnd{k}     the proposal: lrnn inside chunks of k tokens, and at every chunk boundary the state is
             decoded -> normalised -> re-encoded:  h <- LN(MLP(h))   (periodic re-encoding along time)
             Every token is read out from its own state (mid-chunk tokens only see the linear carry).
All recurrences are written as plain sequential loops: this test is about expressivity, not throughput."""
import torch
import torch.nn as nn
import torch.nn.functional as F



class Readout(nn.Module):
    def __init__(self, d, V, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, V))

    def forward(self, h):
        return self.net(h)


class LSTMNet(nn.Module):
    def __init__(self, V, d=256):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(V, d), nn.LSTM(d, d, batch_first=True), Readout(d, V)

    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])


class BoundaryLRNN(nn.Module):
    def __init__(self, V, d=256, k=0, hidden=1024):
        super().__init__()
        self.k = k
        self.emb = nn.Embedding(V, d)
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)
        nn.init.constant_(self.a.bias, 1.5)  # a ~ tanh(1.5) ~ 0.9 at init
        if k:
            self.bnd = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
            self.bnd_ln = nn.LayerNorm(d, elementwise_affine=False)  # norm preservation at the boundary
        self.out = Readout(d, V)

    def forward(self, x):
        e = self.emb(x)
        A, Bv = torch.tanh(self.a(e)), self.b(e)
        h = torch.zeros(x.shape[0], e.shape[-1], device=x.device)
        hs = []
        for t in range(x.shape[1]):
            h = A[:, t] * h + Bv[:, t]
            if self.k and (t + 1) % self.k == 0:
                h = self.bnd_ln(self.bnd(h))
            hs.append(h)
        return self.out(torch.stack(hs, 1))


class DeltaProduct(nn.Module):
    def __init__(self, V, d=256, dk=64, nh=4):
        super().__init__()
        self.nh, self.dk = nh, dk
        self.emb = nn.Embedding(V, d)
        self.kp, self.vp = nn.Linear(d, nh * dk), nn.Linear(d, nh * dk)
        self.bp, self.qp = nn.Linear(d, nh), nn.Linear(d, dk)
        self.out = Readout(dk, V)

    def forward(self, x):
        B, L = x.shape
        e = self.emb(x)
        K = F.normalize(self.kp(e).view(B, L, self.nh, self.dk), dim=-1)
        Vv = self.vp(e).view(B, L, self.nh, self.dk)
        beta = 2 * torch.sigmoid(self.bp(e))  # [0, 2] -> eigenvalues of (I - beta k k^T) in [-1, 1]
        Q = self.qp(e)
        S = torch.zeros(B, self.dk, self.dk, device=x.device)
        os_ = []
        for t in range(L):
            for j in range(self.nh):
                k, v = K[:, t, j], Vv[:, t, j]
                Sk = torch.einsum("bij,bj->bi", S, k)
                S = S + beta[:, t, j, None, None] * (v - Sk)[:, :, None] * k[:, None, :]
            os_.append(torch.einsum("bij,bj->bi", S, Q[:, t]))
        return self.out(torch.stack(os_, 1))


def build(arm, V, d=256):
    if arm == "lstm":
        return LSTMNet(V, d)
    if arm == "deltaprod":
        return DeltaProduct(V, d)
    if arm == "lrnn":
        return BoundaryLRNN(V, d, 0)
    if arm.startswith("bnd"):
        return BoundaryLRNN(V, d, int(arm[3:]))
    raise ValueError(arm)

"""Four predictors of 48 future frames of 5 balls (positions) from 5 context frames.
  parallel  bidirectional transformer: context tokens + 48 time-query tokens, ONE forward pass (1 serial step)
  ar        causal transformer, frame-by-frame rollout (48 serial steps)
  koop      Koopman: encoder phi(5 frames) -> z in R^m (m/2 complex modes), z_t = Lambda^t z_0 in CLOSED FORM
            (all t in parallel), decoder psi(z_t) -> positions.  Re-encoding (Fathi et al.): at time tau,
            z_tau <- phi(last 5 DECODED frames) and continue.  Serial passes = 1 + #re-encodings.
            Event trigger (generic, Koopman-native): manifold residual  ||phi(decoded window at t) - z_t|| / ||z_t||,
            computable for all t in parallel; re-encode at the first t where it exceeds tau."""
import math

import torch
import torch.nn as nn

N, D2, CTX, H = 5, 2, 5, 48
FD = N * D2


class TBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.l1, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None):
        y = self.l1(x)
        x = x + self.att(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.l2(x))


class Parallel(nn.Module):
    def __init__(self, d=192, L=6, h=6):
        super().__init__()
        self.inp, self.time = nn.Linear(FD, d), nn.Embedding(CTX + H, d)
        self.q = nn.Parameter(torch.zeros(1, H, d))
        self.blocks = nn.ModuleList([TBlock(d, h) for _ in range(L)])
        self.out = nn.Linear(d, FD)

    def forward(self, ctx):                                   # ctx [B, 5, 10] -> [B, 48, 10]
        B = ctx.shape[0]
        t = torch.arange(CTX + H, device=ctx.device)
        x = torch.cat([self.inp(ctx), self.q.expand(B, -1, -1)], 1) + self.time(t)
        for b in self.blocks:
            x = b(x)
        return self.out(x[:, CTX:]) + ctx[:, -1:]            # residual on last context frame


class AR(nn.Module):
    def __init__(self, d=192, L=6, h=6):
        super().__init__()
        self.inp, self.time = nn.Linear(FD, d), nn.Embedding(CTX + H, d)
        self.blocks = nn.ModuleList([TBlock(d, h) for _ in range(L)])
        self.out = nn.Linear(d, FD)

    def step_all(self, X):                                    # X [B, T, 10] -> next-frame preds [B, T, 10]
        T = X.shape[1]
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=X.device), 1)
        x = self.inp(X) + self.time(torch.arange(T, device=X.device))
        for b in self.blocks:
            x = b(x, m)
        return self.out(x) + X                                # predict delta

    @torch.no_grad()
    def rollout(self, ctx):
        X = ctx
        for _ in range(H):
            X = torch.cat([X, self.step_all(X)[:, -1:]], 1)
        return X[:, CTX:]


class Koop(nn.Module):
    def __init__(self, m=256, hid=1024):
        super().__init__()
        self.m = m
        self.enc = nn.Sequential(nn.Linear(CTX * FD, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, m))
        self.dec = nn.Sequential(nn.Linear(m, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, FD))
        k = m // 2
        self.rho = nn.Parameter(torch.full((k,), 4.0))                    # |lambda| = sigmoid(rho) ~ 0.98
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi / 2, k))     # spread of frequencies

    def encode(self, win):                                    # win [B, 5, 10]
        return self.enc(win.reshape(win.shape[0], -1))

    def powers(self, z0, ts):
        """z0 [B, m], ts [T] (steps ahead) -> z [B, T, m], closed form Lambda^t z0 (parallel in t)."""
        r = torch.sigmoid(self.rho)
        k = self.m // 2
        a, b = z0[:, :k], z0[:, k:]
        tt = ts.float()[:, None]
        mag, ang = r[None] ** tt, self.theta[None] * tt                    # [T, k]
        c, s = mag * ang.cos(), mag * ang.sin()
        za = c[None] * a[:, None] - s[None] * b[:, None]
        zb = s[None] * a[:, None] + c[None] * b[:, None]
        return torch.cat([za, zb], -1)

    def rollout(self, ctx, reenc_at=(), trigger_tau=None, max_events=48):
        """Returns (pred [B, 48, 10], passes [B]). reenc_at: fixed re-encoding times (horizon indices 1..47).
        trigger_tau: event-triggered re-encoding on the manifold residual (applied per sample)."""
        B, dev = ctx.shape[0], ctx.device
        hist = ctx                                            # last frames available for re-encoding
        z0 = self.encode(ctx)
        start = torch.zeros(B, dtype=torch.long, device=dev)  # horizon index the current segment starts from
        pred = self.dec(self.powers(z0, torch.arange(1, H + 1, device=dev)))      # [B, 48, 10]
        passes = torch.ones(B, device=dev)
        if trigger_tau is None and not reenc_at:
            return pred, passes
        ts = torch.arange(1, H + 1, device=dev)
        if trigger_tau is None:
            for tau in sorted(reenc_at):                      # periodic: re-encode at fixed horizon indices
                win = torch.cat([ctx, pred], 1)[:, CTX + tau - CTX: CTX + tau]       # last 5 frames before tau
                z = self.encode(win)
                seg = self.dec(self.powers(z, ts[: H - tau]))
                pred = torch.cat([pred[:, :tau], seg], 1)
                passes = passes + 1
            return pred, passes
        # event-triggered: residual for all t in parallel, re-encode at first exceedance, repeat
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        zcur, tcur = z0.clone(), torch.zeros(B, dtype=torch.long, device=dev)
        for _ in range(max_events):
            full = torch.cat([ctx, pred], 1)                                  # [B, 53, 10]
            # windows ending at each horizon index t (frames t-4..t), t = 1..48 -> full[:, t .. t+5)
            wins = full.unfold(1, CTX, 1)[:, 1:H + 1].permute(0, 1, 3, 2)     # [B, 48, 5, 10]
            zw = self.encode(wins.reshape(-1, CTX, FD)).view(B, H, -1)
            # latent of the current segment at each t: Lambda^(t - tcur) zcur  (for t > tcur)
            steps = (ts[None] - tcur[:, None]).clamp(min=0)                   # [B, 48]
            zl = self._powers_per_sample(zcur, steps)                         # [B, 48, m]
            res = (zw - zl).norm(dim=-1) / zl.norm(dim=-1).clamp(min=1e-6)
            valid = (ts[None] > tcur[:, None] + 1) & (ts[None] < H) & ~done[:, None]
            exceed = (res > trigger_tau) & valid
            any_ = exceed.any(1)
            if not any_.any():
                break
            first = exceed.float().argmax(1)                                   # 0-based index of first exceedance
            idx = torch.nonzero(any_).flatten()
            t = first[idx] + 1                                                 # horizon index (1-based), per sample
            rows = full[idx]                                                   # [k, 53, 10]
            gather = t[:, None] + torch.arange(CTX, device=dev)[None]          # frames t .. t+4 = window ending at t
            win = rows[torch.arange(len(idx), device=dev)[:, None], gather]    # [k, 5, 10]
            z = self.encode(win)
            steps = (ts[None] - t[:, None]).clamp(min=0)                       # [k, 48]
            seg = self.dec(self._powers_per_sample(z, steps))                  # [k, 48, 10]
            keep_old = ts[None] <= t[:, None]
            pred[idx] = torch.where(keep_old[..., None], pred[idx], seg)
            zcur[idx], tcur[idx] = z, t
            passes[idx] += 1
            done = done | ~any_
        return pred, passes

    def _powers_per_sample(self, z, steps):
        r = torch.sigmoid(self.rho)
        k = self.m // 2
        a, b = z[:, None, :k], z[:, None, k:]
        tt = steps.float()[..., None]
        mag, ang = r ** tt, self.theta * tt
        c, s = mag * ang.cos(), mag * ang.sin()
        return torch.cat([c * a - s * b, s * a + c * b], -1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())

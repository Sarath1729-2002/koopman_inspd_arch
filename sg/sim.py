"""Frictionless hard spheres in the unit box (equal masses, elastic walls and ball-ball collisions), batched torch.
Mirrors the testbed of 'The Seriality Gap in Video Diffusion Models' (2026) in state space: we predict ball
positions directly instead of pixels. Returns positions [B, F, n, 2] and ball-ball collision counts per frame."""
import torch


def simulate(B, F, n=5, r=0.07, substeps=8, speed=(0.012, 0.03), seed=0, device="cpu", init=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init is None:
        pos = r + (1 - 2 * r) * torch.rand(B, n, 2, generator=g)       # vectorised rejection sampling
        iu0 = torch.triu_indices(n, n, 1)
        for _ in range(200):
            dd = (pos[:, iu0[0]] - pos[:, iu0[1]]).norm(dim=-1)
            bad = (dd < 2.2 * r).any(-1)
            if not bad.any():
                break
            pos[bad] = r + (1 - 2 * r) * torch.rand(int(bad.sum()), n, 2, generator=g)
        ang = 2 * torch.pi * torch.rand(B, n, generator=g)
        sp = speed[0] + (speed[1] - speed[0]) * torch.rand(B, n, generator=g)
        vel = torch.stack([sp * ang.cos(), sp * ang.sin()], -1)
    else:
        pos, vel = init
    pos, vel = pos.to(device).clone(), vel.to(device).clone()
    iu = torch.triu_indices(n, n, 1, device=device)
    out = torch.empty(B, F, n, 2, device=device)
    coll = torch.zeros(B, F, dtype=torch.long, device=device)
    for f in range(F):
        out[:, f] = pos
        for _ in range(substeps):
            pos = pos + vel / substeps
            # walls
            lo, hi = pos < r, pos > 1 - r
            vel = torch.where(lo, vel.abs(), torch.where(hi, -vel.abs(), vel))
            pos = pos.clamp(r, 1 - r)
            # ball-ball (pairwise, equal mass, approaching pairs only)
            d = pos[:, iu[0]] - pos[:, iu[1]]                       # [B, P, 2]
            dist = d.norm(dim=-1).clamp(min=1e-9)
            nrm = d / dist[..., None]
            dv = vel[:, iu[0]] - vel[:, iu[1]]
            app = (dv * nrm).sum(-1)
            hit = (dist < 2 * r) & (app < 0)
            if hit.any():
                imp = torch.where(hit, app, torch.zeros_like(app))[..., None] * nrm
                vel = vel.index_add(1, iu[0], -imp).index_add(1, iu[1], imp)
                push = torch.where(hit, (2 * r - dist) / 2, torch.zeros_like(dist))[..., None] * nrm
                pos = pos.index_add(1, iu[0], push).index_add(1, iu[1], -push)
                coll[:, f] += hit.sum(-1)
    return out, coll


def finite_velocity(prev, cur):
    return cur - prev

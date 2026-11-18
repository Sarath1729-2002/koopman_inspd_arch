"""S5 word problem, scored at every token.
Input: a stream of random elements g_1..g_L of S5 (120 tokens). Label at t: the running product s_t = g_t o ... o g_1.
S5 is non-abelian (and not solvable), so no diagonal / commuting linear recurrence can track it; this is the
standard hardest-case state-tracking test (Merrill et al. 2024, Grazzi et al. 2025, DeltaProduct 2025)."""
import itertools

import torch

_CACHE = {}


def group(name):
    """'s5' (120 elements, default) or 's3' (6 elements, non-abelian fallback). Returns (MUL[s, g], identity, size)."""
    if name not in _CACHE:
        n = int(name[1:])
        perms = list(itertools.permutations(range(n)))
        idx = {p: i for i, p in enumerate(perms)}
        comp = lambda s, g: tuple(g[s[i]] for i in range(n))  # new state = g o s
        mul = torch.tensor([[idx[comp(perms[a], perms[b])] for b in range(len(perms))] for a in range(len(perms))])
        _CACHE[name] = (mul, idx[tuple(range(n))], len(perms))
    return _CACHE[name]


MUL, ID, N = group("s5")


def make_batch(B, L, gen, name="s5"):
    MUL, ID, N = group(name)
    x = torch.randint(0, N, (B, L), generator=gen)
    y = torch.empty_like(x)
    s = torch.full((B,), ID, dtype=torch.long)
    for t in range(L):
        s = MUL[s, x[:, t]]
        y[:, t] = s
    return x, y

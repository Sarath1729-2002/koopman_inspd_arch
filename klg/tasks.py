"""Fan et al. (ICLR 2025) looped-transformer length-generalization tasks, same I/O convention:
input  = query  '>'  '#' * (answer_len + 1)
labels = ignore over query and '>', then the answer tokens, then '#' (end), read off in parallel
after T(n) loops (non-autoregressive). n = problem length; loops T(n) grow with n."""
import random

import torch

VOCAB = ["#", ">", "+", "*"] + [str(d) for d in range(10)]
TOK = {c: i for i, c in enumerate(VOCAB)}
PAD, IGNORE = TOK["#"], -100


def _parity(n, r):
    bits = [r.randint(0, 1) for _ in range(n)]
    return [str(b) for b in bits], [str(sum(bits) % 2)]


def _copy(n, r):
    q = [str(r.randint(0, 9)) for _ in range(n)]
    return q, list(q)


def _addition(n, r):
    a = [r.randint(0, 1) for _ in range(n)]
    b = [r.randint(0, 1) for _ in range(n)]
    s = int("".join(map(str, a)), 2) + int("".join(map(str, b)), 2)
    ans = format(s, "b").zfill(n + 1)
    return [str(x) for x in a] + ["+"] + [str(x) for x in b], list(ans)


TASKS = {
    # name: (generator, T(n), default train max n, eval lengths, block depth, heads)  (depth/heads from Fan et al. Tab. 2)
    "parity": (_parity, lambda n: n, 20, list(range(5, 51, 5)), 1, 8),
    "copy": (_copy, lambda n: n, 19, list(range(5, 41, 5)), 2, 8),
    "addition": (_addition, lambda n: n + 1, 19, list(range(5, 36, 5)), 3, 8),
}


def make_batch(task, lengths, seed=None, rng=None):
    gen, T, *_ = TASKS[task]
    r = rng or random.Random(seed)
    xs, ys, ts = [], [], []
    for n in lengths:
        q, ans = gen(n, r)
        x = [TOK[c] for c in q] + [TOK[">"]] + [PAD] * (len(ans) + 1)
        y = [IGNORE] * (len(q) + 1) + [TOK[c] for c in ans] + [PAD]
        xs.append(x); ys.append(y); ts.append(T(n))
    L = max(len(x) for x in xs)
    X = torch.full((len(xs), L), PAD, dtype=torch.long)
    Y = torch.full((len(xs), L), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, : len(x)] = torch.tensor(x)
        Y[i, : len(y)] = torch.tensor(y)
    return X, Y, torch.tensor(ts)

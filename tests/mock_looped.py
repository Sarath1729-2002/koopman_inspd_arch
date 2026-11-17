"""Tiny CPU stand-in with Ouro's interface: model.model.layers looped config.total_ut_steps times,
model.model.norm, model.lm_head, attention_mask/use_cache kwargs. For smoke tests only."""
from types import SimpleNamespace

import torch
import torch.nn as nn


class MockTokenizer:
    pad_token, eos_token, padding_side = None, "\x00", "right"
    pad_token_id = 0

    def _enc(self, s):
        return [min(ord(c), 255) for c in s][:256]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None):
        single = isinstance(texts, str)
        seqs = [self._enc(t) for t in ([texts] if single else texts)]
        if return_tensors is None:
            return SimpleNamespace(input_ids=seqs[0] if single else seqs)
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            if self.padding_side == "left":
                ids[i, L - len(s):], att[i, L - len(s):] = torch.tensor(s), 1
            else:
                ids[i, :len(s)], att[i, :len(s)] = torch.tensor(s), 1
        return _Enc(input_ids=ids, attention_mask=att)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i > 0)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(r.tolist()) for r in rows]


class _Enc(dict):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.__dict__.update(kw)

    def to(self, device):
        return _Enc(**{k: v.to(device) for k, v in self.items()})


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.mix = nn.Linear(d, d)

    def forward(self, hidden_states, attention_mask=None, **kw):
        h = hidden_states
        ctx = torch.cumsum(h, 1) / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
        return (h + 0.5 * self.f(self.n(h)) + 0.1 * self.mix(ctx),)


class Inner(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(256, d)
        self.layers = nn.ModuleList([Layer(d) for _ in range(2)])
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids, attention_mask=None, **kw):
        h = self.embed(input_ids)
        for _ in range(self.config.total_ut_steps):
            for l in self.layers:
                h = l(h, attention_mask=attention_mask)[0]
        return self.norm(h)


class MockLooped(nn.Module):
    def __init__(self, loops, d=32):
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(total_ut_steps=loops, early_exit_threshold=1.0)
        self.model = Inner(d, self.config)
        self.lm_head = nn.Linear(d, 256, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=None, **kw):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids, attention_mask)))

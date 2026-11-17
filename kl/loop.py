"""Hooks into a looped LM (Ouro: the whole decoder stack is applied `total_ut_steps` times).

One forward with T_max loops gives every depth at once: a forward hook on the LAST decoder layer
fires once per loop; its output is the state after that loop (depth d = 1..T_max). We
  * record it (for the depth-d readout  logits_d = lm_head(final_norm(state_d)) ),
  * optionally replace it with a corrected state (course correction) for depths > T_train,
    which is then both what is read out at depth d and what feeds loop d+1.
Because loop d only depends on loops <= d, depth-d readouts from a T_max forward equal the
outputs of a model run with exactly d loops. `kl.inspect` verifies this on the real model.
"""
import torch
import torch.nn as nn


def find_parts(model):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        for name, m in model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 1 and "layer" in name.lower():
                layers, inner = m, model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                break
    norm = getattr(inner, "norm", None) or getattr(inner, "final_norm", None)
    head = getattr(model, "lm_head", None)
    if layers is None or norm is None or head is None:
        raise RuntimeError(f"could not locate layers/norm/lm_head; top modules: {[n for n, _ in model.named_children()]}")
    return inner, layers, norm, head


class LoopController:
    def __init__(self, model, t_train):
        self.model = model
        self.inner, self.layers, self.norm, self.head = find_parts(model)
        self.t_train = t_train
        self.correct = None      # callable(depth, state, ctl) -> state, used for depth > t_train
        self.record = True
        self.keep_states = True
        self.on_state = None     # optional callable(depth, state) for collecting training data
        self.states = []
        self.depth = 0
        self.inner.register_forward_pre_hook(self._reset)
        self.layers[-1].register_forward_hook(self._post)

    def _reset(self, *_args, **_kw):
        self.states, self.depth = [], 0

    @staticmethod
    def _split(out):
        if isinstance(out, tuple):
            return out[0], lambda h: (h,) + tuple(out[1:])
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state, None
        return out, lambda h: h

    def _post(self, module, args, out):
        h, rebuild = self._split(out)
        self.depth += 1
        d = self.depth
        if self.correct is not None and d > self.t_train:
            h = self.correct(d, h, self)
        if self.on_state is not None:
            self.on_state(d, h)
        if self.keep_states:
            self.states.append(h)
        if rebuild is not None and (self.correct is not None and d > self.t_train):
            return rebuild(h)
        return None

    @torch.no_grad()
    def readout(self, d):
        """logits at depth d (1-based) from the recorded state of the last forward."""
        return self.head(self.norm(self.states[d - 1]))


def set_loops(model, T):
    """Set the loop count on the model config (and any module attribute mirroring it)."""
    for obj in [model.config] + [m for m in model.modules()]:
        for attr in ("total_ut_steps",):
            if hasattr(obj, attr) and not callable(getattr(obj, attr)):
                try:
                    setattr(obj, attr, T)
                except Exception:
                    pass
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "total_ut_steps"):
            cfg.total_ut_steps = T
    if hasattr(model.config, "early_exit_threshold"):
        model.config.early_exit_threshold = 1.0


@torch.no_grad()
def count_loops(model, ctl, tok, device):
    ids = tok("1 + 1 =", return_tensors="pt").input_ids.to(device)
    model(input_ids=ids, use_cache=False)
    return ctl.depth

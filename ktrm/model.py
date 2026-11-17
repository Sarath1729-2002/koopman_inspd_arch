"""Load a trained TRM (SamsungSAILMontreal/TinyRecursiveModels) and run it step by step,
exposing every recursion state. Inference here mirrors the original eval loop exactly:
carry starts at H_init/L_init, each ACT step runs H_cycles x (L_cycles z_L updates + 1 z_H
update), the carry passes unchanged to the next ACT step, and eval always runs all steps.
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

IGNORE = -100


def add_trm_path(trm_dir):
    p = os.path.abspath(trm_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_meta(data_dir, split):
    with open(os.path.join(data_dir, split, "dataset.json")) as f:
        return json.load(f)


def arch_config(trm_dir, ckpt_path, overrides):
    """Arch config: the checkpoint run's all_config.yaml if present, else TRM's trm.yaml."""
    cand = os.path.join(os.path.dirname(ckpt_path), "all_config.yaml") if ckpt_path else None
    if cand and os.path.exists(cand):
        arch = OmegaConf.load(cand).arch
        src = cand
    else:
        arch = OmegaConf.load(os.path.join(trm_dir, "config", "arch", "trm.yaml"))
        src = "TRM config/arch/trm.yaml (+ overrides)"
    arch = OmegaConf.merge(arch, OmegaConf.create({k: v for k, v in overrides.items() if v is not None}))
    return OmegaConf.to_container(arch, resolve=True), src


def build_model(trm_dir, arch, meta, batch_size, device, dtype=None):
    add_trm_path(trm_dir)
    from utils.functions import load_model_class  # TRM repo

    cls = load_model_class(arch["name"])
    cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    cfg.update(batch_size=batch_size, vocab_size=meta["vocab_size"], seq_len=meta["seq_len"],
               num_puzzle_identifiers=meta["num_puzzle_identifiers"], causal=False)
    if dtype:
        cfg["forward_dtype"] = dtype
    with torch.device(device):
        model = cls(cfg)
    model.eval()
    return model


def load_checkpoint(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    clean = {}
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in ("_orig_mod.", "module.", "model."):
                if kk.startswith(p):
                    kk, changed = kk[len(p):], True
        clean[kk] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    critical = [m for m in missing if any(s in m for s in ("L_level", "embed_tokens", "lm_head", "q_head", "puzzle_emb.weights"))]
    if critical:
        raise RuntimeError(f"checkpoint is missing core weights: {critical[:5]} (first keys in file: {list(sd)[:5]})")
    if unexpected:
        print(f"[ckpt] ignored {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    pe = model.inner.puzzle_emb.weights.shape if hasattr(model.inner, "puzzle_emb") else None
    print(f"[ckpt] loaded {path}  puzzle_emb {tuple(pe) if pe is not None else None}")


@torch.no_grad()
def rollout(model, batch, n_act, on_state=None, noise=None):
    """Runs n_act ACT steps. on_state(t, zH, zL) after every H-cycle (t = global H-cycle index).
    noise(t, zH, zL) -> (zH, zL) is applied after every H-cycle, before on_state.
    Returns input embeddings x and per-ACT-step (preds [B,seq], q_halt [B])."""
    inner = model.inner
    cfg = inner.config
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    B, L, D = x.shape
    zH = inner.H_init.to(x.dtype).view(1, 1, D).expand(B, L, D).clone()
    zL = inner.L_init.to(x.dtype).view(1, 1, D).expand(B, L, D).clone()
    out, t = [], 0
    for _a in range(n_act):
        for _h in range(cfg.H_cycles):
            for _l in range(cfg.L_cycles):
                zL = inner.L_level(zL, zH + x, **seq_info)
            zH = inner.L_level(zH, zL, **seq_info)
            if noise is not None:
                zH, zL = noise(t, zH, zL)
            if on_state is not None:
                on_state(t, zH, zL)
            t += 1
        preds = inner.lm_head(zH)[:, inner.puzzle_emb_len:].argmax(-1)
        q = inner.q_head(zH[:, 0]).float()[..., 0]
        out.append((preds, q))
    return x, out


def exact_correct(preds, labels):
    mask = labels != IGNORE
    ok = (preds == labels) | ~mask
    return ok.all(-1) & mask.any(-1), ((preds == labels) & mask).sum(-1).float() / mask.sum(-1).clamp_min(1)


class ArcTest:
    """Direct reader of a TRM-built ARC split (bypasses PuzzleDataset so we can sample)."""

    def __init__(self, data_dir, split="test"):
        d = os.path.join(data_dir, split)
        self.meta = load_meta(data_dir, split)
        self.inputs = np.load(os.path.join(d, "all__inputs.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(d, "all__labels.npy"), mmap_mode="r")
        pid = np.load(os.path.join(d, "all__puzzle_identifiers.npy"))
        pidx = np.load(os.path.join(d, "all__puzzle_indices.npy"))
        self.ex_puzzle = np.searchsorted(pidx, np.arange(len(self.inputs)), side="right") - 1
        self.ex_identifier = pid[self.ex_puzzle]
        names_path = os.path.join(data_dir, "identifiers.json")
        names = json.load(open(names_path)) if os.path.exists(names_path) else None
        self.names = [names[i] for i in self.ex_identifier] if names else [str(i) for i in self.ex_identifier]
        self.base = [n.split("|||")[0] for n in self.names]
        self.canonical = np.array(["|||" not in n for n in self.names])

    def sample(self, n, seed, canonical_only=False):
        rng = np.random.default_rng(seed)
        pool = np.where(self.canonical)[0] if canonical_only else np.arange(len(self.inputs))
        return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))

    def batch(self, idx, device):
        lab = self.labels[idx].astype(np.int64)
        ig = self.meta.get("ignore_label_id")
        if ig is not None:
            lab[lab == ig] = IGNORE
        return {
            "inputs": torch.from_numpy(self.inputs[idx].astype(np.int64)).to(device),
            "labels": torch.from_numpy(lab).to(device),
            "puzzle_identifiers": torch.from_numpy(self.ex_identifier[idx].astype(np.int64)).to(device),
        }

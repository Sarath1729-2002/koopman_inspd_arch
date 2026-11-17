"""Smoke-test helper: a random-init TRM saved exactly like pretrain.py saves it
(torch.compile + ACTLossHead prefixes), so the loader path is exercised end to end."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ktrm.model import arch_config, build_model, load_meta  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trm_dir", required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
arch, _ = arch_config(a.trm_dir, None, {"H_cycles": 2, "L_cycles": 2, "L_layers": 2, "hidden_size": 64, "num_heads": 2,
                                        "halt_max_steps": 4})
m = build_model(a.trm_dir, arch, load_meta(a.data_dir, "train"), 8, "cpu", "float32")
with torch.no_grad():  # puzzle embeddings are zero-init; randomize so they matter
    m.inner.puzzle_emb.weights.normal_(0, 0.1)
    m.inner.q_head.weight.normal_(0, 0.05)
sd = {"_orig_mod.model." + k: v for k, v in m.state_dict().items()}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(sd, a.out)
print("saved", a.out, len(sd), "tensors")

import argparse
import json
import os

import torch

from .model import ArcTest, arch_config, build_model, load_checkpoint, load_meta


def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--trm_dir", default="third_party/TinyRecursiveModels")
    ap.add_argument("--data_dir", default="data/arc1concept-aug-1000")
    ap.add_argument("--ckpt", default=None, help="TRM checkpoint file (step_N). Omit only with --random_init")
    ap.add_argument("--random_init", action="store_true", help="smoke tests only")
    ap.add_argument("--out", default="runs/arc1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, help="override forward dtype (bfloat16 default; float32 for CPU tests)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    # arch overrides used by the README's ARC-AGI-1 run (ignored if all_config.yaml sits next to the ckpt)
    ap.add_argument("--H_cycles", type=int, default=3)
    ap.add_argument("--L_cycles", type=int, default=4)
    ap.add_argument("--L_layers", type=int, default=2)
    ap.add_argument("--hidden_size", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    return ap


def setup(a):
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    meta_train = load_meta(a.data_dir, "train")  # num_puzzle_identifiers as at training time
    arch, src = arch_config(a.trm_dir, a.ckpt, {"H_cycles": a.H_cycles, "L_cycles": a.L_cycles, "L_layers": a.L_layers,
                                                "hidden_size": a.hidden_size, "num_heads": a.num_heads})
    model = build_model(a.trm_dir, arch, meta_train, a.batch_size, a.device, a.dtype)
    if a.ckpt:
        load_checkpoint(model, a.ckpt)
    elif not a.random_init:
        raise SystemExit("--ckpt is required (use --random_init only for smoke tests)")
    data = ArcTest(a.data_dir, "test")
    info = {"arch_source": src, "H_cycles": model.config.H_cycles, "L_cycles": model.config.L_cycles,
            "hidden_size": model.config.hidden_size, "puzzle_emb_len": int(model.inner.puzzle_emb_len),
            "halt_max_steps": model.config.halt_max_steps, "ckpt": a.ckpt}
    print("[setup]", json.dumps(info))
    return model, data, info

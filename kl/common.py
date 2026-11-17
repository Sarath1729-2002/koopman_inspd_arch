import argparse
import json
import os

import torch

from .loop import LoopController, count_loops, set_loops


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
    ap.add_argument("--mock", action="store_true", help="tiny CPU stand-in (tests only)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="runs/ouro14")
    ap.add_argument("--t_train", type=int, default=4, help="loops the model was trained with")
    ap.add_argument("--t_max", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def load(a, loops=None):
    loops = loops or a.t_max
    if a.mock:
        from tests.mock_looped import MockTokenizer, MockLooped
        tok, model = MockTokenizer(), MockLooped(loops)
    else:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
        cfg.total_ut_steps = loops
        if hasattr(cfg, "early_exit_threshold"):
            cfg.early_exit_threshold = 1.0
        model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.to(a.device).eval()
    set_loops(model, loops)
    ctl = LoopController(model, a.t_train)
    n = count_loops(model, ctl, tok, a.device)
    if n != loops:
        raise RuntimeError(f"expected {loops} loops per forward, last-layer hook fired {n} times. "
                           "The model may not loop the whole stack; run kl.check and send me the output.")
    return tok, model, ctl


def save_json(obj, a, name):
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, name), "w") as f:
        json.dump(obj, f, indent=1)

"""Stage 0: verify the hooks on the real model before spending GPU time.
  1. structure: which modules are layers / final norm / lm_head
  2. the last-layer hook fires T times per forward for several T
  3. depth-d readout from a T_max forward == logits of a model run with exactly d loops
  4. a first look at drift: state norm and cosine to the previous depth, per depth
"""
import torch

from .common import load, parser, save_json
from .loop import find_parts


@torch.no_grad()
def main():
    a = parser(__doc__).parse_args()
    tok, model, ctl = load(a)
    inner, layers, norm, head = find_parts(model)
    print(f"[inspect] layers={len(layers)} ({type(layers[0]).__name__})  norm={type(norm).__name__}  head={type(head).__name__}")
    print(f"[inspect] config.total_ut_steps={getattr(model.config, 'total_ut_steps', None)}  hook fired {ctl.depth} times")
    text = "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?\nAnswer:"
    ids = tok(text, return_tensors="pt").input_ids.to(a.device)
    out = model(input_ids=ids, use_cache=False)
    final = out.logits.float()
    readouts = [ctl.readout(d).float() for d in range(1, a.t_max + 1)]
    res = {"n_layers": len(layers), "loops_fired": ctl.depth,
           "final_vs_readout_tmax_maxabs": float((final - readouts[-1]).abs().max())}
    print(f"[check] model logits vs depth-{a.t_max} readout: max|diff| = {res['final_vs_readout_tmax_maxabs']:.4g}")
    # separate model with exactly t_train loops
    tok2, model2, ctl2 = load(a, loops=a.t_train)
    o2 = model2(input_ids=ids, use_cache=False).logits.float()
    res["exact_t_train_vs_readout_maxabs"] = float((o2 - readouts[a.t_train - 1]).abs().max())
    res["exact_t_train_top1_agree"] = float((o2.argmax(-1) == readouts[a.t_train - 1].argmax(-1)).float().mean())
    print(f"[check] {a.t_train}-loop model vs depth-{a.t_train} readout: max|diff| = {res['exact_t_train_vs_readout_maxabs']:.4g}, "
          f"top-1 agreement {res['exact_t_train_top1_agree']:.3f}")
    del model2
    st = [s.float() for s in ctl.states] if ctl.states else []
    model(input_ids=ids, use_cache=False)
    st = [s.float() for s in ctl.states]
    drift = []
    for d in range(1, len(st) + 1):
        nrm = float(st[d - 1].norm(dim=-1).mean())
        cos = float(torch.nn.functional.cosine_similarity(st[d - 1], st[d - 2], dim=-1).mean()) if d > 1 else float("nan")
        rel = float(((st[d - 1] - st[d - 2]).norm(dim=-1) / st[d - 2].norm(dim=-1)).mean()) if d > 1 else float("nan")
        drift.append({"depth": d, "mean_norm": nrm, "cos_prev": cos, "rel_change": rel})
        print(f"  depth {d:2d}  |s|={nrm:9.2f}  cos(prev)={cos:.4f}  rel change={rel:.4f}")
    res["drift"] = drift
    ok = res["exact_t_train_top1_agree"] > 0.99
    print(f"\n[inspect] {'OK: readouts match a real d-loop model' if ok else 'MISMATCH: send me this output before running the rest'}")
    save_json(res, a, "inspect.json")


if __name__ == "__main__":
    main()

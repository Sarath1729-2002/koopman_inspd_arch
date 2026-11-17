"""Stage 3: does it hold up in actual generation? GSM8K, 4-shot, greedy, exact-match on the final number.
Runs depths {T_train, 2*T_train, T_max} x {none, best Koopman policy, best control policy}
(best = lowest GSM NLL at depth 2*T_train in eval.json). The model is reloaded with exactly d loops.
"""
import json
import os

import numpy as np
import torch

from .common import load, parser, save_json
from .corrections import CONTROLS, KOOPMAN, Policies
from .data import gsm8k, gsm_answer, gsm_fewshot


@torch.no_grad()
def greedy(model, tok, prompts, max_new, device):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    if hasattr(model, "generate"):
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        tok.padding_side = "right"
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    ids = enc["input_ids"]  # mock: no cache, recompute
    outs = [[] for _ in prompts]
    for _ in range(max_new):
        logits = model(input_ids=ids).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        for b in range(len(prompts)):
            outs[b].append(int(nxt[b]))
    return [tok.decode(o) for o in outs]


def main():
    ap = parser(__doc__)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--depths", default=None)
    ap.add_argument("--policies", default=None, help="default: none + best Koopman + best control from eval.json")
    a = ap.parse_args()
    ev = json.load(open(os.path.join(a.out, "eval.json")))
    d2 = min(2 * a.t_train, a.t_max)
    if a.policies:
        pols = a.policies.split(",")
    else:
        P = ev["policies"]
        bk = min((k for k in P if k.split("@")[0] in KOOPMAN), key=lambda k: P[k]["gsm_nll_by_depth"][d2 - 1])
        bc = min((k for k in P if k.split("@")[0] in CONTROLS), key=lambda k: P[k]["gsm_nll_by_depth"][d2 - 1])
        pols = ["none", bk, bc]
    depths = [int(x) for x in a.depths.split(",")] if a.depths else sorted({a.t_train, d2, a.t_max})
    rng = np.random.default_rng(a.seed)
    test = gsm8k(a.data_dir, "test")
    items = [test[i] for i in rng.choice(len(test), size=min(a.n, len(test)), replace=False)]
    shots = gsm_fewshot(gsm8k(a.data_dir, "train"), a.shots, a.seed)
    prompts = [shots + f"Question: {x['question']}\nAnswer:" for x in items]
    gold = [gsm_answer(x["answer"]) for x in items]
    fit = torch.load(os.path.join(a.out, "fit.pt"), map_location="cpu")
    res, per = {}, {}
    for d in depths:
        tok, model, ctl = load(a, loops=d)
        pol = Policies(fit, a.device)
        for name in (pols if d > a.t_train else ["none"]):
            ctl.correct = pol.make(name)
            ctl.keep_states = True
            texts = []
            for i in range(0, len(prompts), a.batch_size):
                texts += greedy(model, tok, prompts[i: i + a.batch_size], a.max_new, a.device)
            preds = [gsm_answer(t.split("\n\nQuestion:")[0]) for t in texts]
            ok = np.array([p is not None and g is not None and float(p) == float(g) for p, g in zip(preds, gold)])
            res[f"{name}@d{d}"] = float(ok.mean())
            per[f"{name}@d{d}"] = ok
            print(f"[gen] depth {d:2d}  {name:16s} acc {ok.mean():.4f}", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    np.savez(os.path.join(a.out, "gen_per_example.npz"), **per)
    save_json({"n": len(items), "shots": a.shots, "acc": res, "policies": pols, "depths": depths}, a, "generate.json")
    L = ["", "## GSM8K generation (greedy, exact match)", "", "| depth | policy | accuracy |", "|---|---|---|"]
    L += [f"| {k.split('@d')[1]} | {k.split('@d')[0]} | {v:.4f} |" for k, v in res.items()]
    with open(os.path.join(a.out, "REPORT.md"), "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

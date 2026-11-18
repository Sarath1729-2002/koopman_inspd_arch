"""Course-correcting Koopman across depth: replace m decoder layers of a pretrained LLM with linear maps fitted
by least squares on hidden states (no training), and compare WHERE the real layers sit at equal budget.

Arms (all remove the same m layers' worth of nonlinear compute):
  drop_contig  contiguous block [a, a+m) skipped (identity)                     (ShortGPT-style)
  lin_contig   same block replaced by ONE linear map h_a -> h_{a+m}, fitted jointly   (ReplaceMe-style)
  drop_inter   m non-adjacent layers skipped
  lin_inter    m non-adjacent layers, each replaced by its own linear map h_l -> h_{l+1}; real layers in between
               act as the periodic nonlinear re-encoding (Fathi et al. 2024)
Selection is by calibration fit error for both linear arms (best contiguous block / best non-adjacent set via DP).
Layer 0 and the last layer are never replaced. Writes <out>/m<m>.json (and full.json)."""
import argparse, json, math, os, random, re, time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- model / data
def load_model(name, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev).eval()
    return model, tok


def layers_of(model):
    return model.model.layers


def text_windows(path, n, nbytes, seed):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.integers(0, len(data) - nbytes - 1, n):
        out.append(bytes(data[i:i + nbytes]).decode("utf-8", errors="ignore"))
    return out


def token_batches(tok, texts, T, B):
    ids = [tok(t, add_special_tokens=False)["input_ids"][:T] for t in texts]
    ids = [x for x in ids if len(x) == T]
    for i in range(0, len(ids), B):
        yield torch.tensor(ids[i:i + B])


# ---------------------------------------------------------------- hooks
def _hidden_in(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _replace_out(output, new):
    if isinstance(output, torch.Tensor):
        return new
    return (new,) + tuple(output[1:])


class Patcher:
    """plan: {layer_idx: ('id',) | ('lin', W, b)}; the patched layer's output becomes f(its input)."""

    def __init__(self, model, plan):
        self.handles = []
        for l, spec in plan.items():
            def hook(mod, args, kwargs, output, spec=spec):
                h = _hidden_in(args, kwargs)
                if spec[0] == "id":
                    return _replace_out(output, h)
                W, b = spec[1], spec[2]
                return _replace_out(output, (h.float() @ W + b).to(h.dtype))
            self.handles.append(layers_of(model)[l].register_forward_hook(hook, with_kwargs=True))

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- least squares
@torch.no_grad()
def fit(model, batches, spans, dev, ridge=1e-4):
    """spans: list of (a, b) -> W, b mapping input of layer a to input of layer b (= output of layer b-1).
    Returns {(a,b): (W [d,d] fp32, bias [d] fp32, rel_err)}. Position 0 of each sequence (attention sink) is excluded."""
    L = len(layers_of(model))
    need_x = sorted({a for a, _ in spans}); need_y = sorted({b for _, b in spans})
    store = {}
    hs = []
    for l in range(L):
        def pre(mod, args, kwargs, l=l):
            store[l] = _hidden_in(args, kwargs)
        hs.append(layers_of(model)[l].register_forward_pre_hook(pre, with_kwargs=True))
    def post(mod, args, kwargs, output):
        store[L] = output if isinstance(output, torch.Tensor) else output[0]
    hs.append(layers_of(model)[L - 1].register_forward_hook(post, with_kwargs=True))
    d = model.config.hidden_size
    XtX = {a: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for a in need_x}
    XtY = {s: torch.zeros(d + 1, d, dtype=torch.float64, device=dev) for s in spans}
    YtY = {b: torch.zeros((), dtype=torch.float64, device=dev) for b in need_y}
    Ysum = {b: torch.zeros(d, dtype=torch.float64, device=dev) for b in need_y}
    n = 0
    for x in batches:
        store.clear()
        model(x.to(dev), use_cache=False)
        aug = lambda t: torch.cat([t[:, 1:].reshape(-1, d).double(), torch.ones(t.shape[0] * (t.shape[1] - 1), 1, dtype=torch.float64, device=dev)], 1)
        X = {a: aug(store[a]) for a in need_x}
        Y = {b: store[b][:, 1:].reshape(-1, d).double() for b in need_y}
        for a in need_x:
            XtX[a] += X[a].T @ X[a]
        for s in spans:
            XtY[s] += X[s[0]].T @ Y[s[1]]
        for b in need_y:
            YtY[b] += (Y[b] ** 2).sum(); Ysum[b] += Y[b].sum(0)
        n += next(iter(Y.values())).shape[0]
    for h in hs:
        h.remove()
    out = {}
    for (a, b) in spans:
        A = XtX[a]
        lam = ridge * A.diagonal()[:-1].mean()
        Wf = torch.linalg.solve(A + lam * torch.eye(d + 1, dtype=A.dtype, device=dev), XtY[(a, b)])
        sse = YtY[b] - 2 * (Wf * XtY[(a, b)]).sum() + (Wf * (A @ Wf)).sum()
        var = YtY[b] - (Ysum[b] ** 2).sum() / n
        out[(a, b)] = (Wf[:-1].float(), Wf[-1].float(), float(sse / var))
    return out


def best_nonadjacent(err, cand, m):
    """choose m indices from sorted cand with no two consecutive integers, minimising sum err. DP."""
    INF = float("inf")
    k = len(cand)
    dp = [[INF] * (m + 1) for _ in range(k + 1)]   # dp[i][j]: first i cands, j chosen, last not constraining
    take = [[None] * (m + 1) for _ in range(k + 1)]
    for i in range(k + 1):
        dp[i][0] = 0.0
    for i in range(1, k + 1):
        for j in range(1, m + 1):
            best, tk = dp[i - 1][j], False
            prev = i - 1
            if i >= 2 and cand[i - 1] - cand[i - 2] == 1:
                prev = i - 2
            v = dp[prev][j - 1] + err[cand[i - 1]] if (prev >= 0) else INF
            if v < best:
                best, tk = v, True
            dp[i][j], take[i][j] = best, tk
    if dp[k][m] == INF:
        return None
    sel, i, j = [], k, m
    while j > 0:
        if take[i][j]:
            sel.append(cand[i - 1]); i = i - 2 if (i >= 2 and cand[i - 1] - cand[i - 2] == 1) else i - 1; j -= 1
        else:
            i -= 1
    return sorted(sel)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def perplexity(model, batches, dev):
    tot, cnt = 0.0, 0
    for x in batches:
        x = x.to(dev)
        lg = model(x, use_cache=False).logits.float()
        l = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
        tot += l.item(); cnt += x[:, 1:].numel()
    return math.exp(tot / cnt)


def load_gsm8k(path, n):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        q = r.get("question") or r.get("problem") or r.get("prompt")
        a = r.get("answer") or r.get("gold") or r.get("target")
        if q is None or a is None:
            continue
        a = str(a)
        g = a.split("####")[-1] if "####" in a else a
        g = g.strip().replace(",", "")
        out.append((q, g))
    return out[:n]


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


@torch.no_grad()
def gsm8k_acc(model, tok, items, dev, bs=50, max_new=320):
    if not items:
        return None
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    correct = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        prompts = []
        for q, _ in chunk:
            msg = [{"role": "user", "content": q + "\nSolve step by step, then end with 'The answer is <number>'."}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                           if tok.chat_template else "Question: " + q + "\nAnswer: Let's think step by step.")
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for (q, g), o in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            txt = tok.decode(o, skip_special_tokens=True)
            m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", txt)
            p = m.group(1).replace(",", "") if m else last_number(txt)
            try:
                correct += p is not None and abs(float(p) - float(g)) < 1e-6
            except ValueError:
                pass
    return correct / len(items)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--budgets", default="6,9,12")
    ap.add_argument("--data", default="data")
    ap.add_argument("--gsm8k", default="data/gsm8k_test.jsonl")
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--n_ppl", type=int, default=128)
    ap.add_argument("--n_gsm", type=int, default=200)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="also evaluate the unmodified model")
    ap.add_argument("--out", default="results_depth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model, tok = load_model(a.model, dev)
    L = len(layers_of(model))
    calib = list(token_batches(tok, text_windows(os.path.join(a.data, "train.bin"), a.n_calib, a.T * 6, 0), a.T, a.bs))
    pplb = list(token_batches(tok, text_windows(os.path.join(a.data, "eval.bin"), a.n_ppl, a.T * 6, 1), a.T, a.bs))
    gsm = load_gsm8k(a.gsm8k, a.n_gsm) if os.path.exists(a.gsm8k) else []
    print(f"model {a.model}: {L} layers; calib {sum(len(x) for x in calib)} seqs, ppl {sum(len(x) for x in pplb)} seqs, "
          f"gsm8k {len(gsm)} items ({time.time() - t0:.0f}s)", flush=True)

    def evaluate(plan, tag, extra):
        p = Patcher(model, plan)
        try:
            ppl = perplexity(model, pplb, dev)
            acc = gsm8k_acc(model, tok, gsm, dev)
        finally:
            p.remove()
        r = {"arm": tag, "ppl": ppl, "gsm8k": acc, **extra}
        print(f"  {tag:12s} ppl {ppl:8.3f}  gsm8k {acc if acc is None else round(acc, 3)}  {extra} ({time.time() - t0:.0f}s)", flush=True)
        return r

    if a.full:
        json.dump({"m": 0, "arms": [evaluate({}, "full", {})]}, open(os.path.join(a.out, "full.json"), "w"), indent=1)

    budgets = [int(x) for x in a.budgets.split(",")]
    cand = list(range(1, L - 1))
    spans = {(l, l + 1) for l in cand}
    for m in budgets:
        spans |= {(s, s + m) for s in range(1, L - m)}   # block [s, s+m) must end before the last layer
    spans = sorted(spans)
    fits = fit(model, calib, spans, dev)
    err1 = {l: fits[(l, l + 1)][2] for l in cand}
    print("per-layer linear fit rel. error: " + " ".join(f"{l}:{e:.3f}" for l, e in err1.items()), flush=True)
    for m in budgets:
        blocks = [(s, fits[(s, s + m)][2]) for s in range(1, L - m)]
        s, berr = min(blocks, key=lambda t: t[1])
        sel = best_nonadjacent(err1, cand, m)
        res = {"m": m, "L": L, "block": [s, s + m], "block_fit_err": berr, "inter_layers": sel,
               "inter_fit_err_sum": sum(err1[l] for l in sel) if sel else None, "per_layer_err": err1, "arms": []}
        W, b, _ = fits[(s, s + m)]
        res["arms"].append(evaluate({l: ("id",) for l in range(s, s + m)}, "drop_contig", {"layers": [s, s + m]}))
        res["arms"].append(evaluate({s: ("lin", W, b), **{l: ("id",) for l in range(s + 1, s + m)}}, "lin_contig", {"layers": [s, s + m]}))
        if sel:
            res["arms"].append(evaluate({l: ("id",) for l in sel}, "drop_inter", {"layers": sel}))
            res["arms"].append(evaluate({l: ("lin", fits[(l, l + 1)][0], fits[(l, l + 1)][1]) for l in sel}, "lin_inter", {"layers": sel}))
        json.dump(res, open(os.path.join(a.out, f"m{m}.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

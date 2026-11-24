"""Drift diagnostic for a LOOPED LM (Ouro-1.4B: the whole 24-layer stack is applied T times; trained with T = 4).
Question (from the video results): when looped past its trained depth, do Ouro's states leave the manifold of
trained-depth states, and does that drift track the damage? If yes, projection-style course correction has
something to fix; if the drift is no larger than the floor, it cannot help (as in the JEPA world models).

Manifold = PCA(k) of states at depths 1..4 on a fit split (per-token residual-stream states after each loop).
  floor   = relative PCA residual ||h - P h||^2 / ||h||^2 of HELD-OUT states at depths 1..4
  drift_d = the same for states at depth d = 5..T_max
  damage  = per-token increase in gold-solution NLL vs depth 4 (readout lm_head(norm(h_d)))
Reports drift_d / floor, corr(drift_d, damage_d) over tokens, and NLL per depth.
Criteria (fixed before the run):  O1 drift at depth 8 >= 3 x floor AND corr(drift_8, damage_8) >= 0.3
Text: GSM8k-Aug test solutions (data_lcot/test.jsonl).  Writes results_rdiag/ouro.json"""
import argparse, json, os, numpy as np, torch, torch.nn as nn


def find_parts(model):
    inner = getattr(model, 'model', None)
    layers = getattr(inner, 'layers', None) if inner is not None else None
    if layers is None:
        for name, m in model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 1 and 'layer' in name.lower():
                layers, inner = m, (model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model); break
    norm = getattr(inner, 'norm', None) or getattr(inner, 'final_norm', None)
    return inner, layers, norm, model.lm_head


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ByteDance/Ouro-1.4B'); ap.add_argument('--t_train', type=int, default=4)
    ap.add_argument('--t_max', type=int, default=12); ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--k', type=int, default=256); ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--data', default='data_lcot/test.jsonl'); ap.add_argument('--out', default='results_rdiag')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); cfg.total_ut_steps = a.t_max
    if hasattr(cfg, 'early_exit_threshold'):
        cfg.early_exit_threshold = 1.0
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    inner, layers, norm, head = find_parts(model)
    states = []
    inner.register_forward_pre_hook(lambda *x, **k: states.clear())
    layers[-1].register_forward_hook(lambda m, i, o: states.append((o[0] if isinstance(o, tuple) else o).detach()))
    rows = [json.loads(l) for l in open(a.data)][: (16 if a.smoke else a.n)]
    texts = [(r['question'] + '\n', '\n'.join(r['steps']) + '\n### ' + r['answer']) for r in rows]
    half = len(texts) // 2
    fit_states, recs = [], []   # recs: per example, per depth: (token states on solution, nll per token)
    with torch.no_grad():
        for i in range(0, len(texts), a.bs):
            B = texts[i:i + a.bs]
            q = tok([x for x, _ in B], add_special_tokens=False)['input_ids']
            s = tok([y for _, y in B], add_special_tokens=False)['input_ids']
            seqs = [(qq + ss)[:320] for qq, ss in zip(q, s)]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long); m = torch.zeros_like(ids)
            for j, sq in enumerate(seqs):
                ids[j, :len(sq)] = torch.tensor(sq); m[j, :len(sq)] = 1
            ids, m = ids.to(dev), m.to(dev)
            model(input_ids=ids, attention_mask=m, use_cache=False)
            if len(states) != a.t_max:
                raise SystemExit(f'hook fired {len(states)} times, expected {a.t_max} (model may not loop the whole stack)')
            for j, sq in enumerate(seqs):
                lo, hi = len(q[j]), len(sq)          # solution tokens predict positions lo..hi-1 from states lo-1..hi-2
                if hi - lo < 2:
                    continue
                tgt = ids[j, lo:hi]
                per = []
                for d in range(a.t_max):
                    h = states[d][j, lo - 1:hi - 1].float()
                    lp = torch.log_softmax(head(norm(states[d][j:j + 1, lo - 1:hi - 1])).float(), -1)[0]
                    nll = -lp.gather(-1, tgt[:, None])[:, 0]
                    per.append((h.cpu(), nll.cpu()))
                recs.append(per)
                if len(recs) <= half:
                    for d in range(a.t_train):
                        fit_states.append(per[d][0][::4])
    X = torch.cat(fit_states).double()
    mu = X.mean(0)
    U, S, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:a.k].T                                           # [hidden, k]
    def resid(h):
        hc = h.double() - mu
        return (((hc - hc @ V @ V.T) ** 2).sum(-1) / (h.double() ** 2).sum(-1)).float()
    test = recs[half:]
    floor = float(torch.cat([resid(per[d][0]) for per in test for d in range(a.t_train)]).mean())
    out = {'model': a.model, 'k': a.k, 'n_test': len(test), 'floor': floor, 'depth': {}}
    nll4 = [per[a.t_train - 1][1] for per in test]
    for d in range(a.t_max):
        r = torch.cat([resid(per[d][0]) for per in test])
        nl = torch.cat([per[d][1] for per in test]); dmg = nl - torch.cat(nll4)
        c = float(np.corrcoef(r.numpy(), dmg.numpy())[0, 1]) if d >= a.t_train else None
        out['depth'][d + 1] = {'drift': float(r.mean()), 'drift_over_floor': float(r.mean()) / floor,
                               'nll': float(nl.mean()), 'corr_drift_damage': c}
        print(f"[ouro] depth {d + 1:2d}  drift/floor {float(r.mean()) / floor:6.2f}  NLL {float(nl.mean()):.4f}"
              + (f"  corr(drift, damage) {c:+.2f}" if c is not None else ''), flush=True)
    d8 = out['depth'][min(8, a.t_max)]
    out['O1'] = bool(d8['drift_over_floor'] >= 3 and (d8['corr_drift_damage'] or 0) >= 0.3)
    print('O1', 'PASS' if out['O1'] else 'FAIL')
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(f'{a.out}/ouro.json', 'w'), indent=1)

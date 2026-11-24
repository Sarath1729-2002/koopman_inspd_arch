"""Drift diagnostic for CONTINUOUS-THOUGHT reasoning (Coconut on GPT-2, GSM8k-Aug), using the earlier lcot code.
Coconut feeds its last hidden state back as the next input embedding, never passing through a token, which is the
same structure as a video generator feeding back its own latents. Re-encoding = decode the thought to a vocabulary
distribution and re-embed it: snap(h) = softmax(lm_head(h)) @ W_emb.

Step 1 (once): train the plain Coconut arm through 5 stages from ckpt_lcot/stage0.pt (as in the earlier run) and save
ckpt_lcot/coconut_final.pt.  Step 2: on the test set, with n = 10 (trained), 14 and 20 thoughts:
  floor     = 1 - cos(snap(e), e) for real input embeddings e of the gold CoT step tokens (on the vocabulary manifold)
  drift_k   = 1 - cos(snap(h_k), h_k) for thought k (scale-free: Coconut's thoughts live at a different norm)
  informative? AUROC of the mean drift predicting a WRONG answer
  intervention (inference only, no retraining): every 2 thoughts replace h by snap(h) rescaled to ||h||
     (hard = nearest token, soft = expected embedding) -> accuracy vs plain Coconut, paired bootstrap over questions
Criteria (fixed before the run):
  C1 drift at the last trained thought >= 3 x floor AND AUROC >= 0.6
  C2 some snap variant beats plain Coconut at 14 or 20 thoughts, accuracy CI > 0
Writes results_rdiag/coconut.json"""
import argparse, json, os, sys, types, numpy as np, torch, torch.nn.functional as F

try:
    from rdiag.lcot.model import setup_tokenizer_model, latent_rollout, greedy
    from rdiag.lcot.train import encode, load, pad, parse, correct, run_stage
except ImportError as e:
    raise SystemExit(f'could not import the latent-CoT code: {e}')


def snapv(model, h, kind):
    W = model.get_input_embeddings().weight
    lg = model.lm_head(h).float()
    e = (F.softmax(lg, -1).to(W.dtype) @ W) if kind == 'soft' else W[lg.argmax(-1)]
    return e * (h.norm(dim=-1, keepdim=True) / (e.norm(dim=-1, keepdim=True) + 1e-6))


@torch.no_grad()
def run(model, tok, test, n, kind, every, dev, bs=256):
    tr = model.transformer
    bot, eot, eos = tok.convert_tokens_to_ids('<|bot|>'), tok.convert_tokens_to_ids('<|eot|>'), tok.eos_token_id
    oks, drifts = [], []
    for i in range(0, len(test), bs):
        B = test[i:i + bs]
        pre, pm = pad([ex['q'] + [bot] for ex in B], tok.pad_token_id, left=True)
        pre, pm = pre.to(dev), pm.to(dev)
        pos = (pm.cumsum(-1) - 1).clamp(min=0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = tr(input_ids=pre, attention_mask=pm, position_ids=pos, use_cache=True)
            cache, h, mask, nxt = o.past_key_values, o.last_hidden_state[:, -1:], pm, pm.sum(-1, keepdim=True)
            dk = []
            for k in range(n):
                s = snapv(model, h, 'soft')
                dk.append((1 - F.cosine_similarity(s.float(), h.float(), dim=-1))[:, 0])
                inp = snapv(model, h, kind) if (kind != 'none' and (k + 1) % every == 0) else h
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
                o = tr(inputs_embeds=inp, attention_mask=mask, position_ids=nxt, past_key_values=cache, use_cache=True)
                cache, h, nxt = o.past_key_values, o.last_hidden_state, nxt + 1
            gen = greedy(model, cache, mask, nxt, torch.full((len(B), 1), eot, dtype=torch.long, device=dev), 24, eos)
        drifts.append(torch.stack(dk, 1).cpu())
        for ex, g in zip(B, gen):
            oks.append(correct(parse(tok.decode(g, skip_special_tokens=False).split(tok.eos_token)[0]), ex['gold']))
    return np.array(oks, float), torch.cat(drifts).numpy()


def auroc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_lcot'); ap.add_argument('--ckpt', default='ckpt_lcot')
    ap.add_argument('--n_test', type=int, default=None); ap.add_argument('--out', default='results_rdiag')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    tok, model = setup_tokenizer_model('openai-community/gpt2', dev)
    final = os.path.join(a.ckpt, 'coconut_final.pt')
    test = encode(tok, load(os.path.join(a.data, 'test.jsonl'), 64 if a.smoke else a.n_test))
    if os.path.exists(final):
        model.load_state_dict(torch.load(final, map_location=dev)); print('loaded', final)
    else:
        model.load_state_dict(torch.load(os.path.join(a.ckpt, 'stage0.pt'), map_location=dev))
        if not a.smoke:
            train = encode(tok, load(os.path.join(a.data, 'train.jsonl')))
            args = types.SimpleNamespace(stages=5, c=2, snap='none', k=2, tau=1.0, lr=1e-4, seed=0, batch=128,
                                         stage0_epochs=2, epochs_per_stage=1, arm='coconut')
            for s in range(1, 6):
                run_stage(model, tok, train, s, args, dev, [])
            torch.save(model.state_dict(), final); print('saved', final, flush=True)
    model.eval()
    W = model.get_input_embeddings().weight
    toks = torch.tensor(sorted({t for ex in test[:500] for st in ex['steps'] for t in st}), device=dev)
    with torch.no_grad():
        e = W[toks]
        floor = float((1 - F.cosine_similarity(snapv(model, e, 'soft').float(), e.float(), dim=-1)).mean())
    res = {'floor': floor, 'runs': {}}
    base = {}
    for n in ((2,) if a.smoke else (10, 14, 20)):
        for kind in ('none', 'hard', 'soft'):
            ok, dr = run(model, tok, test, n, kind, 2, dev)
            r = {'acc': float(ok.mean()), 'drift_per_thought': dr.mean(0).tolist(),
                 'drift_over_floor_last': float(dr[:, -1].mean()) / floor}
            if kind == 'none':
                base[n] = ok
                r['auroc_wrong'] = auroc(dr.mean(1), 1 - ok)
            else:
                d = ok - base[n]; rng = np.random.default_rng(0)
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
                r['delta_vs_none'] = [float(d.mean())] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]
            res['runs'][f'{kind}_{n}'] = r
            print(f"[coconut] n={n:2d} {kind:4s} acc {r['acc']:.3f}  drift/floor last thought {r['drift_over_floor_last']:.2f}"
                  + (f"  AUROC(drift->wrong) {r['auroc_wrong']:.2f}" if 'auroc_wrong' in r else f"  delta {r['delta_vs_none']}"), flush=True)
    if not a.smoke:
        r10 = res['runs']['none_10']
        res['C1'] = bool(r10['drift_over_floor_last'] >= 3 and r10['auroc_wrong'] >= 0.6)
        res['C2'] = any(res['runs'][f'{k}_{n}']['delta_vs_none'][1] > 0 for k in ('hard', 'soft') for n in (14, 20))
        print('C1', 'PASS' if res['C1'] else 'FAIL', ' C2', 'PASS' if res['C2'] else 'FAIL')
        os.makedirs(a.out, exist_ok=True); json.dump(res, open(f'{a.out}/coconut.json', 'w'), indent=1)
    else:
        print('SMOKE OK')

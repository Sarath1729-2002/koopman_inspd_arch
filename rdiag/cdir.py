"""Is Coconut's drift a DIRECTION of faulty reasoning, and does steering against it help? (Nov 24)
Snapping (C2) removed everything off the vocabulary manifold and cost accuracy, so the off-manifold part carries
useful information. Here only one direction of it is used.
  residual   r_k = (h_k - snap_soft(h_k)) / ||h_k||  for thought k (snap norm-matched, as in coconut.py)
  feature    mean of r_k over the n thoughts, per question
  direction  u = mean(feature | wrong) - mean(feature | right) on the FIT fold, normalised; gap g = projection gap
  2-fold cross-fit: questions split by index parity; every score / steer on a fold uses the direction from the other
  difficulty control: AUROCs are stratified by the number of gold CoT steps (within-stratum AUROC, size-weighted)
  steering (inference only): every thought's input h <- h - alpha * g * ||h|| * u, alpha in {0.5, 1, 2} chosen on the
            fit fold, scored on the held-out fold; control = 3 random unit directions, same alpha and size
Criteria (fixed before the run):
  D1  held-out stratified AUROC(direction score -> wrong) >= 0.65 AND minus stratified AUROC(drift size -> wrong):
      bootstrap CI > 0
  D2  at 10 or 14 thoughts, steered accuracy beats plain Coconut (paired CI > 0) AND beats the random-direction control
      (paired CI > 0)
Also reported: the same direction fitted on raw thoughts h (not the off-manifold residual), cos(u_residual, u_raw).
Usage: python -m rdiag.cdir [--eps 3]   -> results_rdiag/cdir[_e3].json"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
from rdiag.coconut import snapv, auroc
from rdiag.lcot.model import setup_tokenizer_model, greedy
from rdiag.lcot.train import encode, load, pad, parse, correct


@torch.no_grad()
def run(model, tok, test, n, dev, steer=None, bs=256):
    """steer: None or (u [D] tensor, scale) -> input h - scale * ||h|| * u at every thought.
    Returns correctness [N], residual feature [N, D], raw-thought feature [N, D], drift size [N]."""
    tr = model.transformer
    bot, eot, eos = tok.convert_tokens_to_ids('<|bot|>'), tok.convert_tokens_to_ids('<|eot|>'), tok.eos_token_id
    oks, R, H, S = [], [], [], []
    for i in range(0, len(test), bs):
        B = test[i:i + bs]
        pre, pm = pad([ex['q'] + [bot] for ex in B], tok.pad_token_id, left=True)
        pre, pm = pre.to(dev), pm.to(dev)
        pos = (pm.cumsum(-1) - 1).clamp(min=0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = tr(input_ids=pre, attention_mask=pm, position_ids=pos, use_cache=True)
            cache, h, mask, nxt = o.past_key_values, o.last_hidden_state[:, -1:], pm, pm.sum(-1, keepdim=True)
            r_acc, h_acc, s_acc = 0, 0, 0
            for k in range(n):
                hf = h.float()
                s = snapv(model, h, 'soft').float()
                nrm = hf.norm(dim=-1, keepdim=True)
                r_acc = r_acc + ((hf - s) / nrm)[:, 0]
                h_acc = h_acc + (hf / nrm)[:, 0]
                s_acc = s_acc + (1 - F.cosine_similarity(s, hf, dim=-1))[:, 0]
                inp = h
                if steer is not None:
                    u, sc = steer
                    inp = (hf - sc * nrm * u.view(1, 1, -1)).to(h.dtype)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
                o = tr(inputs_embeds=inp, attention_mask=mask, position_ids=nxt, past_key_values=cache, use_cache=True)
                cache, h, nxt = o.past_key_values, o.last_hidden_state, nxt + 1
            gen = greedy(model, cache, mask, nxt, torch.full((len(B), 1), eot, dtype=torch.long, device=dev), 24, eos)
        R.append((r_acc / n).cpu()); H.append((h_acc / n).cpu()); S.append((s_acc / n).cpu())
        for ex, g in zip(B, gen):
            oks.append(correct(parse(tok.decode(g, skip_special_tokens=False).split(tok.eos_token)[0]), ex['gold']))
    return np.array(oks, float), torch.cat(R).numpy(), torch.cat(H).numpy(), torch.cat(S).numpy()


def strat_auroc(score, pos, strata):
    tot, w = 0.0, 0
    for s in np.unique(strata):
        m = strata == s
        if 0 < pos[m].sum() < m.sum():
            tot += auroc(score[m], pos[m]) * m.sum(); w += m.sum()
    return tot / max(w, 1)


def boot(d, rng, B=5000):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return [float(d.mean())] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def direction(feat, wrong):
    d = feat[wrong == 1].mean(0) - feat[wrong == 0].mean(0)
    u = d / (np.linalg.norm(d) + 1e-12)
    g = float((feat[wrong == 1] @ u).mean() - (feat[wrong == 0] @ u).mean())
    return u, g


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_lcot'); ap.add_argument('--ckpt', default='ckpt_lcot')
    ap.add_argument('--out', default='results_rdiag'); ap.add_argument('--eps', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    sfx = '' if a.eps == 1 else f'_e{a.eps}'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(0); rng = np.random.default_rng(0)
    tok, model = setup_tokenizer_model('openai-community/gpt2', dev)
    ck = os.path.join(a.ckpt, f'coconut_final{sfx}.pt')
    if not os.path.exists(ck) and not a.smoke:
        raise SystemExit(f'MISSING {ck} (run rdiag.coconut --eps {a.eps} first)')
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=dev))
    model.eval()
    test = encode(tok, load(os.path.join(a.data, 'test.jsonl'), 64 if a.smoke else None))
    N = len(test); fold = np.arange(N) % 2
    strata = np.array([min(len(ex['steps']), 6) for ex in test])
    alphas = (1.0,) if a.smoke else (0.5, 1.0, 2.0)
    res = {'ckpt': ck, 'n': N, 'runs': {}}
    D = model.config.n_embd
    rand = [torch.randn(D, generator=torch.Generator().manual_seed(s)) for s in range(3)]
    rand = [(r / r.norm()).to(dev) for r in rand]
    for n in ((2,) if a.smoke else (10, 14)):
        ok0, R, H, S = run(model, tok, test, n, dev)
        wrong = 1 - ok0
        score_r, score_h = np.zeros(N), np.zeros(N); U = {}
        for f in (0, 1):
            fit, ev = fold == f, fold != f
            ur, gr = direction(R[fit], wrong[fit]); uh, _ = direction(H[fit], wrong[fit])
            U[f] = (ur, gr, uh)
            score_r[ev], score_h[ev] = R[ev] @ ur, H[ev] @ uh
        a_dir = strat_auroc(score_r, wrong, strata); a_raw = strat_auroc(score_h, wrong, strata)
        a_mag = strat_auroc(S, wrong, strata)
        diffs = []
        for _ in range(1000):
            ix = rng.integers(0, N, N)
            diffs.append(strat_auroc(score_r[ix], wrong[ix], strata[ix]) - strat_auroc(S[ix], wrong[ix], strata[ix]))
        r = {'acc_none': float(ok0.mean()), 'auroc_dir': a_dir, 'auroc_raw_dir': a_raw, 'auroc_drift_size': a_mag,
             'auroc_dir_minus_size': [a_dir - a_mag] + [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
             'cos_resid_raw_dir': float(np.mean([U[f][0] @ U[f][2] for f in (0, 1)])), 'steer': {}}
        print(f'[cdir] n={n} acc {ok0.mean():.3f}  stratified AUROC: direction {a_dir:.3f}  raw-thought direction {a_raw:.3f}  '
              f'drift size {a_mag:.3f}  dir-size {r["auroc_dir_minus_size"]}', flush=True)
        # steering: per fold, run with the OTHER fold's direction; record accuracy on every question for each alpha
        acc_s = {al: np.zeros(N) for al in alphas}; acc_c = {al: np.zeros(N) for al in alphas}
        acc_fit = {al: [0.0, 0.0] for al in alphas}
        for f in (0, 1):
            ur, gr, _ = U[f]; ev, fit = fold != f, fold == f
            u = torch.tensor(ur, dtype=torch.float32, device=dev)
            for al in alphas:
                ok_s = run(model, tok, test, n, dev, steer=(u, al * gr))[0]
                ok_c = np.mean([run(model, tok, test, n, dev, steer=(rv, al * gr))[0] for rv in rand], 0)
                acc_s[al][ev] = ok_s[ev]; acc_c[al][ev] = ok_c[ev]
                acc_fit[al][f] = float(ok_s[fit].mean() - ok0[fit].mean())
                print(f'[cdir] n={n} fold {f} alpha {al}: steered acc fit/held-out {ok_s[fit].mean():.3f}/{ok_s[ev].mean():.3f} '
                      f'random {ok_c[ev].mean():.3f}', flush=True)
        # alpha chosen on the fit fold, per fold; scored on the held-out fold
        best = {f: max(alphas, key=lambda al: acc_fit[al][f]) for f in (0, 1)}
        sel_s = np.where(fold != 0, acc_s[best[0]], acc_s[best[1]])
        sel_c = np.where(fold != 0, acc_c[best[0]], acc_c[best[1]])
        r['steer'] = {'alpha_chosen': best, 'acc_steered': float(sel_s.mean()), 'acc_random': float(sel_c.mean()),
                      'delta_vs_none': boot(sel_s - ok0, rng), 'delta_vs_random': boot(sel_s - sel_c, rng),
                      'by_alpha': {str(al): {'acc': float(acc_s[al].mean()), 'random': float(acc_c[al].mean())} for al in alphas}}
        print(f"[cdir] n={n} steered {sel_s.mean():.3f} vs none {ok0.mean():.3f} vs random {sel_c.mean():.3f}  "
              f"delta none {r['steer']['delta_vs_none']}  delta random {r['steer']['delta_vs_random']}", flush=True)
        res['runs'][str(n)] = r
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    r10 = res['runs']['10']
    res['D1'] = bool(r10['auroc_dir'] >= 0.65 and r10['auroc_dir_minus_size'][1] > 0)
    res['D2'] = any(res['runs'][n]['steer']['delta_vs_none'][1] > 0 and res['runs'][n]['steer']['delta_vs_random'][1] > 0
                    for n in ('10', '14'))
    print('D1', 'PASS' if res['D1'] else 'FAIL', ' D2', 'PASS' if res['D2'] else 'FAIL')
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f'{a.out}/cdir{sfx}.json', 'w'), indent=1)

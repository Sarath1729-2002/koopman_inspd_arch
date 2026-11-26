"""Ouro + sparse Koopman autoencoder (Nov 26): the regime-aware rerun of the Koopman checks that failed with one
dense linear map (KP1/KP2) and of the transfer check that failed with one global direction (G1o).
A sparse Koopman autoencoder (koop/skae.py, after Li ... Goroshin 2026) is fitted on loop 1->4 trajectories of each
token (whitened PCA-256 of h/||h||), on the fit halves of GSM8k solutions AND wikitext. Regimes = k-means (R=8) on the
loop-4 supports. Directions are damage directions on raw normalised states (N3: raw >= residual), damage = NLL(loop 8)
- NLL(loop 4); steering h <- h - alpha * g * ||h|| * u at loops 5-8, alpha from {0.25, 0.5, 1, 2, 4} chosen on the fit
half; scored on held-out halves, bootstrap over sequences. A token's regime is read from its own loop-4 state online.
Criteria (fixed before the run):
  SO1 support novelty at loop 8 (1 - best Jaccard overlap with loop 1-4 supports) tracks damage better than drift
      size: corr(novelty, damage) - corr(drift size, damage) on held-out GSM8k, CI > 0
  SO2 regime-specific directions (fitted on GSM8k) beat one global GSM8k direction on held-out GSM8k loop-8 NLL: CI < 0
  SO3 directions fitted on GSM8k + wikitext: the regime-specific set lowers held-out WIKITEXT loop-8 NLL vs no
      steering (CI < 0) AND vs one global mixed direction (CI < 0)   [the rerun of G1o]
Also reported: how cleanly regimes separate the two domains, per-regime token counts, GSM8k effect of the mixed sets.
Writes results_rdiag/oskae[_tag].json"""
import argparse, json, os, numpy as np, torch
from rdiag.ouro import find_parts
from rdiag.odir import boot_seq
from koop.skae import Space, train_skae, supports, novelty, fit_regimes, assign

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ByteDance/Ouro-1.4B'); ap.add_argument('--t_max', type=int, default=8)
    ap.add_argument('--n', type=int, default=512); ap.add_argument('--n_wiki', type=int, default=384)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--R', type=int, default=8)
    ap.add_argument('--data', default='data_lcot/test.jsonl'); ap.add_argument('--out', default='results_rdiag')
    ap.add_argument('--tag', default=''); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; rng = np.random.default_rng(0); torch.manual_seed(0)
    if a.smoke:
        a.n, a.n_wiki, a.R = 24, 16, 2
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (getattr(tok, 'eos_token_id', None) or 0)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); cfg.total_ut_steps = a.t_max
    if hasattr(cfg, 'early_exit_threshold'):
        cfg.early_exit_threshold = 1.0
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    inner, layers, norm, head = find_parts(model)
    states, STEER = [], [None]   # STEER[0] = (fn(h_loop4) -> (U [B,L,H], G [B,L,1]), alpha)
    CACHE = [None]

    def post(mod, i, o):
        h = o[0] if isinstance(o, tuple) else o
        d = len(states) + 1
        if STEER[0] is not None and 5 <= d <= 8:
            fn, al = STEER[0]
            if CACHE[0] is None:
                CACHE[0] = fn(states[3])
            U, G = CACHE[0]
            hf = h.float(); h = (hf - al * G * hf.norm(dim=-1, keepdim=True) * U).to(h.dtype)
            o = (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        states.append(h.detach())
        return o
    def pre(*x, **k):
        states.clear(); CACHE[0] = None
    inner.register_forward_pre_hook(pre)
    layers[-1].register_forward_hook(post)
    KEEP = (0, 1, 2, 3, 7)

    @torch.no_grad()
    def forward(tx, keep=False, steer=None):
        STEER[0] = steer; recs = []
        for i in range(0, len(tx), a.bs):
            B = tx[i:i + a.bs]
            q = tok([x for x, _ in B], add_special_tokens=False)['input_ids']
            s = tok([y for _, y in B], add_special_tokens=False)['input_ids']
            seqs = [(qq + ss)[:320] for qq, ss in zip(q, s)]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), pad, dtype=torch.long); m = torch.zeros_like(ids)
            for j, sq in enumerate(seqs):
                ids[j, :len(sq)] = torch.tensor(sq); m[j, :len(sq)] = 1
            ids, m = ids.to(dev), m.to(dev)
            model(input_ids=ids, attention_mask=m, use_cache=False)
            for j, sq in enumerate(seqs):
                lo, hi = len(q[j]), len(sq)
                if hi - lo < 2:
                    recs.append(None); continue
                tgt = ids[j, lo:hi]; r = {'nll': {}}
                for d in (3, 7):
                    lp = torch.log_softmax(head(norm(states[d][j:j + 1, lo - 1:hi - 1])).float()[0], -1)
                    r['nll'][d] = (-lp.gather(-1, tgt[:, None])[:, 0]).cpu()
                if keep:
                    r['h'] = {d: states[d][j, lo - 1:hi - 1].half().cpu() for d in KEEP}
                recs.append(r)
        STEER[0] = None
        return recs
    # data: GSM8k solutions + wikitext passages, each split into fit / held-out halves
    rows = [json.loads(l) for l in open(a.data)][:a.n]
    gsm = [(r['question'] + '\n', '\n'.join(r['steps']) + '\n### ' + r['answer']) for r in rows]
    wiki = []
    try:
        if os.environ.get('OSKAE_TEXT'):   # jsonl with "text" (tests / other corpora)
            wt = [json.loads(l)['text'] for l in open(os.environ['OSKAE_TEXT'])][:a.n_wiki]
        else:
            import datasets
            wt = [t for t in datasets.load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')['text'] if len(t) > 600][:a.n_wiki]
        for t in wt:
            ii = tok(t, add_special_tokens=False)['input_ids']; wiki.append((tok.decode(ii[:48]), tok.decode(ii[48:300])))
    except Exception as e:
        print('[oskae] no wikitext:', str(e)[:150], flush=True)
    base = {'gsm': forward(gsm, keep=True), 'wiki': forward(wiki, keep=True) if wiki else []}
    split = {}
    for dom, recs in base.items():
        ok_i = [i for i, r in enumerate(recs) if r is not None]; h_ = len(recs) // 2
        split[dom] = ([i for i in ok_i if i < h_], [i for i in ok_i if i >= h_])
    nrm = lambda x: x.float() / x.float().norm(dim=-1, keepdim=True)
    cat = lambda dom, idx, f: torch.cat([f(base[dom][i]) for i in idx]) if idx else torch.zeros(0)
    fit_doms = [d for d in ('gsm', 'wiki') if split[d][0]]
    X = torch.cat([cat(d, split[d][0], lambda b: torch.cat([nrm(b['h'][t][::3]) for t in (0, 1, 2, 3)])) for d in fit_doms])
    sp = Space(X, r=64 if a.smoke else 256); del X
    sp_dev = Space.load({k: v.to(dev) for k, v in sp.state().items()})
    traj = torch.cat([cat(d, split[d][0], lambda b: torch.stack([sp(nrm(b['h'][t])) for t in (0, 1, 2, 3)], 1)) for d in fit_doms])
    net = train_skae(traj, traj.shape[-1], m=256, k=16, steps=200 if a.smoke else 4000, L=3, dev=dev, log=lambda s: print(s, flush=True))
    k_ = 16
    bank = supports(net, traj[:, :4].flatten(0, 1)[::5].to(dev)); bank = bank[torch.randperm(len(bank))[:8192]]
    S4 = {d: {part: cat(d, split[d][pi], lambda b: supports(net, sp(nrm(b['h'][3])).to(dev))) for pi, part in ((0, 'fit'), (1, 'ev'))} for d in fit_doms}
    C = fit_regimes(torch.cat([S4[d]['fit'] for d in fit_doms]), R=a.R)
    # domain separation of regimes
    reg_fit = {d: assign(S4[d]['fit'], C).numpy() for d in fit_doms}
    sep = {}
    if 'wiki' in reg_fit:
        for r_ in range(a.R):
            ng, nw = int((reg_fit['gsm'] == r_).sum()), int((reg_fit['wiki'] == r_).sum())
            sep[r_] = {'gsm': ng, 'wiki': nw}
        purity = sum(max(v['gsm'], v['wiki']) for v in sep.values()) / max(1, sum(v['gsm'] + v['wiki'] for v in sep.values()))
    else:
        purity = None
    res = {'model': a.model, 'regimes': sep, 'domain_purity': purity, 'novelty_bank': len(bank)}
    print(f'[oskae] regime/domain purity {purity}', flush=True)
    # SO1: novelty vs drift size vs damage, held-out GSM8k
    PV = sp.V.float(); pmu = sp.mu.float()
    def drift_size(h):
        hc = nrm(h) - pmu.cpu(); return (hc - hc @ PV.cpu() @ PV.cpu().T).pow(2).sum(-1)
    ev = split['gsm'][1]
    nov8 = [novelty(supports(net, sp(nrm(base['gsm'][i]['h'][7])).to(dev)), bank, k_).cpu() for i in ev]
    siz8 = [drift_size(base['gsm'][i]['h'][7]) for i in ev]
    dmg = [base['gsm'][i]['nll'][7] - base['gsm'][i]['nll'][3] for i in ev]
    def c_pair(ix):
        n_ = torch.cat([nov8[k] for k in ix]).numpy(); s_ = torch.cat([siz8[k] for k in ix]).numpy(); d_ = torch.cat([dmg[k] for k in ix]).numpy()
        return np.corrcoef(n_, d_)[0, 1], np.corrcoef(s_, d_)[0, 1]
    cn, cs = c_pair(range(len(ev)))
    bs_ = [np.subtract(*c_pair(rng.integers(0, len(ev), len(ev)))) for _ in range(500)]
    res['SO1_detail'] = {'corr_novelty': float(cn), 'corr_drift_size': float(cs), 'diff': [float(cn - cs)] + [float(x) for x in np.percentile(bs_, [2.5, 97.5])]}
    res['SO1'] = bool(res['SO1_detail']['diff'][1] > 0)
    print(f"[oskae] SO1 corr with damage: novelty {cn:+.3f} vs drift size {cs:+.3f}  {res['SO1_detail']['diff']}", flush=True)
    # directions
    def ddir(F_, dm):
        if len(dm) < (20 if a.smoke else 200):
            return None
        qh, ql = torch.quantile(dm, 0.75), torch.quantile(dm, 0.25)
        d = F_[dm >= qh].mean(0) - F_[dm <= ql].mean(0); v = d / d.norm()
        return v, float((F_[dm >= qh] @ v).mean() - (F_[dm <= ql] @ v).mean())
    def dirsets(doms):
        F_ = torch.cat([cat(d, split[d][0], lambda b: nrm(b['h'][7])) for d in doms])
        D_ = torch.cat([cat(d, split[d][0], lambda b: b['nll'][7] - b['nll'][3]) for d in doms])
        R_ = torch.cat([torch.from_numpy(reg_fit[d]) for d in doms])
        ug, gg = ddir(F_, D_)
        U = torch.stack([ug] * a.R); G = torch.full((a.R,), gg)
        for r_ in range(a.R):
            o_ = ddir(F_[R_ == r_], D_[R_ == r_])
            if o_ is not None:
                U[r_], G[r_] = o_
        return (ug, gg), (U, G)
    def make_fn(U, G):
        Ud, Gd, Cd = U.to(dev), G.to(dev), C.to(dev)
        def fn(h4):
            Bq, Lq, H = h4.shape
            reg = assign(supports(net, sp_dev(nrm(h4.flatten(0, 1)))).to(dev), Cd)
            return Ud[reg].view(Bq, Lq, H), Gd[reg].view(Bq, Lq, 1)
        return fn
    def glob_fn(u, g):
        ud = u.to(dev)
        return lambda h4: (ud.view(1, 1, -1).expand(h4.shape[0], h4.shape[1], -1), torch.full((*h4.shape[:2], 1), g, device=dev))
    alphas = (1.0,) if a.smoke else (0.25, 0.5, 1.0, 2.0, 4.0)
    tot = lambda recs, idx: [(float(recs[i]['nll'][7].sum()), len(recs[i]['nll'][7])) for i in idx]
    diff = lambda x, y: [(p[0] - q[0], p[1]) for p, q in zip(x, y)]
    def best_run(fn, doms):
        runs_ = {al: {d: forward(gsm if d == 'gsm' else wiki, steer=(fn, al)) for d in doms} for al in alphas}
        def fitscore(al):
            return sum(boot_seq(tot(runs_[al][d], split[d][0]), rng, 1)[0] for d in doms)
        al = min(alphas, key=fitscore); return al, runs_[al]
    (ug, gg), (Ur, Gr) = dirsets(['gsm'])
    al_g, rg = best_run(glob_fn(ug, gg), ['gsm']); al_r, rr = best_run(make_fn(Ur, Gr), ['gsm'])
    b_ev = tot(base['gsm'], ev)
    res['SO2_detail'] = {'alpha_global': al_g, 'alpha_regime': al_r, 'nll8_none': boot_seq(b_ev, rng, 1)[0],
                         'global_vs_none': boot_seq(diff(tot(rg['gsm'], ev), b_ev), rng),
                         'regime_vs_none': boot_seq(diff(tot(rr['gsm'], ev), b_ev), rng),
                         'regime_vs_global': boot_seq(diff(tot(rr['gsm'], ev), tot(rg['gsm'], ev)), rng)}
    res['SO2'] = bool(res['SO2_detail']['regime_vs_global'][2] < 0)
    print(f"[oskae] SO2 GSM8k: global vs none {res['SO2_detail']['global_vs_none']}  regime vs none {res['SO2_detail']['regime_vs_none']}  "
          f"regime vs global {res['SO2_detail']['regime_vs_global']}", flush=True)
    if 'wiki' in fit_doms:
        (ugm, ggm), (Urm, Grm) = dirsets(['gsm', 'wiki'])
        al_gm, rgm = best_run(glob_fn(ugm, ggm), ['gsm', 'wiki']); al_rm, rrm = best_run(make_fn(Urm, Grm), ['gsm', 'wiki'])
        evw = split['wiki'][1]; bw = tot(base['wiki'], evw)
        res['SO3_detail'] = {'alpha_global_mix': al_gm, 'alpha_regime_mix': al_rm, 'wiki_nll8_none': boot_seq(bw, rng, 1)[0],
                             'wiki_regime_vs_none': boot_seq(diff(tot(rrm['wiki'], evw), bw), rng),
                             'wiki_global_vs_none': boot_seq(diff(tot(rgm['wiki'], evw), bw), rng),
                             'wiki_regime_vs_global': boot_seq(diff(tot(rrm['wiki'], evw), tot(rgm['wiki'], evw)), rng),
                             'gsm_regime_vs_none': boot_seq(diff(tot(rrm['gsm'], ev), b_ev), rng),
                             'gsm_global_vs_none': boot_seq(diff(tot(rgm['gsm'], ev), b_ev), rng)}
        d3 = res['SO3_detail']
        res['SO3'] = bool(d3['wiki_regime_vs_none'][2] < 0 and d3['wiki_regime_vs_global'][2] < 0)
        print(f"[oskae] SO3 wikitext: regime vs none {d3['wiki_regime_vs_none']}  global vs none {d3['wiki_global_vs_none']}  "
              f"regime vs global {d3['wiki_regime_vs_global']}; GSM8k regime vs none {d3['gsm_regime_vs_none']}", flush=True)
    crit = [k for k in ('SO1', 'SO2', 'SO3') if k in res]
    print(' '.join(f"{k} {'PASS' if res[k] else 'FAIL'}" for k in crit))
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/oskae{('_' + a.tag) if a.tag else ''}.json", 'w'), indent=1, default=float)

"""Ouro: is the damage direction real, useful, general, and Koopman-explainable? (Nov 25, follows rdiag/odir.py, where
subtracting one off-manifold "damage" direction u at loops 5-8 lowered depth-8 loss by 0.035 vs none, 0.055 vs random)

Setup as in odir.py: GSM8k-Aug test solutions, teacher-forced; manifold = PCA(256) of loop 1-4 states on the fit half;
residual r = off-manifold part of h / ||h||; damage = NLL(depth d) - NLL(depth 4). Fit half = first n/2 sequences,
held-out half = the rest. Steering at the loops in a window W: h <- h - alpha * g * ||h|| * v for a unit direction v
(g = damage gap of u, so every direction gets the SAME size); alpha chosen on the fit half, everything scored held-out.

Directions compared
  u       damage direction (top vs bottom damage quartile, residual space, depth 8)
  rand    50 random unit directions                         -- "any push helps"
  shuf    20 directions fitted with shuffled damage labels  -- "the fitting step finds something anyway"
  mean    label-free drift direction mean(r_8) - mean(r_4)   -- "it is only re-centering"
  raw     damage direction fitted on raw states h/||h||      -- "the off-manifold part does not matter"
  kmode   Koopman/DMD: affine linear map y_{d+1} = A y_d + b fitted on residual PCA(64) coords of loops 4..12
          (no labels); leading eigenvector of A, oriented along increasing drift
  katt    the same map's fixed point y* = (I - A)^-1 b: direction from the depth-4 states to the attractor
Criteria (fixed before the run; held-out half, bootstrap over sequences):
  N1  u beats >= 95% of the random AND >= 95% of the shuffled-label directions on depth-8 NLL (window 5-8, u's alpha)
  N2  u beats the label-free mean direction (each at its own alpha): paired CI < 0
  N3  u (residual) beats raw (each at its own alpha): paired CI < 0
  U1  extra loops become useful: steered depth 5 (window {5}) or depth 6 (window {5,6}) has lower NLL than unsteered
      depth 4: CI < 0
  U2  answers: GSM8k greedy accuracy (3-shot, 200 questions) with 6 loops steered (window {5,6}) beats 6 loops
      unsteered: paired CI > 0  (also reported: 4 loops, 8 loops +- steering)
  KP1 a Koopman direction (kmode or katt) aligns with u: |cos| >= 0.5
  KP2 a Koopman direction (no labels) lowers depth-8 NLL vs none (CI < 0) AND beats >= 95% of random directions
  A1  the "mode-seeking attractor" claim: beyond depth 4 the per-loop step size shrinks (Spearman(depth, step) <= -0.8)
      AND entropy at depth 8 < depth 4 while NLL at depth 8 > depth 4 (confident and wrong)
  A2  u is a better damage signal than entropy drop: corr(direction score, damage) - corr(entropy drop, damage) CI > 0
  OH2 stopping by the DIRECTION score (keep looping past 4 while s_{d+1} <= s_4 + tau * g) beats fixed 4 loops: CI < 0
  G1o u fitted on GSM8k lowers depth-8 NLL on other text (wikitext-103 test, if cached; or --ood jsonl with "text")
      vs none AND vs the mean of 10 random directions: CI < 0
Writes results_rdiag/odir2[_tag].json"""
import argparse, json, os, numpy as np, torch
from rdiag.ouro import find_parts
from rdiag.odir import boot_seq


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ByteDance/Ouro-1.4B'); ap.add_argument('--t_train', type=int, default=4)
    ap.add_argument('--t_max', type=int, default=12); ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--k', type=int, default=256); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--data', default='data_lcot/test.jsonl'); ap.add_argument('--train', default='data_lcot/train.jsonl')
    ap.add_argument('--ood', default=''); ap.add_argument('--gen_n', type=int, default=200)
    ap.add_argument('--n_rand', type=int, default=50); ap.add_argument('--n_shuf', type=int, default=20)
    ap.add_argument('--out', default='results_rdiag'); ap.add_argument('--tag', default='')
    ap.add_argument('--no_gen', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; rng = np.random.default_rng(0); torch.manual_seed(0)
    if a.smoke:
        a.n_rand, a.n_shuf, a.gen_n = 3, 2, 4
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def load_model(T):
        cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); cfg.total_ut_steps = T
        if hasattr(cfg, 'early_exit_threshold'):
            cfg.early_exit_threshold = 1.0
        return AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    model = load_model(a.t_max)
    states, STEER = [], [None]   # STEER[0] = (v [H] float on dev, scale, set of loop indices)

    def hooks(m):
        inner, layers, norm, head = find_parts(m)

        def post(mod, i, o):
            h = o[0] if isinstance(o, tuple) else o
            d = len(states) + 1
            if STEER[0] is not None and d in STEER[0][2]:
                v, sc, _ = STEER[0]
                hf = h.float(); h = (hf - sc * hf.norm(dim=-1, keepdim=True) * v).to(h.dtype)
                o = (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            states.append(h.detach())
            return o
        inner.register_forward_pre_hook(lambda *x, **k: states.clear())
        layers[-1].register_forward_hook(post)
        return norm, head
    norm, head = hooks(model)
    rows = [json.loads(l) for l in open(a.data)][: (24 if a.smoke else a.n)]
    texts = [(r['question'] + '\n', '\n'.join(r['steps']) + '\n### ' + r['answer']) for r in rows]
    half = len(texts) // 2

    @torch.no_grad()
    def forward(tx, keep=False, steer=None, depths=(3, 4, 5, 7, 11)):
        """per sequence: dict depth-> nll [n_tok] (0-based depth); with keep also 'h' list of states and 'ent'"""
        STEER[0] = steer; recs = []
        for i in range(0, len(tx), a.bs):
            B = tx[i:i + a.bs]
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
                raise SystemExit(f'hook fired {len(states)} times, expected {a.t_max}')
            for j, sq in enumerate(seqs):
                lo, hi = len(q[j]), len(sq)
                if hi - lo < 2:
                    recs.append(None); continue
                tgt = ids[j, lo:hi]; r = {'nll': {}}
                for d in (range(a.t_max) if keep else depths):
                    lg = head(norm(states[d][j:j + 1, lo - 1:hi - 1])).float()[0]
                    lp = torch.log_softmax(lg, -1)
                    r['nll'][d] = (-lp.gather(-1, tgt[:, None])[:, 0]).cpu()
                    if keep:
                        r.setdefault('ent', {})[d] = (-(lp.exp() * lp).sum(-1)).cpu()
                if keep:
                    r['h'] = [states[d][j, lo - 1:hi - 1].half().cpu() for d in range(a.t_max)]
                recs.append(r)
        STEER[0] = None
        return recs

    base = forward(texts, keep=True)
    ok_i = [i for i, r in enumerate(base) if r is not None]
    fit_i, ev_i = [i for i in ok_i if i < half], [i for i in ok_i if i >= half]
    X = torch.cat([base[i]['h'][d][::4].float() for i in fit_i for d in range(a.t_train)]).double()
    mu = X.mean(0); V = torch.linalg.svd(X - mu, full_matrices=False)[2][:a.k].T.contiguous(); del X

    def resid(h):
        hc = h.double() - mu
        return ((hc - hc @ V @ V.T) / h.double().norm(dim=-1, keepdim=True)).float()
    D4, D8 = a.t_train - 1, min(8, a.t_max) - 1
    for i in ok_i:
        base[i]['r'] = {d: resid(base[i]['h'][d].float()) for d in range(D4, a.t_max)}
        base[i]['dmg'] = base[i]['nll'][D8] - base[i]['nll'][D4]
    cat = lambda idx, f: torch.cat([f(base[i]) for i in idx])
    R8f, R4f, dmf = cat(fit_i, lambda b: b['r'][D8]), cat(fit_i, lambda b: b['r'][D4]), cat(fit_i, lambda b: b['dmg'])

    def damage_dir(F_, dm):
        qh, ql = torch.quantile(dm, 0.75), torch.quantile(dm, 0.25)
        d = F_[dm >= qh].mean(0) - F_[dm <= ql].mean(0); v = d / d.norm()
        return v, float((F_[dm >= qh] @ v).mean() - (F_[dm <= ql] @ v).mean())
    u, g = damage_dir(R8f, dmf)
    H8f = cat(fit_i, lambda b: b['h'][D8].float() / b['h'][D8].float().norm(dim=-1, keepdim=True))
    w_raw, _ = damage_dir(H8f, dmf); del H8f
    m = R8f.mean(0) - R4f.mean(0); m_dir = m / m.norm()
    shuf = []
    for s_ in range(a.n_shuf):
        perm = torch.from_numpy(np.random.default_rng(100 + s_).permutation(len(dmf)))
        shuf.append(damage_dir(R8f, dmf[perm])[0])
    rand = [torch.randn(u.shape[0], generator=torch.Generator().manual_seed(s_)) for s_ in range(a.n_rand)]
    rand = [r_ / r_.norm() for r_ in rand]
    # Koopman / DMD on the extrapolated loop dynamics of the residual (no labels)
    Rs = [torch.stack([base[i]['r'][d][::4] for d in range(D4, a.t_max)]) for i in fit_i]   # [T', n_tok, H]
    Rall = torch.cat([x.flatten(0, 1) for x in Rs]).double(); rmu = Rall.mean(0)
    P = torch.linalg.svd(Rall - rmu, full_matrices=False)[2][:64].T                         # [H, 64]
    Y0 = torch.cat([(x[:-1] - rmu) @ P for x in [x.double() for x in Rs]], 1).flatten(0, 1)
    Y1 = torch.cat([(x[1:] - rmu) @ P for x in [x.double() for x in Rs]], 1).flatten(0, 1)
    Z = torch.cat([Y0, torch.ones(len(Y0), 1, dtype=Y0.dtype)], 1)
    lam = 1e-4 * torch.trace(Z.T @ Z) / Z.shape[1]
    M = torch.linalg.solve(Z.T @ Z + lam * torch.eye(Z.shape[1], dtype=Z.dtype), Z.T @ Y1)
    A, bvec = M[:-1].T, M[-1]
    ev, evec = torch.linalg.eig(A)
    j = int(ev.abs().argmax()); rho = float(ev.abs().max())
    km = (P @ evec[:, j].real).float(); km = km / km.norm()
    if float((R8f @ km).mean() - (R4f @ km).mean()) < 0:
        km = -km
    ystar = torch.linalg.solve(torch.eye(A.shape[0], dtype=A.dtype) - A, bvec)
    y4 = ((R4f.double() - rmu) @ P).mean(0)
    ka = (P @ (ystar - y4)).float(); ka = ka / ka.norm()
    del Rs, Rall, Y0, Y1, Z
    dirs = {'u': u, 'mean': m_dir, 'raw': w_raw, 'kmode': km, 'katt': ka}
    cosu = {k: float(v @ u) for k, v in dirs.items()}
    res = {'model': a.model, 'n_fit': len(fit_i), 'n_eval': len(ev_i), 'gap': g, 'cos_with_u': cosu,
           'koopman': {'rho': rho, 'eig_top8': sorted(ev.abs().tolist())[-8:], 'eig_lead': [float(ev[j].real), float(ev[j].imag)]}}
    print(f'[odir2] gap {g:.4f}  cos with u: ' + '  '.join(f'{k} {v:+.3f}' for k, v in cosu.items()) + f'  rho(A) {rho:.3f}', flush=True)
    # --- A1 / A2: dynamics diagnostics on the held-out half (no intervention)
    per_d = {}
    for d in range(a.t_max):
        nl = cat(ev_i, lambda b: b['nll'][d]); en = cat(ev_i, lambda b: b['ent'][d])
        st = cat(ev_i, lambda b: (b['h'][d].float() - b['h'][d - 1].float()).norm(dim=-1) / b['h'][d - 1].float().norm(dim=-1)) if d else None
        per_d[d + 1] = {'nll': float(nl.mean()), 'entropy': float(en.mean()), 'step': float(st.mean()) if st is not None else None,
                        'norm_ratio_vs_d4': float(cat(ev_i, lambda b: b['h'][d].float().norm(dim=-1) / b['h'][D4].float().norm(dim=-1)).mean())}
    from scipy.stats import spearmanr
    steps = [per_d[d]['step'] for d in range(a.t_train + 1, a.t_max + 1)]
    sp = float(spearmanr(range(len(steps)), steps)[0])
    res['dynamics'] = per_d; res['step_spearman_beyond4'] = sp
    res['A1'] = bool(sp <= -0.8 and per_d[8]['entropy'] < per_d[4]['entropy'] and per_d[8]['nll'] > per_d[4]['nll'])
    def a2(ix):
        sc = torch.cat([base[ev_i[k]]['r'][D8] @ u for k in ix]).numpy()
        ed = torch.cat([base[ev_i[k]]['ent'][D4] - base[ev_i[k]]['ent'][D8] for k in ix]).numpy()
        dm = torch.cat([base[ev_i[k]]['dmg'] for k in ix]).numpy()
        return corr(sc, dm), corr(ed, dm)
    c_dir, c_ent = a2(range(len(ev_i)))
    bs = [np.subtract(*a2(rng.integers(0, len(ev_i), len(ev_i)))) for _ in range(500)]
    res['A2_detail'] = {'corr_dir': c_dir, 'corr_entropy_drop': c_ent, 'diff': [c_dir - c_ent] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]}
    res['A2'] = bool(res['A2_detail']['diff'][1] > 0)
    print(f"[odir2] dynamics: step size beyond 4 Spearman {sp:+.2f}; entropy d4/d8 {per_d[4]['entropy']:.3f}/{per_d[8]['entropy']:.3f}; "
          f"NLL d4/d8 {per_d[4]['nll']:.4f}/{per_d[8]['nll']:.4f}; corr with damage: direction {c_dir:+.3f} entropy drop {c_ent:+.3f}", flush=True)
    # --- OH2: stopping by direction score
    def stop(idx, tau):
        out = []
        for i in idx:
            b = base[i]; s4 = b['r'][D4] @ u; n_ = len(s4)
            dsel = torch.full((n_,), D4, dtype=torch.long); go = torch.ones(n_, dtype=torch.bool)
            for d in range(D4 + 1, a.t_max):
                go = go & ((b['r'][d] @ u) <= s4 + tau * g); dsel = torch.where(go, torch.full_like(dsel, d), dsel)
            nl = torch.stack([b['nll'][d] for d in range(a.t_max)])
            out.append((float(nl.gather(0, dsel[None])[0].sum()), n_))
        return out
    taus = [0.25 * k for k in range(9)]
    tau = min(taus, key=lambda t: boot_seq(stop(fit_i, t), rng, 1)[0])
    st_, fx = stop(ev_i, tau), [(float(base[i]['nll'][D4].sum()), len(base[i]['nll'][D4])) for i in ev_i]
    res['OH2_detail'] = {'tau': tau, 'nll_stop': boot_seq(st_, rng), 'nll_fixed4': boot_seq(fx, rng),
                         'diff': boot_seq([(x[0] - y[0], x[1]) for x, y in zip(st_, fx)], rng)}
    res['OH2'] = bool(res['OH2_detail']['diff'][2] < 0)
    for i in ok_i:   # free states
        base[i].pop('h', None)
    # --- steering
    texts_ev = [texts[i] for i in ev_i]
    def tot(recs, idx_map, d):
        return [(float(recs[k]['nll'][d].sum()), len(recs[k]['nll'][d])) for k in idx_map]
    def steer_run(v, al, win, tx=None):
        return forward(tx if tx is not None else texts, steer=(v.to(dev), al * g, set(win)))
    alphas = (1.0,) if a.smoke else (0.25, 0.5, 1.0, 2.0, 4.0)
    W5, W56, W58 = (5,), (5, 6), tuple(range(5, min(8, a.t_max) + 1))
    base_ev = {d: tot(base, ev_i, d) for d in (D4, 4, 5, D8, a.t_max - 1)}
    curves, store = {}, {}
    for name in ('u', 'mean', 'raw', 'kmode', 'katt'):
        curves[name] = {}
        for al in alphas:
            rr = steer_run(dirs[name], al, W58)
            store[(name, al)] = rr
            curves[name][al] = {'fit': boot_seq(tot(rr, fit_i, D8), rng, 1)[0], 'eval': boot_seq(tot(rr, ev_i, D8), rng, 1)[0]}
        print(f'[odir2] {name:5s} depth-8 NLL by alpha (held-out): ' + ' '.join(f'{al}:{c["eval"]:.4f}' for al, c in curves[name].items())
              + f"   none {boot_seq(base_ev[D8], rng, 1)[0]:.4f}", flush=True)
    best = {n_: min(alphas, key=lambda al: curves[n_][al]['fit']) for n_ in curves}
    res['dose_response'] = {n_: {str(al): c for al, c in cv.items()} for n_, cv in curves.items()}; res['alpha_chosen'] = best
    ev_of = lambda n_: tot(store[(n_, best[n_])], ev_i, D8)
    diff = lambda x, y: [(p[0] - q[0], p[1]) for p, q in zip(x, y)]
    U8 = ev_of('u')
    res['steer_vs_none'] = {n_: boot_seq(diff(ev_of(n_), base_ev[D8]), rng) for n_ in curves}
    # nulls at u's alpha, held-out texts only
    al_u = best['u']; u_eval = boot_seq(U8, rng, 1)[0]
    null_r = [boot_seq(tot(steer_run(v, al_u, W58, texts_ev), range(len(ev_i)), D8), rng, 1)[0] for v in rand]
    null_s = [boot_seq(tot(steer_run(v, al_u, W58, texts_ev), range(len(ev_i)), D8), rng, 1)[0] for v in shuf]
    res['nulls'] = {'u_eval': u_eval, 'random': null_r, 'shuffled': null_s,
                    'u_better_than_random_frac': float(np.mean([u_eval < x for x in null_r])),
                    'u_better_than_shuffled_frac': float(np.mean([u_eval < x for x in null_s]))}
    res['N1'] = bool(res['nulls']['u_better_than_random_frac'] >= 0.95 and res['nulls']['u_better_than_shuffled_frac'] >= 0.95)
    res['N2_detail'] = boot_seq(diff(U8, ev_of('mean')), rng); res['N2'] = bool(res['N2_detail'][2] < 0)
    res['N3_detail'] = boot_seq(diff(U8, ev_of('raw')), rng); res['N3'] = bool(res['N3_detail'][2] < 0)
    print(f"[odir2] u beats {res['nulls']['u_better_than_random_frac']:.0%} of random, {res['nulls']['u_better_than_shuffled_frac']:.0%} of shuffled; "
          f"u - mean {res['N2_detail']}; u - raw {res['N3_detail']}", flush=True)
    kp = []
    for n_ in ('kmode', 'katt'):
        ke = boot_seq(ev_of(n_), rng, 1)[0]
        kp.append(res['steer_vs_none'][n_][2] < 0 and np.mean([ke < x for x in null_r]) >= 0.95)
    res['KP1'] = bool(max(abs(cosu['kmode']), abs(cosu['katt'])) >= 0.5); res['KP2'] = bool(any(kp))
    # U1: useful extra loops
    u1 = {}
    for win, d in ((W5, 4), (W56, 5)):
        sel = {al: steer_run(u, al, win) for al in alphas}
        al_b = min(alphas, key=lambda al: boot_seq(tot(sel[al], fit_i, d), rng, 1)[0])
        u1[f'depth{d + 1}'] = {'alpha': al_b, 'steered': boot_seq(tot(sel[al_b], ev_i, d), rng),
                               'unsteered_same_depth': boot_seq(base_ev[d], rng), 'unsteered_depth4': boot_seq(base_ev[D4], rng),
                               'diff_vs_depth4': boot_seq(diff(tot(sel[al_b], ev_i, d), base_ev[D4]), rng)}
    res['U1_detail'] = u1; res['U1'] = bool(any(v['diff_vs_depth4'][2] < 0 for v in u1.values()))
    print('[odir2] U1 ' + '  '.join(f"{k}: steered {v['steered'][0]:.4f} vs depth-4 {v['unsteered_depth4'][0]:.4f} (unsteered same depth {v['unsteered_same_depth'][0]:.4f})" for k, v in u1.items()), flush=True)
    # G1o: other text
    ood = None
    try:
        if a.ood:
            ood = [json.loads(l)['text'] for l in open(a.ood)]
        else:
            import datasets
            ood = [t for t in datasets.load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')['text'] if len(t) > 600]
    except Exception as e:
        print('[odir2] no other-text data:', str(e)[:120])
    if ood:
        ood = ood[: (16 if a.smoke else 256)]
        tx = []
        for t in ood:
            ii = tok(t, add_special_tokens=False)['input_ids']
            tx.append((tok.decode(ii[:48]), tok.decode(ii[48:300])))
        b0 = tot(forward(tx), range(len(tx)), D8); bu = tot(steer_run(u, al_u, W58, tx), range(len(tx)), D8)
        br = [tot(steer_run(v, al_u, W58, tx), range(len(tx)), D8) for v in rand[:10]]
        brm = [(float(np.mean([r_[k][0] for r_ in br])), b0[k][1]) for k in range(len(tx))]
        res['G1o_detail'] = {'n': len(tx), 'nll8_none': boot_seq(b0, rng, 1)[0], 'u_vs_none': boot_seq(diff(bu, b0), rng),
                             'u_vs_random': boot_seq(diff(bu, brm), rng)}
        res['G1o'] = bool(res['G1o_detail']['u_vs_none'][2] < 0 and res['G1o_detail']['u_vs_random'][2] < 0)
        print(f"[odir2] other text: u vs none {res['G1o_detail']['u_vs_none']}  vs random {res['G1o_detail']['u_vs_random']}", flush=True)
    # U2: answers (reloads the model at 4 / 6 / 8 loops)
    if not a.no_gen:
        try:
            tr = [json.loads(l) for l in open(a.train)][:3]
            shot = ''.join(f"Question: {r['question']}\nAnswer: {' '.join(r['steps'])}\n### {r['answer']}\n\n" for r in tr)
            qs = [json.loads(l) for l in open(a.data)][-a.gen_n:]
            from rdiag.lcot.train import parse, correct
            tok.padding_side = 'left'
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            al56 = u1['depth6']['alpha']; acc = {}
            del model; torch.cuda.empty_cache()
            for T, cond in ((4, None), (6, None), (6, W56), (8, None), (8, W58)):
                model = load_model(T); norm, head = hooks(model); oks = []
                for i in range(0, len(qs), 16):
                    B = qs[i:i + 16]
                    enc = tok([shot + f"Question: {r['question']}\nAnswer:" for r in B], return_tensors='pt', padding=True).to(dev)
                    STEER[0] = None if cond is None else (u.to(dev), (al56 if T == 6 else al_u) * g, set(cond))
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
                    STEER[0] = None
                    for r, o in zip(B, out[:, enc['input_ids'].shape[1]:]):
                        txt = tok.decode(o, skip_special_tokens=True).split('Question:')[0]
                        oks.append(float(correct(parse(txt), r['answer'])))
                acc[f'T{T}' + ('_steer' if cond else '')] = np.array(oks)
                print(f"[odir2] GSM8k {T} loops{' steered' if cond else ''}: acc {np.mean(oks):.3f}", flush=True)
                del model; torch.cuda.empty_cache()
            bd = lambda x: [float(x.mean())] + [float(v) for v in np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(3000)], [2.5, 97.5])]
            res['U2_detail'] = {k: float(v.mean()) for k, v in acc.items()}
            res['U2_detail']['T6_steer_minus_T6'] = bd(acc['T6_steer'] - acc['T6'])
            res['U2_detail']['T8_steer_minus_T8'] = bd(acc['T8_steer'] - acc['T8'])
            res['U2'] = bool(res['U2_detail']['T6_steer_minus_T6'][1] > 0)
        except Exception as e:
            print('[odir2] generation failed:', repr(e)[:300])
    crit = [k for k in ('N1', 'N2', 'N3', 'U1', 'U2', 'KP1', 'KP2', 'A1', 'A2', 'OH2', 'G1o') if k in res]
    print(' '.join(f"{k} {'PASS' if res[k] else 'FAIL'}" for k in crit))
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/odir2{('_' + a.tag) if a.tag else ''}.json", 'w'), indent=1, default=float)

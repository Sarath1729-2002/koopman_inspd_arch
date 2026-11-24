"""Ouro-1.4B looped past its trained depth (4): is the drift a DIRECTION of damage, does steering against it help,
and is drift a usable STOPPING signal? (Nov 24; follows rdiag/ouro.py, where corr(drift size, damage) reached +0.56)
Manifold = PCA(256) of depth 1..4 states on the fit half (as in ouro.py). For a token state h at depth d:
  residual  r = ((h - mu) - P(h - mu)) / ||h||          (off-manifold part; drift size = ||r||^2)
  damage    = NLL(depth d) - NLL(depth 4) of the next gold-solution token
  direction u = mean(r | damage top quartile) - mean(r | bottom quartile) at depth 8 on the FIT half, normalised;
            gap g = projection gap between the two groups
Steering: at loops 5..8 the last layer's output is replaced by h - alpha * g * ||h|| * u (alpha in {0.5, 1, 2},
  chosen on the fit half), scored on the held-out half; control = 3 random unit directions, same alpha and size.
Stopping: per token, start at depth 4 and keep looping while drift(d+1) <= tau x floor (tau chosen on the fit half
  from {1.0, 1.1, ..., 2.0}). Compared with a fixed 4 loops, every other fixed depth, and the per-token oracle depth.
Criteria (fixed before the run), all on the held-out half, bootstrap over sequences:
  OD1 corr(direction score, damage) at depth 8 exceeds corr(drift size, damage): CI of the difference > 0
  OD2 steering lowers depth-8 NLL vs no steering (CI < 0) AND vs the random-direction control (CI < 0)
  OH1 drift-based stopping beats a fixed 4 loops on mean gold-token NLL (CI < 0)
Ouro's own learned exit gate is NOT compared here (it needs the gate's outputs; left for later).
Writes results_rdiag/odir.json"""
import argparse, json, os, numpy as np, torch
from rdiag.ouro import find_parts


def boot_seq(per_seq, rng, B=3000):
    """per_seq: list of (sum, count) per sequence -> mean over tokens, CI by resampling sequences"""
    s = np.array([x for x, _ in per_seq]); c = np.array([y for _, y in per_seq])
    m = s.sum() / c.sum(); bs = []
    for _ in range(B):
        ix = rng.integers(0, len(s), len(s)); bs.append(s[ix].sum() / c[ix].sum())
    return [float(m)] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ByteDance/Ouro-1.4B'); ap.add_argument('--t_train', type=int, default=4)
    ap.add_argument('--t_max', type=int, default=12); ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--k', type=int, default=256); ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--steer_until', type=int, default=8)
    ap.add_argument('--data', default='data_lcot/test.jsonl'); ap.add_argument('--out', default='results_rdiag')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; rng = np.random.default_rng(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); cfg.total_ut_steps = a.t_max
    if hasattr(cfg, 'early_exit_threshold'):
        cfg.early_exit_threshold = 1.0
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    inner, layers, norm, head = find_parts(model)
    states, STEER = [], [None]   # STEER[0] = (u [hidden] cuda float, scale) or None

    def post(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        d = len(states) + 1                       # loop index of this output (1-based)
        if STEER[0] is not None and a.t_train < d <= a.steer_until:
            u, sc = STEER[0]
            hf = h.float(); hf = hf - sc * hf.norm(dim=-1, keepdim=True) * u
            h = hf.to(h.dtype)
            o = (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        states.append(h.detach())
        return o
    inner.register_forward_pre_hook(lambda *x, **k: states.clear())
    layers[-1].register_forward_hook(post)
    rows = [json.loads(l) for l in open(a.data)][: (16 if a.smoke else a.n)]
    texts = [(r['question'] + '\n', '\n'.join(r['steps']) + '\n### ' + r['answer']) for r in rows]
    half = len(texts) // 2

    @torch.no_grad()
    def forward(keep_states):
        """per sequence: list over depth of (state [n_tok, H] cpu float or None, nll [n_tok])"""
        recs = []
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
                raise SystemExit(f'hook fired {len(states)} times, expected {a.t_max}')
            for j, sq in enumerate(seqs):
                lo, hi = len(q[j]), len(sq)
                if hi - lo < 2:
                    recs.append(None); continue
                tgt = ids[j, lo:hi]; per = []
                for d in range(a.t_max):
                    lp = torch.log_softmax(head(norm(states[d][j:j + 1, lo - 1:hi - 1])).float(), -1)[0]
                    per.append((states[d][j, lo - 1:hi - 1].float().cpu() if keep_states else None,
                                (-lp.gather(-1, tgt[:, None])[:, 0]).cpu()))
                recs.append(per)
        return recs

    recs = forward(True)
    fit_i = [i for i in range(half) if recs[i] is not None]; ev_i = [i for i in range(half, len(recs)) if recs[i] is not None]
    X = torch.cat([recs[i][d][0][::4] for i in fit_i for d in range(a.t_train)]).double()
    mu = X.mean(0); V = torch.linalg.svd(X - mu, full_matrices=False)[2][:a.k].T
    del X

    def resid(h):
        hc = h.double() - mu
        return ((hc - hc @ V @ V.T) / h.double().norm(dim=-1, keepdim=True)).float()
    D8 = min(8, a.t_max) - 1; D4 = a.t_train - 1
    floor = float(torch.cat([resid(recs[i][d][0]).pow(2).sum(-1) for i in ev_i for d in range(a.t_train)]).mean())
    # per sequence / depth: drift size, direction feature, nll
    def feats(idx):
        out = []
        for i in idx:
            per = recs[i]
            R8 = resid(per[D8][0]); dm = per[D8][1] - per[D4][1]
            dr = torch.stack([resid(per[d][0]).pow(2).sum(-1) for d in range(a.t_max)])     # [T, n_tok]
            nl = torch.stack([per[d][1] for d in range(a.t_max)])                            # [T, n_tok]
            out.append((R8, dm, dr, nl))
        return out
    Ff, Fe = feats(fit_i), feats(ev_i)
    R8 = torch.cat([x[0] for x in Ff]); dm = torch.cat([x[1] for x in Ff])
    qh, ql = torch.quantile(dm, 0.75), torch.quantile(dm, 0.25)
    dvec = R8[dm >= qh].mean(0) - R8[dm <= ql].mean(0); u = dvec / dvec.norm()
    g = float((R8[dm >= qh] @ u).mean() - (R8[dm <= ql] @ u).mean())
    for i in range(len(recs)):     # free the states
        if recs[i] is not None:
            recs[i] = [(None, nl) for _, nl in recs[i]]
    # OD1: correlation on held-out sequences, bootstrap over sequences
    def corr_pair(ix):
        sc = torch.cat([Fe[k][0] @ u for k in ix]).numpy(); sz = torch.cat([Fe[k][2][D8] for k in ix]).numpy()
        dd = torch.cat([Fe[k][1] for k in ix]).numpy()
        return np.corrcoef(sc, dd)[0, 1], np.corrcoef(sz, dd)[0, 1]
    c_dir, c_size = corr_pair(range(len(Fe)))
    bs = [np.subtract(*corr_pair(rng.integers(0, len(Fe), len(Fe)))) for _ in range(1000)]
    res = {'model': a.model, 'floor': floor, 'gap': g, 'n_fit': len(fit_i), 'n_eval': len(ev_i),
           'OD1_detail': {'corr_dir': float(c_dir), 'corr_size': float(c_size),
                          'diff': [float(c_dir - c_size)] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]}}
    print(f"[odir] depth 8 held-out corr with damage: direction {c_dir:+.3f}  drift size {c_size:+.3f}  diff {res['OD1_detail']['diff']}", flush=True)
    # OH1: stopping rule
    def stop_nll(F_, tau):
        per_seq = []
        for (_, _, dr, nl) in F_:
            T, n = dr.shape
            dsel = torch.full((n,), D4, dtype=torch.long)
            go = torch.ones(n, dtype=torch.bool)
            for d in range(D4 + 1, T):               # continue past 4 while the next depth stays near the manifold
                go = go & (dr[d] <= tau * floor); dsel = torch.where(go, torch.full_like(dsel, d), dsel)
            v = nl.gather(0, dsel[None])[0]
            per_seq.append((float(v.sum()), n))
        return per_seq
    taus = [round(1.0 + 0.1 * j, 1) for j in range(11)]
    fixed = lambda F_, d: [(float(x[3][d].sum()), x[3].shape[1]) for x in F_]
    tau = min(taus, key=lambda t: boot_seq(stop_nll(Ff, t), rng, 1)[0])
    st, fx = stop_nll(Fe, tau), fixed(Fe, D4)
    diff = [(s[0] - f[0], s[1]) for s, f in zip(st, fx)]
    orc = [(float(x[3].min(0).values.sum()), x[3].shape[1]) for x in Fe]
    res['OH1_detail'] = {'tau': tau, 'nll_stop': boot_seq(st, rng), 'nll_fixed4': boot_seq(fx, rng),
                         'nll_oracle': boot_seq(orc, rng), 'diff_stop_minus_fixed4': boot_seq(diff, rng),
                         'nll_fixed': {d + 1: boot_seq(fixed(Fe, d), rng, 1)[0] for d in range(a.t_max)}}
    print(f"[odir] stopping (tau {tau}): NLL {res['OH1_detail']['nll_stop'][0]:.4f} vs fixed-4 {res['OH1_detail']['nll_fixed4'][0]:.4f} "
          f"vs oracle {res['OH1_detail']['nll_oracle'][0]:.4f}  diff {res['OH1_detail']['diff_stop_minus_fixed4']}", flush=True)
    # OD2: steering
    alphas = (1.0,) if a.smoke else (0.5, 1.0, 2.0)
    rand = [torch.randn(u.shape[0], generator=torch.Generator().manual_seed(s)) for s in range(3)]
    rand = [(r / r.norm()).to(dev) for r in rand]
    ud = u.to(dev)
    base = [(float(recs[i][D8][1].sum()), len(recs[i][D8][1])) if recs[i] is not None else None for i in range(len(recs))]
    def nll8(steer):
        STEER[0] = steer; rr = forward(False); STEER[0] = None
        return [(float(r[D8][1].sum()), len(r[D8][1]), float(r[-1][1].sum())) if r is not None else None for r in rr]
    out_s, out_c = {}, {}
    for al in alphas:
        out_s[al] = nll8((ud, al * g))
        cs = [nll8((rv, al * g)) for rv in rand]
        out_c[al] = [None if x is None else (np.mean([c[k][0] for c in cs]), x[1], np.mean([c[k][2] for c in cs]))
                     for k, x in enumerate(cs[0])]
        mf = lambda L, idx: sum(L[i][0] for i in idx) / sum(L[i][1] for i in idx)
        print(f'[odir] steer alpha {al}: depth-8 NLL fit/held-out {mf(out_s[al], fit_i):.4f}/{mf(out_s[al], ev_i):.4f}  '
              f'random {mf(out_c[al], ev_i):.4f}  none {mf(base, ev_i):.4f}', flush=True)
    al = min(alphas, key=lambda x: sum(out_s[x][i][0] for i in fit_i) / sum(out_s[x][i][1] for i in fit_i))
    d_none = [(out_s[al][i][0] - base[i][0], base[i][1]) for i in ev_i]
    d_rand = [(out_s[al][i][0] - out_c[al][i][0], base[i][1]) for i in ev_i]
    res['OD2_detail'] = {'alpha': al, 'nll8_steered': boot_seq([out_s[al][i][:2] for i in ev_i], rng),
                         'nll8_none': boot_seq([base[i] for i in ev_i], rng),
                         'nll8_random': boot_seq([out_c[al][i][:2] for i in ev_i], rng),
                         'diff_vs_none': boot_seq(d_none, rng), 'diff_vs_random': boot_seq(d_rand, rng),
                         'nll12_steered_vs_none': float(sum(out_s[al][i][2] for i in ev_i) / sum(base[i][1] for i in ev_i)
                                                        - sum(Fe[k][3][-1].sum() for k in range(len(Fe))) / sum(base[i][1] for i in ev_i))}
    print(f"[odir] steering (alpha {al}): diff vs none {res['OD2_detail']['diff_vs_none']}  vs random {res['OD2_detail']['diff_vs_random']}", flush=True)
    res['OD1'] = bool(res['OD1_detail']['diff'][1] > 0)
    res['OD2'] = bool(res['OD2_detail']['diff_vs_none'][2] < 0 and res['OD2_detail']['diff_vs_random'][2] < 0)
    res['OH1'] = bool(res['OH1_detail']['diff_stop_minus_fixed4'][2] < 0)
    print(' '.join(f"{k} {'PASS' if res[k] else 'FAIL'}" for k in ('OD1', 'OD2', 'OH1')))
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f'{a.out}/odir.json', 'w'), indent=1, default=float)

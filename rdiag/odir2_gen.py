"""U2 for odir2 (GSM8k answer accuracy with steering), rerun with a hand-written greedy loop: Ouro's
UniversalTransformerCache does not work with this transformers version's generate() ("key_cache has no setter").
Recomputes only what U2 needs (same seeds and splits as odir2.py): the unsteered pass, the damage direction u (residual)
and the raw-state direction, and each one's alpha for windows {5,6} and {5..8} chosen on the fit half by NLL at depth
6 / 8. Then greedy decoding (3-shot, last 200 test questions, no KV cache) at 4 / 6 / 8 loops, +- steering.
U2 (pre-registered in odir2.py): 6 loops steered with u beats 6 loops unsteered, paired CI > 0.
Also reported: raw-direction steering at 6 and 8 loops, u at 8 loops, steered 6 vs unsteered 4.
Merges U2 into results_rdiag/odir2[_tag].json (writes odir2_gen[_tag].json too)."""
import argparse, json, os, numpy as np, torch
from rdiag.ouro import find_parts
from rdiag.odir import boot_seq

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ByteDance/Ouro-1.4B'); ap.add_argument('--t_train', type=int, default=4)
    ap.add_argument('--t_max', type=int, default=12); ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--k', type=int, default=256); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--data', default='data_lcot/test.jsonl'); ap.add_argument('--train', default='data_lcot/train.jsonl')
    ap.add_argument('--gen_n', type=int, default=200); ap.add_argument('--max_new', type=int, default=160)
    ap.add_argument('--out', default='results_rdiag'); ap.add_argument('--tag', default='')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from rdiag.lcot.train import parse, correct
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; rng = np.random.default_rng(0); torch.manual_seed(0)
    if a.smoke:
        a.n, a.gen_n, a.max_new = 24, 4, 6
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (getattr(tok, 'eos_token_id', None) or 0)

    def load_model(T):
        cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); cfg.total_ut_steps = T
        if hasattr(cfg, 'early_exit_threshold'):
            cfg.early_exit_threshold = 1.0
        return AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    states, STEER = [], [None]

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
    model = load_model(a.t_max); norm, head = hooks(model)
    rows = [json.loads(l) for l in open(a.data)][:a.n]
    texts = [(r['question'] + '\n', '\n'.join(r['steps']) + '\n### ' + r['answer']) for r in rows]
    half = len(texts) // 2

    @torch.no_grad()
    def forward(tx, keep=False, steer=None, depths=(3, 5, 7)):
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
                for d in (range(a.t_max) if keep else depths):
                    lp = torch.log_softmax(head(norm(states[d][j:j + 1, lo - 1:hi - 1])).float()[0], -1)
                    r['nll'][d] = (-lp.gather(-1, tgt[:, None])[:, 0]).cpu()
                if keep:
                    r['h'] = {d: states[d][j, lo - 1:hi - 1].float().cpu() for d in (0, 1, 2, 3, 7)}
                recs.append(r)
        STEER[0] = None
        return recs
    base = forward(texts, keep=True)
    ok_i = [i for i, r in enumerate(base) if r is not None]
    fit_i = [i for i in ok_i if i < half]
    X = torch.cat([base[i]['h'][d][::4] for i in fit_i for d in range(a.t_train)]).double()
    mu = X.mean(0); V = torch.linalg.svd(X - mu, full_matrices=False)[2][:a.k].T.contiguous(); del X
    D4, D8 = a.t_train - 1, 7

    def resid(h):
        hc = h.double() - mu
        return ((hc - hc @ V @ V.T) / h.double().norm(dim=-1, keepdim=True)).float()

    def damage_dir(F_, dm):
        qh, ql = torch.quantile(dm, 0.75), torch.quantile(dm, 0.25)
        d = F_[dm >= qh].mean(0) - F_[dm <= ql].mean(0); v = d / d.norm()
        return v, float((F_[dm >= qh] @ v).mean() - (F_[dm <= ql] @ v).mean())
    dm = torch.cat([base[i]['nll'][D8] - base[i]['nll'][D4] for i in fit_i])
    u, g = damage_dir(torch.cat([resid(base[i]['h'][D8]) for i in fit_i]), dm)
    w_raw, _ = damage_dir(torch.cat([base[i]['h'][D8] / base[i]['h'][D8].norm(dim=-1, keepdim=True) for i in fit_i]), dm)
    for i in ok_i:
        base[i].pop('h')
    alphas = (1.0,) if a.smoke else (0.25, 0.5, 1.0, 2.0, 4.0)
    W56, W58 = {5, 6}, {5, 6, 7, 8}
    tot = lambda recs, d: boot_seq([(float(recs[i]['nll'][d].sum()), len(recs[i]['nll'][d])) for i in fit_i], rng, 1)[0]
    pick = {}
    for name, v in (('u', u), ('raw', w_raw)):
        for wn, win, d in (('56', W56, 5), ('58', W58, D8)):
            pick[(name, wn)] = min(alphas, key=lambda al: tot(forward(texts, steer=(v.to(dev), al * g, win)), d))
    print('[odir2_gen] alphas (fit half): ' + '  '.join(f'{k[0]}/{k[1]}: {v}' for k, v in pick.items()), flush=True)
    del model; torch.cuda.empty_cache()
    tr = [json.loads(l) for l in open(a.train)][:3]
    shot = ''.join(f"Question: {r['question']}\nAnswer: {' '.join(r['steps'])}\n### {r['answer']}\n\n" for r in tr)
    qs = [json.loads(l) for l in open(a.data)][-a.gen_n:]

    @torch.no_grad()
    def greedy(prompts, steer):
        """right-padded batch, no cache: full recompute each step (positions stay correct for every sequence)"""
        P = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]
        Lm = max(map(len, P)) + a.max_new
        buf = torch.full((len(P), Lm), pad, dtype=torch.long, device=dev); msk = torch.zeros_like(buf)
        for j, p in enumerate(P):
            buf[j, :len(p)] = torch.tensor(p); msk[j, :len(p)] = 1
        cur = torch.tensor([len(p) for p in P], device=dev); start = cur.clone(); live = torch.ones(len(P), dtype=torch.bool, device=dev)
        for step in range(a.max_new):
            L = int(cur.max())
            STEER[0] = steer
            lg = model(input_ids=buf[:, :L], attention_mask=msk[:, :L], use_cache=False).logits
            STEER[0] = None
            nx = lg[torch.arange(len(P), device=dev), cur - 1].argmax(-1)
            ix = torch.nonzero(live)[:, 0]
            buf[ix, cur[ix]] = nx[ix]; msk[ix, cur[ix]] = 1; cur[ix] += 1
            if getattr(tok, 'eos_token_id', None) is not None:
                live &= nx != tok.eos_token_id
            if step % 8 == 7:
                for j in ix.tolist():
                    if 'Question:' in tok.decode(buf[j, start[j]:cur[j]]):
                        live[j] = False
            if not live.any():
                break
        return [tok.decode(buf[j, start[j]:cur[j]], skip_special_tokens=True).split('Question:')[0] for j in range(len(P))]
    acc = {}
    for T, cond in ((4, None), (6, None), (6, ('u', '56')), (6, ('raw', '56')), (8, None), (8, ('u', '58')), (8, ('raw', '58'))):
        model = load_model(T); norm, head = hooks(model); oks = []
        st = None if cond is None else ((u if cond[0] == 'u' else w_raw).to(dev), pick[cond] * g, W56 if cond[1] == '56' else W58)
        for i in range(0, len(qs), a.bs):
            B = qs[i:i + a.bs]
            outs = greedy([shot + f"Question: {r['question']}\nAnswer:" for r in B], st)
            oks += [float(correct(parse(o), r['answer'])) for o, r in zip(outs, B)]
        key = f'T{T}' + ('' if cond is None else f'_{cond[0]}')
        acc[key] = np.array(oks)
        print(f'[odir2_gen] GSM8k {key}: acc {np.mean(oks):.3f}', flush=True)
        del model; torch.cuda.empty_cache()
    bd = lambda x: [float(x.mean())] + [float(v) for v in np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(3000)], [2.5, 97.5])]
    det = {k: float(v.mean()) for k, v in acc.items()}
    for k1, k2 in (('T6_u', 'T6'), ('T6_raw', 'T6'), ('T8_u', 'T8'), ('T8_raw', 'T8'), ('T6_u', 'T4'), ('T6_raw', 'T4')):
        det[f'{k1}_minus_{k2}'] = bd(acc[k1] - acc[k2])
    det['alphas'] = {f'{k[0]}_{k[1]}': v for k, v in pick.items()}
    out = {'U2_detail': det, 'U2': bool(det['T6_u_minus_T6'][1] > 0)}
    print('U2', 'PASS' if out['U2'] else 'FAIL', {k: v for k, v in det.items() if 'minus' in k})
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    sfx = ('_' + a.tag) if a.tag else ''
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(f'{a.out}/odir2_gen{sfx}.json', 'w'), indent=1)
    p = f'{a.out}/odir2{sfx}.json'
    if os.path.exists(p):
        r = json.load(open(p)); r.update(out); json.dump(r, open(p, 'w'), indent=1)

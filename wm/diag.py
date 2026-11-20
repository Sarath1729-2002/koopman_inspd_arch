"""Mechanism diagnostics: WHY does course correction (fail to) help?  Open-loop H steps, no correction, and at
every step measure what a projection onto the encoder manifold WOULD do.
Per step t (MSE in standardized obs units, averaged over windows):
  err        prediction error of the uncorrected rollout
  shift_full ||decode(reencode_full(o_hat)) - o_hat||^2   how much a full re-encode moves the prediction
  proj_full  error of that re-encoded prediction vs truth  (<err => projecting helps this step)
  shift_z / proj_z  same for the code snap (RSSM only; equals full for KAE)
  resid      relative state residual ||h' - h|| / ||h||   (the event-trigger signal)
  rec_true   decode(encode(o_true)) error                  (quality of the projection operator itself)
Oracle rollouts (upper bounds on any course correction that uses the encoder):
  oracle_init10  every 10 steps rebuild the state from the TRUE observation (h <- init(enc(o)))
  oracle_post10  every 10 steps keep h and update z with the posterior on the TRUE observation (RSSM filtering)
Also correlation between resid and future error growth (does the trigger signal predict trouble?)."""
import argparse, json, os, glob, numpy as np, torch
from wm.train import load, build, windows, CTX
from wm.kae import KAE


@torch.no_grad()
def run(model, o, a, H):
    kae = isinstance(model, KAE)
    hs, zs, *_ = model.observe(o, a, n_ctx=CTX, mode=True)
    h, z = hs[:, -1], zs[:, -1]
    hi, zi, hp, zp = h.clone(), z.clone(), h.clone(), z.clone()
    rec = {k: [] for k in ('err', 'shift_full', 'proj_full', 'shift_z', 'proj_z', 'resid', 'rec_true',
                           'oracle_init10', 'oracle_post10')}
    for t in range(H):
        at, ot = a[:, CTX + t], o[:, CTX + 1 + t]
        # uncorrected rollout
        h = model.step(h, z, at)
        z = z if kae else model.sample(model.dist(model.prior(h)), mode=True)
        oh = model.decode(h, z)
        e = model.enc(oh)
        hf, zf = model.encode_state(e, mode=True)
        of = model.decode(hf, zf)
        if kae:
            oz = of
        else:
            zz = model.sample(model.dist(model.post(torch.cat([h, e], -1))), mode=True)
            oz = model.decode(h, zz)
        et = model.enc(ot)
        ht, ztr = model.encode_state(et, mode=True)
        mse = lambda x, y: (x - y).pow(2).mean(-1)
        rec['err'].append(mse(oh, ot)); rec['shift_full'].append(mse(of, oh)); rec['proj_full'].append(mse(of, ot))
        rec['shift_z'].append(mse(oz, oh)); rec['proj_z'].append(mse(oz, ot))
        rec['resid'].append((hf - h).norm(dim=-1) / (h.norm(dim=-1) + 1e-6))
        rec['rec_true'].append(mse(model.decode(ht, ztr), ot))
        # oracle: rebuild from true obs every 10 steps
        hi = model.step(hi, zi, at)
        zi = zi if kae else model.sample(model.dist(model.prior(hi)), mode=True)
        rec['oracle_init10'].append(mse(model.decode(hi, zi), ot))
        if (t + 1) % 10 == 0:
            hi, zi = ht, ztr
        # oracle: posterior update on true obs every 10 steps (keeps h)
        hp = model.step(hp, zp, at)
        if kae:
            zp = zp
        elif (t + 1) % 10 == 0:
            zp = model.sample(model.dist(model.post(torch.cat([hp, et], -1))), mode=True)
        else:
            zp = model.sample(model.dist(model.prior(hp)), mode=True)
        rec['oracle_post10'].append(mse(model.decode(hp, zp), ot) if not kae else rec['oracle_init10'][-1])
    return {k: torch.stack(v, 1) for k, v in rec.items()}  # [B, H]


def summarize(R, H):
    bands = {'1-10': slice(0, 10), '26-50': slice(25, 50), '101-end': slice(100, H)} if H > 100 else \
            {'1-10': slice(0, 10), '26-50': slice(25, 50)}
    out = {}
    for k, v in R.items():
        out[k] = {b: float(v[:, sl].mean()) for b, sl in bands.items()}
    # does the trigger signal predict trouble?  corr(resid_t, err_{t+5} - err_t) across windows and steps
    r, e = R['resid'][:, :-5].flatten(), (R['err'][:, 5:] - R['err'][:, :-5]).flatten()
    out['corr_resid_future_err_growth'] = float(torch.corrcoef(torch.stack([r, e]))[0, 1])
    r2 = R['resid'].flatten(); e2 = R['err'].flatten()
    out['corr_resid_err'] = float(torch.corrcoef(torch.stack([r2, e2]))[0, 1])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--horizon', type=int, default=180); ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    ap.add_argument('--out', default='results_wm')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, ote, ate = load(a.env, a.data)
    o, ac = windows(ote, ate, a.n, np.random.default_rng(11), CTX + a.horizon)
    o, ac = torch.from_numpy(o).to(dev), torch.from_numpy(ac).to(dev)
    f_out = f'{a.out}/diag_{a.env}.json'
    res = json.load(open(f_out)) if os.path.exists(f_out) else {}
    for f in sorted(glob.glob(f'{a.ckpt}/{a.env}_*_s{a.seed}.pt')):
        ck = torch.load(f, map_location=dev)
        if ck['arm'] in res or ck['arm'] == 'koop_cont':
            continue
        model = build(ck['arm'], ck['obs_dim'], ck['act_dim']).to(dev); model.load_state_dict(ck['state']); model.eval()
        S = summarize(run(model, o, ac, a.horizon), a.horizon)
        res[ck['arm']] = S
        b = '101-end' if a.horizon > 100 else '26-50'
        print(f"[{a.env}] {ck['arm']:12s} {b}: err {S['err'][b]:.4f}  proj_full {S['proj_full'][b]:.4f}  "
              f"proj_z {S['proj_z'][b]:.4f}  shift_full {S['shift_full'][b]:.4f}  rec_true {S['rec_true'][b]:.4f}  "
              f"oracle_init10 {S['oracle_init10'][b]:.4f}  oracle_post10 {S['oracle_post10'][b]:.4f}  "
              f"corr(resid, future growth) {S['corr_resid_future_err_growth']:+.2f}", flush=True)
        json.dump(res, open(f_out + '.tmp', 'w'), indent=1); os.replace(f_out + '.tmp', f_out)
    print('done', a.env)

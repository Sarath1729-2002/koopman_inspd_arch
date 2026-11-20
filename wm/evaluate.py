"""Open-loop 50-step prediction with/without course correction (decode -> re-encode).
Writes results_wm/eval_<env>.json  (per arm x schedule: mse per horizon step, #re-encodes)."""
import argparse, json, os, glob, numpy as np, torch
from wm.train import load, build, windows, CTX, L

H = L - CTX  # 50


@torch.no_grad()
def rollout(model, o, a, sched, bs=500, kind='full'):
    mses, nre = [], []
    for i in range(0, o.shape[0], bs):
        oi, ai = o[i:i + bs], a[i:i + bs]
        hs, zs, *_ = model.observe(oi, ai, n_ctx=CTX, mode=True)
        preds, n = model.imagine(hs[:, -1], zs[:, -1], ai[:, CTX:], sched, kind)
        mses.append((preds - oi[:, CTX + 1:]).pow(2).mean(-1)); nre.append(n)
    return torch.cat(mses), torch.cat(nre)


def periodic(P):
    return lambda t, r: torch.full_like(r, (t + 1) % P == 0, dtype=torch.bool) if (t + 1) < H else None


def event(thr):
    return lambda t, r: (r > thr) if (t + 1) < H else None


def calibrate(model, o, a, target, lo=1e-3, hi=10.0, iters=14, kind='full'):
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        n = rollout(model, o, a, event(mid), kind=kind)[1].mean().item()
        if n > target: lo = mid
        else: hi = mid
    return float(np.sqrt(lo * hi))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_test', type=int, default=2000); ap.add_argument('--n_val', type=int, default=400)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    ap.add_argument('--out', default='results_wm')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    otr, atr, ote, ate = load(a.env, a.data)
    o, ac = windows(ote, ate, a.n_test, np.random.default_rng(5))
    ov, av = windows(otr, atr, a.n_val, np.random.default_rng(7))
    T = lambda x: torch.from_numpy(x).to(dev)
    o, ac, ov, av = T(o), T(ac), T(ov), T(av)
    res = {}
    for f in sorted(glob.glob(f'{a.ckpt}/{a.env}_*_s{a.seed}.pt')):
        ck = torch.load(f, map_location=dev)
        model = build(ck['arm'], ck['obs_dim'], ck['act_dim']).to(dev); model.load_state_dict(ck['state']); model.eval()
        torch.manual_seed(0)
        scheds = {'none': (None, 'full')}
        for kind, pre in (('full', ''), ('z', 'z')):
            for P in (25, 10, 5, 2, 1):
                scheds[f'{pre}per{P}'] = (periodic(P), kind)
            for k in (2, 5, 10):
                thr = calibrate(model, ov, av, k, kind=kind)
                scheds[f'{pre}evt{k}'] = (event(thr), kind)
        for name, (sc, kind) in scheds.items():
            m, n = rollout(model, o, ac, sc, kind=kind)
            mh = m.mean(0).cpu().numpy()
            key = f"{ck['arm']}/{name}"
            res[key] = {'mse_h': mh.tolist(), 'n_reenc': n.mean().item(),
                        'b1_10': float(mh[:10].mean()), 'b11_25': float(mh[10:25].mean()), 'b26_50': float(mh[25:].mean()),
                        'all': float(mh.mean())}
            print(f'[{a.env}] {key:22s} reenc {n.mean().item():5.1f}  mse 1-10 {mh[:10].mean():.4f}  '
                  f'11-25 {mh[10:25].mean():.4f}  26-50 {mh[25:].mean():.4f}', flush=True)
        json.dump(res, open(f'{a.out}/eval_{a.env}.json', 'w'), indent=1)
    print('done', a.env)

"""Open-loop prediction with/without course correction (decode -> re-encode).
--horizon H (default 50 = training rollout length; up to 185 with T=200 episodes).
Schedules: none; full re-encode periodic perP / event evtK; code-snap (keep h, snap z) zperP / zevtK.
Event thresholds are calibrated on training windows to K re-encodes per 50 imagined steps (scaled to H).
Writes <out>/eval_<env>[_<tag>].json"""
import argparse, json, os, glob, numpy as np, torch
from wm.train import load, build, windows, CTX


@torch.no_grad()
def rollout(model, o, a, sched, bs=500, kind='full'):
    mses, nre = [], []
    for i in range(0, o.shape[0], bs):
        oi, ai = o[i:i + bs], a[i:i + bs]
        hs, zs, *_ = model.observe(oi, ai, n_ctx=CTX, mode=True)
        preds, n = model.imagine(hs[:, -1], zs[:, -1], ai[:, CTX:], sched, kind)
        mses.append((preds - oi[:, CTX + 1:]).pow(2).mean(-1)); nre.append(n)
    return torch.cat(mses), torch.cat(nre)


def periodic(P, H):
    return lambda t, r: torch.full_like(r, (t + 1) % P == 0, dtype=torch.bool) if (t + 1) < H else None


def event(thr, H):
    return lambda t, r: (r > thr) if (t + 1) < H else None


def calibrate(model, o, a, target, H, kind, lo=1e-4, hi=10.0, iters=16):
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        n = rollout(model, o, a, event(mid, H), kind=kind)[1].mean().item()
        if n > target: lo = mid
        else: hi = mid
    return float(np.sqrt(lo * hi))


def bands(mh):
    b = {'b1_10': mh[:10].mean(), 'b11_25': mh[10:25].mean(), 'b26_50': mh[25:50].mean(), 'all': mh.mean()}
    if len(mh) > 50:
        b['b51_100'] = mh[50:100].mean()
        b['b101_end'] = mh[100:].mean() if len(mh) > 100 else float('nan')
    return {k: float(v) for k, v in b.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--horizon', type=int, default=50); ap.add_argument('--tag', default='')
    ap.add_argument('--arms', default='', help='comma list; default all checkpoints for this env')
    ap.add_argument('--n_test', type=int, default=2000); ap.add_argument('--n_val', type=int, default=400)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    ap.add_argument('--out', default='results_wm')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    H = a.horizon; Lw = CTX + H
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    otr, atr, ote, ate = load(a.env, a.data)
    o, ac = windows(ote, ate, a.n_test, np.random.default_rng(5), Lw)
    ov, av = windows(otr, atr, a.n_val, np.random.default_rng(7), Lw)
    T = lambda x: torch.from_numpy(x).to(dev)
    o, ac, ov, av = T(o), T(ac), T(ov), T(av)
    out = f"{a.out}/eval_{a.env}{'_' + a.tag if a.tag else ''}.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    want = set(a.arms.split(',')) if a.arms else None
    scale = H / 50
    for f in sorted(glob.glob(f'{a.ckpt}/{a.env}_*_s{a.seed}.pt')):
        ck = torch.load(f, map_location=dev)
        if want and ck['arm'] not in want:
            continue
        if any(k.startswith(ck['arm'] + '/') for k in res):
            print('done already', ck['arm']); continue
        model = build(ck['arm'], ck['obs_dim'], ck['act_dim']).to(dev); model.load_state_dict(ck['state']); model.eval()
        torch.manual_seed(0)
        scheds = {'none': (None, 'full')}
        for kind, pre in (('full', ''), ('z', 'z')):
            for P in (25, 10, 5, 2, 1):
                scheds[f'{pre}per{P}'] = (periodic(P, H), kind)
            for k in (2, 5, 10):
                thr = calibrate(model, ov, av, k * scale, H, kind)
                scheds[f'{pre}evt{k}'] = (event(thr, H), kind)
        for name, (sc, kind) in scheds.items():
            m, n = rollout(model, o, ac, sc, kind=kind)
            mh = m.mean(0).cpu().numpy()
            key = f"{ck['arm']}/{name}"
            res[key] = {'mse_h': mh.tolist(), 'n_reenc': n.mean().item(), **bands(mh)}
            b = res[key]
            tail = f"  51-100 {b['b51_100']:.4f}  101-{H} {b['b101_end']:.4f}" if H > 50 else ''
            print(f'[{a.env}] {key:22s} reenc {n.mean().item():5.1f}  1-10 {b["b1_10"]:.4f}  '
                  f'11-25 {b["b11_25"]:.4f}  26-50 {b["b26_50"]:.4f}{tail}', flush=True)
        json.dump(res, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
    print('done', a.env)

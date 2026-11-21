"""SHORT-DATA regime evaluation: models trained only on 16-step segments, deployed open-loop for 180 steps.
Per env and family X in {kae, gru, koop}, seeds 0-2:
  X_s10  : none, per10, per5, res17 (residual trigger, ~17 re-encodes), dens17 (density trigger, ~17 re-encodes)
  X_s10n : none, per10                       (noise-injection baseline, available with short data)
  ensemble of the 3 X_s10 seeds (mean prediction): none, per10
Density trigger = Mahalanobis distance of the imagined latent from the distribution of latents seen in training
(the latent leaving the region the dynamics were trained on, which the decoder hides).
Also: corr(trigger score_t, error growth over the next 5 steps) for residual vs density.
Writes results_wm/short_<env>.json"""
import argparse, json, os, numpy as np, torch
from wm.train import load, build, windows, seg_windows, CTX
from wm.kae import KAE

H = 180


def load_m(ckpt, env, arm, seed, dev):
    f = f'{ckpt}/{env}_{arm}_s{seed}.pt'
    if not os.path.exists(f):
        return None
    ck = torch.load(f, map_location=dev)
    m = build(ck['arm'], ck['obs_dim'], ck['act_dim']).to(dev); m.load_state_dict(ck['state']); m.eval()
    return m


@torch.no_grad()
def latent_stats(m, otr, atr, dev, n=2000):
    o, a = seg_windows(otr, atr, n, np.random.default_rng(3))
    o, a = torch.from_numpy(o).to(dev), torch.from_numpy(a).to(dev)
    hs, *_ = m.observe(o, a, mode=True)
    Hh = hs[:, 4:].reshape(-1, hs.shape[-1]).double()
    mu = Hh.mean(0)
    C = torch.cov((Hh - mu).T) + 1e-3 * torch.eye(Hh.shape[1], device=dev, dtype=torch.float64)
    P = torch.linalg.inv(C).float()
    mu = mu.float()
    return lambda h: ((h - mu) @ P * (h - mu)).sum(-1).clamp(min=0).sqrt()


@torch.no_grad()
def roll(m, o, a, fn=None, score=None, bs=500):
    P, N = [], []
    for i in range(0, o.shape[0], bs):
        oi, ai = o[i:i + bs], a[i:i + bs]
        hs, zs, *_ = m.observe(oi, ai, n_ctx=CTX, mode=True)
        p, n = m.imagine(hs[:, -1], zs[:, -1], ai[:, CTX:], fn, 'full', score)
        P.append(p); N.append(n)
    return torch.cat(P), torch.cat(N)


@torch.no_grad()
def trace(m, o, a, score):
    """Uncorrected rollout; per-step error and trigger scores (residual and density)."""
    hs, zs, *_ = m.observe(o, a, n_ctx=CTX, mode=True)
    h, z = hs[:, -1], zs[:, -1]
    kae = isinstance(m, KAE)
    E, R, D = [], [], []
    for t in range(H):
        h = m.step(h, z, a[:, CTX + t])
        z = z if kae else m.sample(m.dist(m.prior(h)), mode=True)
        oh = m.decode(h, z)
        hn, _ = m.encode_state(m.enc(oh), mode=True)
        E.append((oh - o[:, CTX + 1 + t]).pow(2).mean(-1))
        R.append((hn - h).norm(dim=-1) / (h.norm(dim=-1) + 1e-6)); D.append(score(h))
    E, R, D = torch.stack(E, 1), torch.stack(R, 1), torch.stack(D, 1)
    g = (E[:, 5:] - E[:, :-5]).flatten()
    c = lambda x: float(torch.corrcoef(torch.stack([x[:, :-5].flatten(), g]))[0, 1])
    c2 = lambda x: float(torch.corrcoef(torch.stack([x.flatten(), E.flatten()]))[0, 1])
    return {'corr_resid_growth': c(R), 'corr_dens_growth': c(D), 'corr_resid_err': c2(R), 'corr_dens_err': c2(D)}


def per(P):
    return lambda t, r: torch.full_like(r, (t + 1) % P == 0, dtype=torch.bool) if (t + 1) < H else None


def evt(thr):
    return lambda t, r: (r > thr) if (t + 1) < H else None


def calib(m, o, a, target, score=None, lo=1e-4, hi=1e4):
    for _ in range(24):
        mid = float(np.sqrt(lo * hi))
        n = roll(m, o, a, evt(mid), score)[1].mean().item()
        lo, hi = (mid, hi) if n > target else (lo, mid)
    return float(np.sqrt(lo * hi))


def bands(err):
    mh = err.mean(0).cpu().numpy()
    return {'b26_50': float(mh[25:50].mean()), 'b101_end': float(mh[100:].mean()), 'mse_h': mh.tolist()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True); ap.add_argument('--families', default='kae,gru,koop')
    ap.add_argument('--n_test', type=int, default=1500); ap.add_argument('--n_val', type=int, default=300)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    ap.add_argument('--out', default='results_wm')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    otr, atr, ote, ate = load(a.env, a.data)
    T = lambda x: torch.from_numpy(x).to(dev)
    o, ac = map(T, windows(ote, ate, a.n_test, np.random.default_rng(5), CTX + H))
    ov, av = map(T, windows(otr, atr, a.n_val, np.random.default_rng(7), CTX + H))
    f_out = f'{a.out}/short_{a.env}.json'
    res = json.load(open(f_out)) if os.path.exists(f_out) else {}
    save = lambda: (json.dump(res, open(f_out + '.tmp', 'w'), indent=1), os.replace(f_out + '.tmp', f_out))
    mse = lambda p: (p - o[:, CTX + 1:]).pow(2).mean(-1)
    for X in a.families.split(','):
        base, noisy = f'{X}_s10', f'{X}_s10n'
        ens_preds = {'none': [], 'per10': []}
        for seed in (0, 1, 2):
            m = load_m(a.ckpt, a.env, base, seed, dev)
            if m is None:
                print('missing', a.env, base, seed); continue
            dens = latent_stats(m, otr, atr, dev)
            key = f'{base}/s{seed}'
            if key + '/none' not in res:
                for name, fn, sc in (('none', None, None), ('per10', per(10), None), ('per5', per(5), None)):
                    p, n = roll(m, o, ac, fn, sc)
                    res[f'{key}/{name}'] = {**bands(mse(p)), 'n_reenc': n.mean().item()}
                thr_r = calib(m, ov, av, 17.0)
                thr_d = calib(m, ov, av, 17.0, dens)
                for name, fn, sc in (('res17', evt(thr_r), None), ('dens17', evt(thr_d), dens)):
                    p, n = roll(m, o, ac, fn, sc)
                    res[f'{key}/{name}'] = {**bands(mse(p)), 'n_reenc': n.mean().item()}
                res[f'{key}/trace'] = trace(m, o[:500], ac[:500], dens)
                save()
            for name, fn in (('none', None), ('per10', per(10))):
                ens_preds[name].append(roll(m, o, ac, fn)[0])
            r = res
            print(f"[{a.env}] {key:14s} none {r[key + '/none']['b101_end']:.4f}  per10 {r[key + '/per10']['b101_end']:.4f}  "
                  f"per5 {r[key + '/per5']['b101_end']:.4f}  res17 {r[key + '/res17']['b101_end']:.4f} "
                  f"({r[key + '/res17']['n_reenc']:.0f})  dens17 {r[key + '/dens17']['b101_end']:.4f} "
                  f"({r[key + '/dens17']['n_reenc']:.0f})  corr growth: resid {r[key + '/trace']['corr_resid_growth']:+.2f} "
                  f"dens {r[key + '/trace']['corr_dens_growth']:+.2f}", flush=True)
            mn = load_m(a.ckpt, a.env, noisy, seed, dev)
            if mn is not None and f'{noisy}/s{seed}/none' not in res:
                for name, fn in (('none', None), ('per10', per(10))):
                    p, n = roll(mn, o, ac, fn)
                    res[f'{noisy}/s{seed}/{name}'] = {**bands(mse(p)), 'n_reenc': n.mean().item()}
                save()
            del m
        for name, ps in ens_preds.items():
            if len(ps) >= 2:
                res[f'{base}/ens/{name}'] = bands(mse(torch.stack(ps).mean(0)))
        save()
    print('done', a.env)

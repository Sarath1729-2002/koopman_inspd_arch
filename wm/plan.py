"""Model-predictive control (CEM) in the REAL HalfCheetah with a learned world model, with/without course correction
inside the planner's imagined rollouts.  Hypothesis: a planner actively searches for action sequences whose imagined
outcome is good; it exploits model errors (drives the latent off the data manifold). Projecting back onto the encoder
manifold could remove exploitable directions, which a training-time fix cannot anticipate.
Metrics per (model, correction): real return over the episode, and optimism gap = mean over steps of
(predicted return of the chosen plan over Hp steps) - (realized return over the next Hp steps)."""
import argparse, json, os, numpy as np, torch
from wm.train import build, CTX
from wm.kae import KAE

XV = 8  # HalfCheetah-v5 obs index of root x-velocity (qvel[0]) with current x excluded


class Planner:
    def __init__(s, model, mean, std, dev, Hp=25, pop=400, elites=40, iters=4, corr='none', thr=None):
        s.m, s.dev, s.Hp, s.pop, s.el, s.it, s.corr, s.thr = model, dev, Hp, pop, elites, iters, corr, thr
        s.mean, s.std = torch.tensor(mean, device=dev), torch.tensor(std, device=dev)
        s.kae = isinstance(model, KAE)

    def fn(s):
        if s.corr == 'none':
            return None, 'full'
        if s.corr == 'per10':
            return (lambda t, r: torch.full_like(r, (t + 1) % 10 == 0, dtype=torch.bool)), 'full'
        if s.corr == 'zper5':
            return (lambda t, r: torch.full_like(r, (t + 1) % 5 == 0, dtype=torch.bool)), ('full' if s.kae else 'z')
        if s.corr == 'evt':
            return (lambda t, r: r > s.thr), 'full'
        raise ValueError(s.corr)

    @torch.no_grad()
    def returns(s, h, z, A):
        fn, kind = s.fn()
        preds, _ = s.m.imagine(h.expand(A.shape[0], -1), z.expand(A.shape[0], -1), A, fn, kind)
        xv = preds[..., XV] * s.std[XV] + s.mean[XV]
        return (xv - 0.1 * A.pow(2).sum(-1)).sum(1)

    @torch.no_grad()
    def act(s, obs_hist, act_hist, prev):
        o = torch.tensor(obs_hist, device=s.dev).unsqueeze(0)
        a = torch.tensor(act_hist, device=s.dev).unsqueeze(0)
        hs, zs, *_ = s.m.observe(o, a, n_ctx=o.shape[1] - 1, mode=True)
        h, z = hs[:, -1], zs[:, -1]
        A_dim = a.shape[-1]
        mu = torch.zeros(s.Hp, A_dim, device=s.dev)
        if prev is not None:
            mu[:-1] = prev[1:]
        sd = torch.full_like(mu, 0.5)
        for _ in range(s.it):
            A = (mu + sd * torch.randn(s.pop, s.Hp, A_dim, device=s.dev)).clamp(-1, 1)
            R = s.returns(h, z, A)
            idx = R.topk(s.el).indices
            E = A[idx]
            mu, sd = E.mean(0), E.std(0) + 0.05
        pred = s.returns(h, z, mu.unsqueeze(0))[0].item()
        return mu, pred


def episode(planner, mean, std, seed, T=200):
    import gymnasium as gym
    env = gym.make('HalfCheetah-v5', max_episode_steps=10 ** 6)
    o, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    nz = lambda x: np.clip((x - mean) / std, -10, 10).astype(np.float32)
    obs_hist, act_hist = [nz(o)], []
    for _ in range(CTX):  # warm-up with random actions to fill the filter context
        u = rng.uniform(-1, 1, env.action_space.shape[0]).astype(np.float32)
        o, *_ = env.step(u); act_hist.append(u); obs_hist.append(nz(o))
    rewards, preds, prev = [], [], None
    for t in range(T):
        mu, pred = planner.act(np.stack(obs_hist[-(CTX + 1):]), np.stack(act_hist[-CTX:]), prev)
        u = mu[0].cpu().numpy().astype(np.float32)
        o, r, *_ = env.step(u)
        rewards.append(float(r)); preds.append(pred); prev = mu
        obs_hist.append(nz(o)); act_hist.append(u)
    Hp = planner.Hp
    real_h = [sum(rewards[t:t + Hp]) for t in range(T - Hp)]
    gap = float(np.mean([preds[t] - real_h[t] for t in range(T - Hp)]))
    return float(np.sum(rewards)), gap


def calibrate_thr(model, dev, data, Hp, target=3.0):
    """Event threshold giving ~target re-encodes per Hp-step imagined rollout on training windows."""
    from wm.train import load, windows
    from wm.evaluate import rollout, event
    otr, atr, _, _ = load('cheetah', data)
    o, a = windows(otr, atr, 400, np.random.default_rng(7), CTX + Hp)
    o, a = torch.from_numpy(o).to(dev), torch.from_numpy(a).to(dev)
    lo, hi = 1e-4, 10.0
    for _ in range(16):
        mid = float(np.sqrt(lo * hi))
        n = rollout(model, o, a, event(mid, Hp), kind='full')[1].mean().item()
        lo, hi = (mid, hi) if n > target else (lo, mid)
    return float(np.sqrt(lo * hi))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='gru,gru_rdr,koop,koop_rdr,gru_rdr10,kae_k10,kae_k50')
    ap.add_argument('--corrs', default='none,per10,zper5,evt')
    ap.add_argument('--episodes', type=int, default=4); ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    ap.add_argument('--out', default='results_wm')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(f'{a.data}/cheetah.npz'); mean, std = d['mean'], d['std']
    f_out = f'{a.out}/plan_cheetah.json'
    res = json.load(open(f_out)) if os.path.exists(f_out) else {}
    # reference: random-action return
    if 'random' not in res:
        import gymnasium as gym
        rs = []
        for ep in range(a.episodes):
            env = gym.make('HalfCheetah-v5', max_episode_steps=10 ** 6); env.reset(seed=100 + ep)
            rs.append(sum(float(env.step(env.action_space.sample())[1]) for _ in range(a.T)))
        res['random'] = {'returns': rs, 'mean': float(np.mean(rs))}
        print('random policy return', np.mean(rs), flush=True)
    for arm in a.arms.split(','):
        f = f'{a.ckpt}/cheetah_{arm}_s0.pt'
        if not os.path.exists(f):
            print('missing', f); continue
        ck = torch.load(f, map_location=dev)
        model = build(ck['arm'], ck['obs_dim'], ck['act_dim']).to(dev); model.load_state_dict(ck['state']); model.eval()
        thr = None
        for corr in a.corrs.split(','):
            key = f'{arm}/{corr}'
            if key in res:
                continue
            if corr == 'evt' and thr is None:
                thr = calibrate_thr(model, dev, a.data, 25)
            pl = Planner(model, mean, std, dev, corr=corr, thr=thr)
            torch.manual_seed(0)
            out = [episode(pl, mean, std, seed=100 + ep, T=a.T) for ep in range(a.episodes)]
            R = [x[0] for x in out]; G = [x[1] for x in out]
            res[key] = {'returns': R, 'gaps': G, 'mean': float(np.mean(R)), 'se': float(np.std(R) / np.sqrt(len(R))),
                        'gap': float(np.mean(G))}
            print(f'[plan] {key:18s} return {np.mean(R):8.1f} +- {np.std(R) / np.sqrt(len(R)):6.1f}   '
                  f'optimism gap {np.mean(G):8.2f}', flush=True)
            json.dump(res, open(f_out + '.tmp', 'w'), indent=1); os.replace(f_out + '.tmp', f_out)
    print('done plan')

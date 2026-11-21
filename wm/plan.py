"""Model-predictive control (CEM) in the REAL MuJoCo env with a learned world model, with/without course correction
inside the planner's imagined rollouts.  Hypothesis: the planner exploits latent drift that the decoder hides;
periodic decode->re-encode removes it.  Baselines that also fight exploitation: shorter planning horizon (hp10),
and a 3-seed ensemble with a disagreement penalty (PETS-style, mean - LAMBDA * std of imagined returns).
Metrics: real return over T model steps, optimism gap = mean_t [predicted Hp-return of the chosen plan
- realized return over the next Hp steps].
Keys in results: '<arm>/s<seed>/<corr>' with corr in {none, per10, per5, hp10}; '<arm>/ens/<none|per10>'."""
import argparse, json, os, numpy as np, torch
from wm.train import build, CTX
from wm.data import ENVS

LAMBDA = 1.0
# obs layout (Gymnasium v5, current x excluded): (z index, angle index, x-velocity index, healthy z range, |angle| max)
SPEC = {'cheetah': dict(xv=8, ctrl=0.1, healthy=None),
        'hopper': dict(xv=5, z=0, ang=1, zr=(0.7, 1e9), amax=0.2, ctrl=1e-3, hb=1.0),
        'walker': dict(xv=8, z=0, ang=1, zr=(0.8, 2.0), amax=1.0, ctrl=1e-3, hb=1.0)}


def soft(x, k=0.02):
    return torch.sigmoid(x / k)


class Planner:
    def __init__(s, models, env, mean, std, dev, Hp=25, pop=400, elites=40, iters=4, corr='none'):
        s.ms, s.env, s.dev, s.Hp, s.pop, s.el, s.it, s.corr = models, env, dev, Hp, pop, elites, iters, corr
        s.mean, s.std = torch.tensor(mean, device=dev), torch.tensor(std, device=dev)
        s.sp, s.rep = SPEC[env], ENVS[env][1]

    def fn(s):
        if s.corr in ('none', 'hp10'):
            return None
        P = {'per10': 10, 'per5': 5}[s.corr]
        return lambda t, r: torch.full_like(r, (t + 1) % P == 0, dtype=torch.bool)

    def reward(s, preds, A):
        o = preds * s.std + s.mean
        sp = s.sp
        r = o[..., sp['xv']] - sp['ctrl'] * A.pow(2).sum(-1)
        if sp.get('hb'):
            z, ang = o[..., sp['z']], o[..., sp['ang']]
            h = soft(z - sp['zr'][0]) * soft(sp['zr'][1] - z) * soft(sp['amax'] - ang.abs())
            r = r + sp['hb'] * h
        return s.rep * r  # one model step = rep env steps

    @torch.no_grad()
    def returns(s, states, A):
        Rs = []
        for m, (h, z) in zip(s.ms, states):
            preds, _ = m.imagine(h.expand(A.shape[0], -1), z.expand(A.shape[0], -1), A, s.fn(), 'full')
            Rs.append(s.reward(preds, A).sum(1))
        R = torch.stack(Rs)
        return R.mean(0) - (LAMBDA * R.std(0) if len(s.ms) > 1 else 0.0)

    @torch.no_grad()
    def act(s, obs_hist, act_hist, prev):
        o = torch.tensor(obs_hist, device=s.dev).unsqueeze(0)
        a = torch.tensor(act_hist, device=s.dev).unsqueeze(0)
        states = []
        for m in s.ms:
            hs, zs, *_ = m.observe(o, a, n_ctx=o.shape[1] - 1, mode=True)
            states.append((hs[:, -1], zs[:, -1]))
        Ad = a.shape[-1]
        mu = torch.zeros(s.Hp, Ad, device=s.dev)
        if prev is not None:
            n = min(s.Hp, prev.shape[0]) - 1
            mu[:n] = prev[1:n + 1]
        sd = torch.full_like(mu, 0.5)
        for _ in range(s.it):
            A = (mu + sd * torch.randn(s.pop, s.Hp, Ad, device=s.dev)).clamp(-1, 1)
            idx = s.returns(states, A).topk(s.el).indices
            E = A[idx]
            mu, sd = E.mean(0), E.std(0) + 0.05
        return mu, s.returns(states, mu.unsqueeze(0))[0].item()


def make_env(env):
    import gymnasium as gym
    name, rep = ENVS[env]
    kw = {} if env == 'cheetah' else dict(terminate_when_unhealthy=False)
    return gym.make(name, max_episode_steps=10 ** 7, **kw), rep


def episode(planner, env_key, mean, std, seed, T=200):
    env, rep = make_env(env_key)
    o, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    nz = lambda x: np.clip((x - mean) / std, -10, 10).astype(np.float32)
    A = env.action_space.shape[0]
    obs_hist, act_hist = [nz(o)], []
    for _ in range(CTX):  # warm-up with random actions to fill the filter context
        u = rng.uniform(-1, 1, A).astype(np.float32)
        for _ in range(rep):
            o, *_ = env.step(u)
        act_hist.append(u); obs_hist.append(nz(o))
    rewards, preds, prev = [], [], None
    for t in range(T):
        mu, pred = planner.act(np.stack(obs_hist[-(CTX + 1):]), np.stack(act_hist[-CTX:]), prev)
        u = mu[0].cpu().numpy().astype(np.float32)
        r = 0.0
        for _ in range(rep):
            o, ri, *_ = env.step(u); r += float(ri)
        rewards.append(r); preds.append(pred); prev = mu
        obs_hist.append(nz(o)); act_hist.append(u)
    Hp = planner.Hp
    gap = float(np.mean([preds[t] - sum(rewards[t:t + Hp]) for t in range(T - Hp)]))
    return float(np.sum(rewards)), gap


def load_model(ckpt, env, arm, seed, dev):
    f = f'{ckpt}/{env}_{arm}_s{seed}.pt'
    if not os.path.exists(f):
        return None
    ck = torch.load(f, map_location=dev)
    m = build(ck['arm'], ck['obs_dim'], ck['act_dim']).to(dev); m.load_state_dict(ck['state']); m.eval()
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='cheetah')
    ap.add_argument('--arms', default='gru,gru_rdr,koop,koop_rdr,gru_rdr10,kae_k10,kae_k50')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--corrs', default='none,per10')
    ap.add_argument('--ensemble', action='store_true', help='also plan with the seed ensemble (none, per10)')
    ap.add_argument('--episodes', type=int, default=4); ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    ap.add_argument('--out', default='results_wm'); ap.add_argument('--fname', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(f'{a.data}/{a.env}.npz'); mean, std = d['mean'], d['std']
    f_out = f"{a.out}/{a.fname or 'plan_' + a.env}.json"
    res = json.load(open(f_out)) if os.path.exists(f_out) else {}
    save = lambda: (json.dump(res, open(f_out + '.tmp', 'w'), indent=1), os.replace(f_out + '.tmp', f_out))
    seeds = [int(x) for x in a.seeds.split(',')]

    def run(key, models, corr, Hp=25):
        if key in res:
            return
        pl = Planner(models, a.env, mean, std, dev, Hp=Hp, corr=corr)
        torch.manual_seed(0)
        out = [episode(pl, a.env, mean, std, seed=100 + ep, T=a.T) for ep in range(a.episodes)]
        R, G = [x[0] for x in out], [x[1] for x in out]
        res[key] = {'returns': R, 'gaps': G, 'mean': float(np.mean(R)), 'se': float(np.std(R) / np.sqrt(len(R))),
                    'gap': float(np.mean(G))}
        print(f'[plan {a.env}] {key:24s} return {np.mean(R):8.1f} +- {np.std(R) / np.sqrt(len(R)):6.1f}   '
              f'optimism gap {np.mean(G):8.2f}', flush=True)
        save()

    if 'random' not in res:
        rs = []
        for ep in range(a.episodes):
            env, rep = make_env(a.env); env.reset(seed=100 + ep); tot = 0.0
            for _ in range(a.T):
                u = env.action_space.sample()
                for _ in range(rep):
                    tot += float(env.step(u)[1])
            rs.append(tot)
        res['random'] = {'returns': rs, 'mean': float(np.mean(rs))}
        print(f'[plan {a.env}] random policy return {np.mean(rs):.1f}', flush=True); save()
    for arm in a.arms.split(','):
        ms = {s: load_model(a.ckpt, a.env, arm, s, dev) for s in seeds}
        for s, m in ms.items():
            if m is None:
                print('missing', a.env, arm, s); continue
            for corr in a.corrs.split(','):
                run(f'{arm}/s{s}/{corr}', [m], corr, Hp=10 if corr == 'hp10' else 25)
        if a.ensemble:
            mm = [m for m in ms.values() if m is not None]
            if len(mm) >= 2:
                for corr in ('none', 'per10'):
                    run(f'{arm}/ens/{corr}', mm, corr)
    print('done plan', a.env)

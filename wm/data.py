"""Collect MuJoCo proprioceptive trajectories (Gymnasium v5) with temporally-correlated random actions.
Saves data_wm/<env>.npz with obs_train [N,T+1,O], act_train [N,T,A], obs_test, act_test, mean, std."""
import argparse, os, numpy as np
from multiprocessing import Pool

ENVS = {'cheetah': ('HalfCheetah-v5', 1), 'walker': ('Walker2d-v5', 4), 'hopper': ('Hopper-v5', 4)}


def collect(job):
    env_key, seed, n_ep, T = job
    import gymnasium as gym
    name, rep = ENVS[env_key]
    kw = {} if env_key == 'cheetah' else dict(terminate_when_unhealthy=False)
    e = gym.make(name, max_episode_steps=10 ** 6, **kw)
    rng = np.random.default_rng(seed)
    A = e.action_space.shape[0]
    O, Ac = [], []
    for _ in range(n_ep):
        o, _ = e.reset(seed=int(rng.integers(1 << 30)))
        alpha = rng.uniform(0.0, 0.9)          # action smoothness varies per episode
        a = rng.uniform(-1, 1, A)
        os_, as_ = [o], []
        for t in range(T):
            a = alpha * a + (1 - alpha) * rng.uniform(-1, 1, A)
            as_.append(a.copy())
            for _ in range(rep):
                o, *_ = e.step(a)
            os_.append(o)
        O.append(np.asarray(os_, np.float32)); Ac.append(np.asarray(as_, np.float32))
    return np.stack(O), np.stack(Ac)


def gather(env_key, seed0, n_ep, T, procs):
    per = int(np.ceil(n_ep / procs))
    with Pool(procs) as p:
        res = p.map(collect, [(env_key, seed0 + i, per, T) for i in range(procs)])
    return np.concatenate([r[0] for r in res])[:n_ep], np.concatenate([r[1] for r in res])[:n_ep]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', default='cheetah,walker,hopper')
    ap.add_argument('--n_train', type=int, default=2000)
    ap.add_argument('--n_test', type=int, default=200)
    ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--out', default='data_wm')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for k in a.envs.split(','):
        f = f'{a.out}/{k}.npz'
        if os.path.exists(f):
            print('exists', f); continue
        otr, atr = gather(k, 1000, a.n_train, a.T, a.procs)
        ote, ate = gather(k, 900000, a.n_test, a.T, a.procs)
        mean, std = otr.reshape(-1, otr.shape[-1]).mean(0), otr.reshape(-1, otr.shape[-1]).std(0) + 1e-3
        np.savez_compressed(f + '.tmp.npz', obs_train=otr, act_train=atr, obs_test=ote, act_test=ate, mean=mean, std=std)
        os.replace(f + '.tmp.npz', f)
        print(k, otr.shape, atr.shape, ote.shape, flush=True)

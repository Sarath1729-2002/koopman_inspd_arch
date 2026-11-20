"""Train one world model.  --arm {gru, koop, koop_cont, gru_rdr, koop_rdr, gru_rdrre, koop_rdrre}
*_rdr adds Rollout-Decoded Reconstruction (free-run prior from the context state, decode, MSE to truth).
*_rdrre = RDR with course correction IN the training rollout: after each imagined step, with prob P_RE the state is
rebuilt from the decoded observation (h <- init(enc(o_hat)), z <- posterior), gradients flowing through decode/encode,
so the dynamics learn to continue from re-encoded states."""
import argparse, os, time, numpy as np, torch
from wm.model import RSSM

ARMS = {'gru': ('gru', 'disc', False), 'koop': ('koop', 'disc', False), 'koop_cont': ('koop', 'cont', False),
        'gru_rdr': ('gru', 'disc', True), 'koop_rdr': ('koop', 'disc', True),
        'gru_rdrre': ('gru', 'disc', 're'), 'koop_rdrre': ('koop', 'disc', 're')}
P_RE = 0.1
L, CTX = 64, 14  # window: obs[0..64], act[0..63]; filter t<=14, imagine 50 steps


def load(env, root='data_wm'):
    d = np.load(f'{root}/{env}.npz')
    m, s = d['mean'], d['std']
    nz = lambda o: np.clip((o - m) / s, -10, 10).astype(np.float32)
    return nz(d['obs_train']), d['act_train'], nz(d['obs_test']), d['act_test']


def windows(obs, act, B, rng, Lw=L):
    i = rng.integers(0, obs.shape[0], B)
    t = rng.integers(0, obs.shape[1] - Lw - 1 + 1, B)
    ar = np.arange(Lw + 1)
    return obs[i[:, None], t[:, None] + ar], act[i[:, None], t[:, None] + ar[:-1]]


def build(arm, obs_dim, act_dim):
    trans, lat = ARMS[arm][:2]
    return RSSM(obs_dim, act_dim, trans=trans, latent=lat)


def losses(model, o, a, rdr):
    hs, zs, dyn, rep, cons = model.observe(o, a)
    rec = (model.decode(hs, zs) - o).pow(2).mean()
    kl = 0.5 * dyn.clamp(min=1.0).mean() + 0.1 * rep.clamp(min=1.0).mean()
    loss = rec + kl + cons.mean()
    out = {'rec': rec.item(), 'kl': dyn.mean().item(), 'cons': cons.mean().item()}
    if rdr:
        h, z = hs[:, CTX], zs[:, CTX]
        fn = (lambda t, r: torch.rand_like(r) < P_RE) if rdr == 're' else None
        preds, _ = model.imagine(h, z, a[:, CTX:], fn, 'full')
        r = (preds - o[:, CTX + 1:]).pow(2).mean()
        loss = loss + r
        out['rdr'] = r.item()
    return loss, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True); ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--batch', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--log', type=int, default=1000)
    ap.add_argument('--data', default='data_wm'); ap.add_argument('--ckpt', default='ckpt_wm')
    a = ap.parse_args()
    os.makedirs(a.ckpt, exist_ok=True)
    f = f'{a.ckpt}/{a.env}_{a.arm}_s{a.seed}.pt'
    if os.path.exists(f):
        print('exists', f); raise SystemExit
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    otr, atr, _, _ = load(a.env, a.data)
    model = build(a.arm, otr.shape[-1], atr.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 500))
    t0 = time.time(); tag = f'[{a.env} {a.arm} s{a.seed}]'
    for it in range(a.steps):
        o, ac = windows(otr, atr, a.batch, rng)
        o, ac = torch.from_numpy(o).to(dev), torch.from_numpy(ac).to(dev)
        loss, info = losses(model, o, ac, ARMS[a.arm][2])
        if not torch.isfinite(loss):
            print(tag, 'non-finite loss at', it, flush=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0); opt.step(); sched.step()
        if it % a.log == 0 or it == a.steps - 1:
            print(tag, f'step {it}/{a.steps}', ' '.join(f'{k} {v:.4f}' for k, v in info.items()),
                  f'({time.time() - t0:.0f}s)', flush=True)
    torch.save({'arm': a.arm, 'state': model.state_dict(), 'obs_dim': otr.shape[-1], 'act_dim': atr.shape[-1]}, f + '.tmp')
    os.replace(f + '.tmp', f)
    print(tag, f'saved ({(time.time() - t0) / 60:.1f} min)', flush=True)

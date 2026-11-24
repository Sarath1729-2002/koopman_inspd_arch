"""KILL TEST: course correction (decode -> re-encode) in JEPA-style latent world models (Meta's jepa-wms release).

Models: DINO-WM (predictor trained teacher-forced, 1-step) and JEPA-WM (2-step rollout training), frozen DINOv2-S
patch features (16x16x384), Push-T / Wall / PointMaze, frameskip 5. Open-loop rollouts from 1 real context frame on
VALIDATION trajectories with the logged actions; compare predicted features with the encoder's features of the true
future frames.

Arms (hook applied to the newest predicted frame before it enters the predictor's context):
  none       plain open-loop unroll (as in their planner)
  reencP     every P steps: z <- E(D(z))  with their released image decoder head (Fathi-style projection)
  noise1     control: z <- z + eps, |eps| = |E(D(z)) - z| per sample, random direction
  oracle4    upper bound: every 4 steps replace z with the TRUE encoding of that frame (how much headroom exists)
Per step and sample we record: latent error ||z - z_true||^2/||z_true||^2 (visual), drift signal ||E(D(z)) - z||^2/||z||^2,
and the drift floor on real frames ||E(D(z_true)) - z_true||^2/||z_true||^2.
Writes results_jwm/<model>.json.  Run from repo root:  python -m jwm.drift --model dino_wm_pusht"""
import argparse, json, os, sys, time, numpy as np, torch

JW = os.path.abspath(os.environ.get('JEPAWM_DIR', 'ext/jepa-wms'))


def load_stubs():
    sys.path.insert(0, JW)
    try:   # h5py is only needed by the RoboCasa loader; on this cluster it can clash with opencv's HDF5 build
        import h5py  # noqa: F401
    except Exception as e:
        import types
        print(f'h5py unavailable ({type(e).__name__}); stubbed (not used for Push-T / Wall / PointMaze)', flush=True)
        sys.modules['h5py'] = types.ModuleType('h5py')
    try:   # clusterscope only knows Meta clusters; datasets resolve through $JEPAWM_DSET instead
        import clusterscope
        clusterscope.cluster = lambda *a, **k: 'default'
    except Exception:
        pass


def load(model_name, dev):
    load_stubs()
    import yaml
    import hubconf
    from evals.utils import make_datasets
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
    from src.utils.yaml_utils import expand_env_vars
    cfg_rel, wkey = hubconf._MODEL_CONFIGS[model_name]
    a = expand_env_vars(yaml.safe_load(open(os.path.join(JW, cfg_rel))))
    mk = a['model_kwargs']
    cwd = os.getcwd(); os.chdir(JW)
    dset, pre = make_datasets(mk['data'], mk['data_aug'], filter_first_episodes=10 ** 6)
    os.chdir(cwd)
    wm = init_module(folder=None, checkpoint=hubconf._get_checkpoint_path(wkey), model_kwargs=mk['pretrain_kwargs'],
                     wrapper_kwargs=mk.get('wrapper_kwargs', {}), cfgs_data=mk['data'], device=dev,
                     action_dim=dset.action_dim, proprio_dim=dset.proprio_dim, preprocessor=pre)
    wm.eval()
    fs = mk['data']['custom'].get('frameskip', 1)
    rs = mk['pretrain_kwargs'].get('rollout_cfg', {}).get('rollout_steps', 1)
    return wm, dset, pre, fs, rs


def episodes(dset, pre, fs, T, n, rng):
    """(visual uint8-range float [N, T+1, C, H, W], raw proprio [N, T+1, P], normalized actions [N, T, fs*A])"""
    L = []
    for i in range(len(dset)):
        try:
            L.append(dset.get_seq_length(i))
        except Exception:
            L.append(len(dset[i][1]))
    need = (T + 1) * fs
    ok = [i for i, l in enumerate(L) if l >= need]
    print(f'{len(ok)}/{len(L)} validation episodes long enough for T={T} (need {need} env steps)', flush=True)
    ok = list(rng.permutation(ok))[:n]
    V, Pp, A = [], [], []
    for i in ok:
        obs, act = dset[i][0], dset[i][1]
        s = int(rng.integers(0, L[i] - need + 1))
        idx = list(range(s, s + need, fs))[:T + 1]
        vis = pre.inverse_transform(obs['visual'][idx].cpu()) * 255.0
        V.append(vis.clamp(0, 255)); Pp.append(pre.denormalize_proprios(obs['proprio'][idx].cpu()))
        a = act[s:s + T * fs]
        A.append(a.reshape(T, -1))
    return torch.stack(V), torch.stack(Pp), torch.stack(A)


@torch.no_grad()
def reencode(wm, z):
    """z [B,1,V,h,w,D] -> E(D(z)) with the released image head (uint8 pixels, as real frames)."""
    from einops import rearrange
    fr = wm.model.heads['image_head'].decode(z)                 # [B,1,V,H,W,C] uint8 (cpu)
    vis = rearrange(fr[:, :, 0], 'b t h w c -> b t c h w').float().to(wm.device)
    return wm.encode(vis)                                      # [B,1,1,h,w,D]


def rel(a, b):
    return ((a - b) ** 2).flatten(1).sum(1) / (b ** 2).flatten(1).sum(1)


@torch.no_grad()
def rollout(wm, ctx, acts, ztrue, arm, cw):
    """ctx: TensorDict visual [B,1,...], proprio [B,1,...]; acts [B,T,A]; ztrue [B,T,...] (frames 1..T)."""
    import re
    P = int(re.findall(r'\d+', arm)[0]) if any(ch.isdigit() for ch in arm) else 0
    vid, prop = ctx['visual'], ctx['proprio']
    af = wm.model.encode_act(acts)
    T = acts.shape[1]
    err, drift = [], []
    for h in range(T):
        pv, _, pp = wm.model.forward_pred(vid[:, -cw:], af[:, :h + 1][:, -cw:], prop[:, -cw:])
        z, p = pv[:, -1:], pp[:, -1:]
        zr = reencode(wm, z)
        drift.append(rel(zr, z)); 
        if P and (h + 1) % P == 0 and h + 1 < T:
            if arm.startswith('reenc'):
                z = zr
            elif arm.startswith('noise'):
                d = (zr - z).flatten(1).norm(dim=1)
                e = torch.randn_like(z); e = e / e.flatten(1).norm(dim=1).view(-1, *[1] * (z.dim() - 1)) * d.view(-1, *[1] * (z.dim() - 1))
                z = z + e
            elif arm.startswith('oracle'):
                z = ztrue[:, h:h + 1]
        err.append(rel(z if not arm.startswith('oracle') else pv[:, -1:], ztrue[:, h:h + 1]))
        vid = torch.cat([vid, z], 1); prop = torch.cat([prop, p], 1)
    return torch.stack(err, 1), torch.stack(drift, 1)        # [B,T]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--T', type=int, default=15)
    ap.add_argument('--n', type=int, default=96); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--arms', default='none,reenc1,reenc2,reenc4,noise1,oracle4')
    ap.add_argument('--out', default='results_jwm'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--load_only', action='store_true', help='download/cache weights + data check, then exit')
    ap.add_argument('--cache_only', action='store_true', help='login node: download weights only (no dataset)')
    a = ap.parse_args()
    if a.cache_only:   # same path as torch.hub.load: checkpoint (HF), decoder head, DINOv2 -- all into local caches
        load_stubs()
        import hubconf
        hubconf._load_model(a.model, device='cpu')
        print(f'CACHED {a.model}', flush=True); raise SystemExit
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0); rng = np.random.default_rng(0)
    wm, dset, pre, fs, rsteps = load(a.model, dev)
    cw = wm.ctxt_window
    if a.load_only:
        print(f'LOADED {a.model}: {len(dset)} val episodes, frameskip {fs}, train rollout steps {rsteps}'); raise SystemExit
    T = 3 if a.smoke else a.T
    while True:
        try:
            V, Pr, A = episodes(dset, pre, fs, T, 4 if a.smoke else a.n, rng); break
        except Exception as e:
            if T <= 4: raise
            print('shrinking T:', e); T -= 2
    if len(V) < 8 and not a.smoke and T > 5:
        T = max(5, T // 2); V, Pr, A = episodes(dset, pre, fs, T, a.n, rng)
    print(f'model {a.model}: frameskip {fs}, rollout_steps(train) {rsteps}, ctxt_window {cw}, T {T}, '
          f'episodes {len(V)}, action dim {A.shape[-1]} (model {wm.action_dim})', flush=True)
    from tensordict import TensorDict
    res = {'model': a.model, 'T': T, 'frameskip': fs, 'train_rollout_steps': rsteps, 'n': int(len(V)), 'arms': {}}
    t0 = time.time()
    E, D, F = {k: [] for k in a.arms.split(',')}, {k: [] for k in a.arms.split(',')}, []
    for i in range(0, len(V), a.bs):
        v, p, ac = V[i:i + a.bs].to(dev), Pr[i:i + a.bs].to(dev), A[i:i + a.bs].to(dev).float()
        zt = wm.encode(TensorDict({'visual': v, 'proprio': p}, batch_size=[v.shape[0]]))
        ctx = TensorDict({'visual': zt['visual'][:, :1], 'proprio': zt['proprio'][:, :1]}, batch_size=[v.shape[0]])
        ztrue = zt['visual'][:, 1:]
        F.append(torch.stack([rel(reencode(wm, ztrue[:, k:k + 1]), ztrue[:, k:k + 1]) for k in range(T)], 1).cpu())
        for arm in E:
            e, d = rollout(wm, ctx, ac, ztrue, arm, cw)
            E[arm].append(e.cpu()); D[arm].append(d.cpu())
        print(f'  batch {i // a.bs + 1}: {time.time() - t0:.0f}s', flush=True)
    F = torch.cat(F)
    res['floor_per_step'] = F.mean(0).tolist()
    for arm in E:
        e, d = torch.cat(E[arm]), torch.cat(D[arm])
        res['arms'][arm] = {'err': e.numpy().tolist(), 'drift': d.numpy().tolist()}
        print(f'[{a.model}] {arm:8s} err step1 {e[:, 0].mean():.4f}  late {e[:, T // 2:].mean():.4f}  '
              f'drift late {d[:, T // 2:].mean():.4f} (floor {F.mean():.4f})', flush=True)
    if a.smoke:
        print('SMOKE OK'); raise SystemExit
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f'{a.out}/{a.model}.json', 'w'))
    print('saved', f'{a.out}/{a.model}.json')

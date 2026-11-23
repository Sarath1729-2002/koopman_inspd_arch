"""Course correction in a pretrained streaming video world model (Open-Oasis 500M, Minecraft; DiT over ViT-VAE
latents, trained on 32-frame windows, rolled out far beyond that).
Arms (all start from the same prompt frames and use the dataset's real action sequence):
  base        default inference (context labelled with stabilization noise level 15)
  stab1       no stabilization (shows how bad drift is without the model's own inference-time fix)
  stab50/100  stronger stabilization (the competitor knob that already exists at inference)
  reencP      decode -> re-encode the last P generated latent frames every P frames (P = 1, 4, 16), stab 15
  reenc1_stab50
Per frame we log: latent re-encode shift ||E(D(z)) - z||^2 / ||z||^2 (is the latent leaving the VAE manifold?),
latent channel mean/std, decoded brightness, saturation, sharpness (Laplacian variance), frame-to-frame change,
ResNet-50 features (Frechet distance to real frames, PCA-64, per 64-frame bin). Saves one mp4 per arm.
Usage: python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm base --frames 512 --out results_vid"""
import argparse, glob, json, os, sys, time, numpy as np, torch
from einops import rearrange

SCALE = 0.07843137255   # Oasis latent scaling factor
MAX_NOISE = 1000


def arm_cfg(arm):
    """base | stabN | reencP[_stabN] | reencP_mm | mm | evtK (re-encode a frame when its shift > K x real floor)
    | partA (every frame z <- z + 0.A (E(D(z)) - z))
    suffixes: _lf  low-frequency-preserving re-encode: keep the model's own blurred latent (colour / lighting), take only
                   the high-frequency structure from E(D(z))
              _mp  mean-preserving re-encode: keep each channel's spatial mean of z
              _s1  sampling seed 1 (noise-floor estimate)"""
    c = dict(stab=15, P=0, evt=0.0, alpha=1.0, mm=False, keep=None, seed=0, noise=False, sink=0, ctx=False, post=False)
    parts = arm.split('_')
    for q in parts:
        if q.startswith('stab'): c['stab'] = int(q[4:])
        elif q.startswith('sink'): c['sink'] = int(q[4:])
        elif q == 'ctx': c['ctx'] = True     # correction feeds ONLY the model's context; the output video keeps the raw frames
        elif q == 'post': c['post'] = True   # control: no feedback; re-encode every output frame once AFTER the rollout   # attention sink: first K frames (frame 0 = real prompt) stay in context
        elif q.startswith('reenc'): c['P'] = int(q[5:])
        elif q.startswith('evt'): c['evt'] = float(q[3:])
        elif q.startswith('part'): c['P'], c['alpha'] = 1, float('0.' + q[4:])
        elif q == 'mm': c['mm'] = True
        elif q in ('lf', 'mp'): c['keep'] = q
        elif q.startswith('noise'): c['P'], c['noise'] = int(q[5:]), True   # control: random nudge of equal size
        elif q.startswith('s') and q[1:].isdigit(): c['seed'] = int(q[1:])
    return c


def blur(z, k=5):
    import torch.nn.functional as F
    return F.avg_pool2d(F.pad(z, (k // 2,) * 4, mode='replicate'), k, stride=1)


def keep_lowfreq(z, zr, mode):
    """Combine the projected latent zr = E(D(z)) with the model's own latent z: the projection supplies structure,
    z keeps its colour / lighting so the VAE's reconstruction bias cannot compound."""
    if mode == 'lf':
        return zr - blur(zr) + blur(z)
    if mode == 'mp':
        return zr - zr.mean((-2, -1), keepdim=True) + z.mean((-2, -1), keepdim=True)
    return zr


def load_state(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location='cpu', weights_only=False)
    for k in ('state_dict', 'model', 'ema'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


def find(oasis_dir, stems):
    for s in stems:
        for ext in ('.safetensors', '.pt'):
            hits = glob.glob(os.path.join(oasis_dir, '**', s + ext), recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(f'none of {stems} in {oasis_dir}')


def code_root(oasis_dir):
    """Find dit.py and vae.py anywhere under oasis_dir (mirrors nest them differently); put their dirs (and package
    parents) on sys.path. Returns the dir holding dit.py."""
    found = {}
    for root, dirs, files in os.walk(oasis_dir):
        for f in ('dit.py', 'vae.py'):
            if f in files and f not in found:
                found[f] = root
    if 'dit.py' not in found or 'vae.py' not in found:
        listing = sorted(os.path.relpath(os.path.join(r, f), oasis_dir) for r, _, fs in os.walk(oasis_dir) for f in fs
                         if f.endswith('.py'))[:60]
        raise FileNotFoundError(f'need dit.py AND vae.py under {oasis_dir}; found {found}; .py files: {listing}')
    for d in set(found.values()):
        for p in (d, os.path.dirname(d)):
            if p not in sys.path:
                sys.path.insert(0, p)
    return found['dit.py']


def find_file(oasis_dir, name):
    for root, _, files in os.walk(oasis_dir):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def import_mod(name, d):
    """import dit / vae either as top-level modules or as part of a package directory."""
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        return importlib.import_module(f'{os.path.basename(d)}.{name}')


def sigmoid_beta_schedule(T, start=-3, end=3, tau=1, clamp_min=1e-5):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    v_start, v_end = torch.tensor(start / tau).sigmoid(), torch.tensor(end / tau).sigmoid()
    ac = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 0, 0.999)


class Oasis:
    def __init__(s, oasis_dir, dev):
        code_dir = code_root(oasis_dir)
        print('Oasis code dir:', code_dir, flush=True)
        DiT_models = import_mod('dit', code_dir).DiT_models
        VAE_models = import_mod('vae', os.path.dirname(find_file(oasis_dir, 'vae.py'))).VAE_models
        s.dev = dev
        s.model = DiT_models['DiT-S/2']()
        missing = s.model.load_state_dict(load_state(find(oasis_dir, ['oasis500m'])), strict=False)
        print('DiT missing/unexpected keys:', len(missing.missing_keys), len(missing.unexpected_keys), flush=True)
        s.vae = VAE_models['vit-l-20-shallow-encoder']()
        s.vae.load_state_dict(load_state(find(oasis_dir, ['vit-l-20'])))
        s.model, s.vae = s.model.to(dev).eval(), s.vae.to(dev).eval()
        betas = sigmoid_beta_schedule(MAX_NOISE)   # same schedule as Oasis utils.py (own copy: no import clash)
        s.ac = rearrange(torch.cumprod(1.0 - betas.float().to(dev), 0), 'T -> T 1 1 1')
        s.max_frames = getattr(s.model, 'max_frames', 32)
        s.patch = s.vae.patch_size

    @torch.no_grad()
    def encode(s, img):  # img [N,3,H,W] in [0,1] -> latents [N,c,h,w]
        H, W = img.shape[-2:]
        with torch.autocast('cuda', dtype=torch.half):
            z = s.vae.encode(img * 2 - 1).mean * SCALE
        return rearrange(z.float(), 'n (h w) c -> n c h w', h=H // s.patch, w=W // s.patch)

    @torch.no_grad()
    def decode(s, z, bs=16):  # z [N,c,h,w] -> img [N,3,H,W] in [0,1]
        out = []
        for i in range(0, z.shape[0], bs):
            zz = rearrange(z[i:i + bs], 'n c h w -> n (h w) c')
            with torch.autocast('cuda', dtype=torch.half):
                x = s.vae.decode(zz / SCALE)
            out.append(((x.float() + 1) / 2).clamp(0, 1))
        return torch.cat(out)

    def reencode(s, z):
        return s.encode(s.decode(z))

    @torch.no_grad()
    def rollout(s, x, actions, frames, cfg, floor=0.0, ref_mu=None, ref_sd=None, ddim_steps=10, log_every=64):
        """x: prompt latents [B, n_p, c, h, w]; actions [B, frames, A].
        Returns latents [B, frames, c, h, w], shift [B, frames], number of re-encoded frames per sample [B]."""
        stab, P = cfg['stab'], cfg['P']
        B, n_p = x.shape[:2]
        n_re = torch.zeros(B, device=s.dev)

        def mm(zz):  # match per-channel latent mean/std (over h, w) to real-frame latent statistics
            mu, sd = zz.mean((-2, -1), keepdim=True), zz.std((-2, -1), keepdim=True) + 1e-6
            return (zz - mu) / sd * ref_sd.view(1, -1, 1, 1) + ref_mu.view(1, -1, 1, 1)
        noise_range = torch.linspace(-1, MAX_NOISE - 1, ddim_steps + 1).long()
        shift = torch.zeros(B, frames, device=s.dev)
        t0 = time.time()
        raw = []
        for i in range(n_p, frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=s.dev).clamp(-20, 20)
            x = torch.cat([x, chunk], 1)
            start = max(0, i + 1 - s.max_frames)
            K = cfg.get('sink', 0)
            if K and start > 0:   # sink frames + the most recent (max_frames - K) frames
                idx = torch.tensor(list(range(K)) + list(range(max(K, i + 1 - (s.max_frames - K)), i + 1)), device=s.dev)
            else:
                idx = torch.arange(start, i + 1, device=s.dev)
            for ni in reversed(range(1, ddim_steps + 1)):
                t_ctx = torch.full((B, i), stab - 1, dtype=torch.long, device=s.dev)
                t = torch.full((B, 1), int(noise_range[ni]), dtype=torch.long, device=s.dev)
                t_next = torch.full((B, 1), int(noise_range[ni - 1]), dtype=torch.long, device=s.dev)
                t_next = torch.where(t_next < 0, t, t_next)
                t, t_next = torch.cat([t_ctx, t], 1)[:, idx], torch.cat([t_ctx, t_next], 1)[:, idx]
                xc = x[:, idx]
                with torch.autocast('cuda', dtype=torch.half):
                    v = s.model(xc, t, actions[:, idx])
                v = v.float()
                a_t = s.ac[t]
                x_start = a_t.sqrt() * xc - (1 - a_t).sqrt() * v
                x_noise = ((1 / a_t).sqrt() * xc - x_start) / (1 / a_t - 1).sqrt()
                a_next = s.ac[t_next].clone()
                a_next[:, :-1] = 1.0
                if ni == 1:
                    a_next[:, -1:] = 1.0
                x_pred = a_next.sqrt() * x_start + x_noise * (1 - a_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]
            z = x[:, -1]
            if cfg.get('ctx'):
                raw.append(z.clone())
            zr_raw = s.reencode(z)
            zr = keep_lowfreq(z, zr_raw, cfg.get('keep'))
            shift[:, i] = (zr_raw - z).pow(2).flatten(1).sum(1) / z.pow(2).flatten(1).sum(1)
            if cfg['evt'] > 0:
                m = shift[:, i] > cfg['evt'] * floor
                if m.any():
                    x[m, -1] = zr[m]; n_re += m.float()
            elif P and (i - n_p + 1) % P == 0:
                k = min(P, i - n_p + 1)
                if cfg.get('noise'):
                    zk = rearrange(x[:, -k:], 'b k c h w -> (b k) c h w')
                    d = (keep_lowfreq(zk, s.reencode(zk), cfg.get('keep')) - zk).flatten(1).norm(dim=1)
                    eps = torch.randn_like(zk)
                    eps = eps / eps.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * d.view(-1, 1, 1, 1)
                    x[:, -k:] = rearrange(zk + eps, '(b k) c h w -> b k c h w', b=B)
                elif k == 1:
                    x[:, -1] = z + cfg['alpha'] * (zr - z)
                else:
                    zk = rearrange(x[:, -k:], 'b k c h w -> (b k) c h w')
                    x[:, -k:] = rearrange(keep_lowfreq(zk, s.reencode(zk), cfg.get('keep')),
                                          '(b k) c h w -> b k c h w', b=B)
                n_re += k
            if cfg['mm']:
                x[:, -1] = mm(x[:, -1])
            if (i + 1) % log_every == 0:
                print(f'  frame {i + 1}/{frames}  shift {shift[:, i].mean():.4f}  ({time.time() - t0:.0f}s)', flush=True)
        if cfg.get('ctx'):
            x = torch.cat([x[:, :n_p], torch.stack(raw, 1)], 1)
        if cfg.get('post'):
            for j in range(n_p, x.shape[1]):
                x[:, j] = s.reencode(x[:, j])
            n_re += x.shape[1] - n_p
        return x, shift, n_re


# ------------------------------------------------------------------ data + metrics
def read_frames(path, max_frames=None):
    import av
    fr = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            fr.append(torch.from_numpy(f.to_ndarray(format='rgb24')))
            if max_frames and len(fr) >= max_frames:
                break
    return torch.stack(fr).permute(0, 3, 1, 2).float() / 255.0   # [T,3,H,W] in [0,1]


def write_mp4(path, frames_u8, fps=20):
    import av
    T, H, W, _ = frames_u8.shape
    with av.open(path, 'w') as c:
        st = c.add_stream('libx264', rate=fps); st.width, st.height, st.pix_fmt = W, H, 'yuv420p'
        for f in frames_u8.numpy():
            for pk in st.encode(av.VideoFrame.from_ndarray(f, format='rgb24')):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)


ACT_ROOT = [None]


def oasis_one_hot(raw):
    """Original Oasis release stores actions as a list of dicts; convert with its own utils.one_hot_actions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('oasis_utils', find_file(ACT_ROOT[0], 'utils.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.one_hot_actions(raw)


def load_actions(path, offset, n, A=25):
    if path is None:
        return torch.zeros(n, A)
    a = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(a, (list, tuple)) and len(a) and isinstance(a[0], dict):
        a = oasis_one_hot(a)
    a = torch.as_tensor(a).float()
    if a.dim() == 3:
        a = a[0]
    a = torch.cat([torch.zeros_like(a[:1]), a[offset:]], 0)   # Oasis convention: zero action for the prompt frame
    reps = int(np.ceil(n / a.shape[0]))
    return a.repeat(reps, 1)[:n]


def vit(dev, allow_random=False):
    import torchvision
    try:
        m = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
    except Exception as e:
        if not allow_random:
            raise RuntimeError('ViT-B/16 weights missing: run the weight download on the login node') from e
        print('!! ViT weights unavailable, random weights (selftest only)'); m = torchvision.models.vit_b_16()
    m.heads = torch.nn.Identity()
    return m.to(dev).eval()


def resnet(dev, allow_random=False):
    import torchvision
    try:
        m = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    except Exception as e:
        if not allow_random:
            raise RuntimeError('ResNet-50 weights missing: run vid/prep.sh on the login node') from e
        print('!! ResNet weights unavailable, random weights (selftest only)'); m = torchvision.models.resnet50()
    m.fc = torch.nn.Identity()
    return m.to(dev).eval()


@torch.no_grad()
def feats(net, img, bs=64):
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    out = []
    for i in range(0, img.shape[0], bs):
        x = F.interpolate(img[i:i + bs], size=(224, 224), mode='bilinear', align_corners=False)
        with torch.autocast('cuda', dtype=torch.half):
            out.append(net((x - mean) / std).float())
    return torch.cat(out)


@torch.no_grad()
def img_stats(img):
    gray = img.mean(1, keepdim=True)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sharp = torch.nn.functional.conv2d(gray, lap).flatten(1).var(1)
    sat = (img.max(1).values - img.min(1).values).flatten(1).mean(1)
    return {'bright': img.flatten(1).mean(1), 'sat': sat, 'sharp': sharp}


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr(sqrtm(s1 s2)) = sum of sqrt of the (real, non-negative) eigenvalues of s1 s2 -- version-independent
    tr_cs = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_cs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--oasis_dir', default='ext/open-oasis')
    ap.add_argument('--arm', default='base')
    ap.add_argument('--frames', type=int, default=512)
    ap.add_argument('--offsets', default='100,300,500,700')
    ap.add_argument('--video_idx', type=int, default=-1, help='which sample video gives prompts/actions (-1: first, old naming)')
    ap.add_argument('--out', default='results_vid')
    ap.add_argument('--selftest', action='store_true', help='3 frames, 1 prompt: checks the whole pipeline')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    tag = a.arm if a.video_idx < 0 else f'{a.arm}_v{a.video_idx}'
    f_out = f'{a.out}/{tag}.json'
    if os.path.exists(f_out) and not a.selftest:
        print('exists', f_out); raise SystemExit
    torch.manual_seed(0)
    O = Oasis(a.oasis_dir, dev)
    vids = sorted(glob.glob(os.path.join(a.oasis_dir, '**', '*.mp4'), recursive=True))
    acts = sorted(glob.glob(os.path.join(a.oasis_dir, '**', '*.one_hot_actions.pt'), recursive=True) +
                  glob.glob(os.path.join(a.oasis_dir, '**', '*.actions.pt'), recursive=True))
    ACT_ROOT[0] = a.oasis_dir
    vids = [v for v in vids if not os.path.basename(v).startswith(('video', 'out'))]   # skip generated outputs
    print('videos', [os.path.basename(v) for v in vids][:5], 'actions', [os.path.basename(x) for x in acts][:5], flush=True)
    assert vids, 'no .mp4 prompt videos found under oasis_dir'
    pairs = []
    for v in vids:
        stem = os.path.basename(v).replace('.mp4', '')
        match = [x for x in acts if os.path.basename(x).startswith(stem)] or \
                [x for x in acts if os.path.basename(x).startswith(stem.replace('video', 'actions'))] or acts[:1] or [None]
        pairs.append((v, match[0]))
    if pairs[0][1] is None:
        print('!! no action file found: using all-zero (no-op) actions for every arm', flush=True)
    vi = max(a.video_idx, 0)
    real = read_frames(pairs[vi][0])
    torch.manual_seed(arm_cfg(a.arm)['seed'])
    print('video', pairs[vi], 'frames', tuple(real.shape), flush=True)
    # real reference (pooled over all sample videos): VAE re-encode floor and per-channel latent stats
    ref = torch.cat([read_frames(v)[::2] for v, _ in pairs])[:1200].to(dev)
    zr = O.encode(ref[:256])
    floor = ((O.reencode(zr) - zr).pow(2).flatten(1).sum(1) / zr.pow(2).flatten(1).sum(1)).mean().item()
    ref_mu, ref_sd = zr.mean((0, 2, 3)), zr.std((0, 2, 3))
    offsets = [int(x) for x in a.offsets.split(',')]
    frames = (40 if arm_cfg(a.arm)['sink'] else 3) if a.selftest else a.frames   # 40 > 32 exercises the sink path
    if a.selftest:
        offsets = offsets[:1]
    offsets = [min(o_, real.shape[0] - 1) for o_ in offsets]
    prompts = torch.stack([real[o_] for o_ in offsets]).to(dev)                       # [B,3,H,W]
    actions = torch.stack([load_actions(pairs[vi][1], o_, frames) for o_ in offsets]).to(dev)
    x0 = O.encode(prompts).unsqueeze(1)                                                # [B,1,c,h,w]
    cfg = arm_cfg(a.arm); stab, P = cfg['stab'], cfg['P']
    print(f'arm {a.arm}: {cfg}  real floor {floor:.4f}  B={len(offsets)} frames={frames} '
          f'latent {tuple(x0.shape)} max_frames {O.max_frames}', flush=True)
    t0 = time.time()
    outs = []   # roll prompts out in chunks of 4 to bound GPU memory (v1 used ~42 GB at 4 prompts x 512 frames)
    for c0 in range(0, x0.shape[0], 4):
        outs.append(O.rollout(x0[c0:c0 + 4], actions[c0:c0 + 4], frames, cfg, floor, ref_mu, ref_sd))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    lat, shift, n_re = (torch.cat([o_[j] for o_ in outs]) for j in range(3))
    t_roll = time.time() - t0
    print(f'rollout done in {t_roll:.0f}s', flush=True)
    B, T = lat.shape[:2]
    allow = a.selftest or os.environ.get('VID_ALLOW_RANDOM') == '1'
    nets = {'resnet': resnet(dev, allow), 'vit': vit(dev, allow)}
    stats = {k: np.zeros((B, T), np.float32) for k in ('bright', 'sat', 'sharp')}
    change = np.zeros((B, T - 1), np.float32)
    feats_all = {k: [] for k in nets}
    strip_t = sorted({0, T // 4, T // 2, (3 * T) // 4, T - 1})
    strips, prev = {}, None
    mp4 = None
    if not a.selftest:
        try:
            import av
            mp4 = av.open(f'{a.out}/{tag}.mp4', 'w'); vs = mp4.add_stream('libx264', rate=20)
            vs.width, vs.height, vs.pix_fmt = lat.shape[-1] * O.patch, lat.shape[-2] * O.patch, 'yuv420p'
        except Exception as e:
            print('mp4 disabled:', e); mp4 = None
    CH = 32
    for t0_ in range(0, T, CH):
        t1_ = min(T, t0_ + CH)
        im = O.decode(rearrange(lat[:, t0_:t1_], 'b t c h w -> (b t) c h w'))           # [B*k,3,H,W]
        k = t1_ - t0_
        for name, v in img_stats(im).items():
            stats[name][:, t0_:t1_] = rearrange(v, '(b t) -> b t', b=B).cpu().numpy()
        imv = im.view(B, k, *im.shape[1:])
        seq = imv if prev is None else torch.cat([prev, imv], 1)
        ch = (seq[:, 1:] - seq[:, :-1]).abs().flatten(2).mean(2).cpu().numpy()
        change[:, max(0, t0_ - 1):t1_ - 1] = ch
        prev = imv[:, -1:]
        for name, net in nets.items():
            feats_all[name].append(feats(net, im).cpu().view(B, k, -1).half())
        for tt in strip_t:
            if t0_ <= tt < t1_:
                strips[tt] = (imv[0, tt - t0_].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if mp4 is not None:
            import av
            for f in (imv[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8):
                for pk in vs.encode(av.VideoFrame.from_ndarray(f, format='rgb24')):
                    mp4.mux(pk)
    if mp4 is not None:
        for pk in vs.encode():
            mp4.mux(pk)
        mp4.close()
    F = {n: torch.cat(v, 1).float().numpy() for n, v in feats_all.items()}          # [B,T,D]
    # real reference features (pooled sample videos), cached once per output dir
    ref_f = f'{a.out}/ref_feats.npz'
    if os.path.exists(ref_f) and not a.selftest:
        Rf = dict(np.load(ref_f))
    else:
        Rf = {n: feats(net, ref).cpu().numpy() for n, net in nets.items()}
        if not a.selftest:
            np.savez(ref_f + '.tmp.npz', **Rf); os.replace(ref_f + '.tmp.npz', ref_f)   # atomic: runs share it
    st_ref = {k: float(v.mean()) for k, v in img_stats(ref).items()}
    st_vid = {k: float(v.mean()) for k, v in img_stats(real[::4][:300].to(dev)).items()}
    lat_ref = {'mean': float(zr.mean()), 'std': float(zr.std())}
    # action responsiveness: does frame change still follow the actions? (camera = keys 15,16; movement 11-14)
    act = actions[:, :T].cpu().numpy()
    a_mag = np.abs(act[:, 1:, 15:17]).sum(-1) + np.abs(act[:, 1:, 11:15]).sum(-1)
    a_all = np.abs(act[:, 1:]).sum(-1)
    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))
    resp = [corr(a_mag[b], change[b]) for b in range(B)]
    resp_all = [corr(a_all[b], change[b]) for b in range(B)]
    from sklearn.decomposition import PCA
    fd = {}
    for n in nets:
        pca = PCA(64).fit(Rf[n]); R64 = pca.transform(Rf[n])
        bins, vals = [], []
        for b0 in range(0, T, 128):
            G = F[n][:, b0:b0 + 128].reshape(-1, F[n].shape[-1])
            if G.shape[0] >= 32:
                bins.append(b0); vals.append(frechet(pca.transform(G), R64))
        fd[n] = vals
    res = {'arm': a.arm, 'video_idx': vi, 'video': os.path.basename(pairs[vi][0]), 'cfg': cfg,
           'n_reenc': n_re.cpu().numpy().tolist(), 'stab': stab, 'P': P, 'frames': T, 'offsets': offsets,
           'shift_floor_real': floor, 'ref': {**st_ref, 'lat': lat_ref}, 'ref_video': st_vid,
           'shift': shift.cpu().numpy().tolist(), **{k: v.tolist() for k, v in stats.items()},
           'lat_mean': lat.mean((2, 3, 4)).cpu().numpy().tolist(), 'lat_std': lat.std((2, 3, 4)).cpu().numpy().tolist(),
           'change': change.tolist(), 'resp': resp, 'resp_all': resp_all, 'fd_bins': bins, 'fd': fd['resnet'],
           'fd_vit': fd['vit'], 'seconds_rollout': t_roll, 'seconds': time.time() - t0}
    if a.selftest:
        print('SELFTEST OK', {'floor': floor, 'fd': fd, 'resp': resp, 'feat_shapes': {n: v.shape for n, v in F.items()}})
        raise SystemExit
    np.savez_compressed(f'{a.out}/{tag}_feats.npz', **{n: v.astype(np.float16) for n, v in F.items()}, actions=act)
    try:
        from PIL import Image
        Image.fromarray(np.concatenate([strips[tt] for tt in strip_t], 1)).save(f'{a.out}/{tag}_strip.jpg', quality=85)
    except Exception as e:
        print('strip failed:', e)
    json.dump(res, open(f_out + '.tmp', 'w')); os.replace(f_out + '.tmp', f_out)
    sh = np.mean(res['shift'], 0)
    print(f'[{tag}] re-encodes/sample {n_re.mean().item():.0f}  shift first/last {sh[1:65].mean():.4f} / {sh[-64:].mean():.4f} '
          f'(floor {floor:.4f})  FD resnet {fd["resnet"][0]:.1f}->{fd["resnet"][-1]:.1f}  vit {fd["vit"][0]:.1f}->{fd["vit"][-1]:.1f}  '
          f'bright last/video {stats["bright"][:, -64:].mean() / st_vid["bright"]:.2f}  resp {np.mean(resp):+.2f}  '
          f'rollout {t_roll:.0f}s', flush=True)

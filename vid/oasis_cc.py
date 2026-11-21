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
    stab, P = 15, 0
    if arm.startswith('stab'):
        stab = int(arm[4:])
    if arm.startswith('reenc'):
        parts = arm.split('_')
        P = int(parts[0][5:])
        if len(parts) > 1:
            stab = int(parts[1][4:])
    return stab, P


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
    def rollout(s, x, actions, frames, stab, P, ddim_steps=10, log_every=64):
        """x: prompt latents [B, n_p, c, h, w]; actions [B, frames, A]. Returns latents [B, frames, c, h, w], shift [B, frames]."""
        B, n_p = x.shape[:2]
        noise_range = torch.linspace(-1, MAX_NOISE - 1, ddim_steps + 1).long()
        shift = torch.zeros(B, frames, device=s.dev)
        t0 = time.time()
        for i in range(n_p, frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=s.dev).clamp(-20, 20)
            x = torch.cat([x, chunk], 1)
            start = max(0, i + 1 - s.max_frames)
            for ni in reversed(range(1, ddim_steps + 1)):
                t_ctx = torch.full((B, i), stab - 1, dtype=torch.long, device=s.dev)
                t = torch.full((B, 1), int(noise_range[ni]), dtype=torch.long, device=s.dev)
                t_next = torch.full((B, 1), int(noise_range[ni - 1]), dtype=torch.long, device=s.dev)
                t_next = torch.where(t_next < 0, t, t_next)
                t, t_next = torch.cat([t_ctx, t], 1)[:, start:], torch.cat([t_ctx, t_next], 1)[:, start:]
                xc = x[:, start:]
                with torch.autocast('cuda', dtype=torch.half):
                    v = s.model(xc, t, actions[:, start:i + 1])
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
            zr = s.reencode(z)
            shift[:, i] = (zr - z).pow(2).flatten(1).sum(1) / z.pow(2).flatten(1).sum(1)
            if P and (i - n_p + 1) % P == 0:
                k = min(P, i - n_p + 1)
                x[:, -k:] = zr.unsqueeze(1) if k == 1 else rearrange(
                    s.reencode(rearrange(x[:, -k:], 'b k c h w -> (b k) c h w')), '(b k) c h w -> b k c h w', b=B)
            if (i + 1) % log_every == 0:
                print(f'  frame {i + 1}/{frames}  shift {shift[:, i].mean():.4f}  ({time.time() - t0:.0f}s)', flush=True)
        return x, shift


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
    ap.add_argument('--out', default='results_vid')
    ap.add_argument('--selftest', action='store_true', help='3 frames, 1 prompt: checks the whole pipeline')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    f_out = f'{a.out}/{a.arm}.json'
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
    real = read_frames(pairs[0][0])
    print('video', pairs[0], 'frames', tuple(real.shape), flush=True)
    offsets = [int(x) for x in a.offsets.split(',')]
    frames = 3 if a.selftest else a.frames
    if a.selftest:
        offsets = offsets[:1]
    offsets = [min(o_, real.shape[0] - 1) for o_ in offsets]
    prompts = torch.stack([real[o_] for o_ in offsets]).to(dev)                       # [B,3,H,W]
    actions = torch.stack([load_actions(pairs[0][1], o_, frames) for o_ in offsets]).to(dev)
    x0 = O.encode(prompts).unsqueeze(1)                                                # [B,1,c,h,w]
    stab, P = arm_cfg(a.arm)
    print(f'arm {a.arm}: stabilization {stab}, re-encode every {P}  B={len(offsets)} frames={frames} '
          f'latent {tuple(x0.shape)} max_frames {O.max_frames}', flush=True)
    t0 = time.time()
    lat, shift = O.rollout(x0, actions, frames, stab, P)
    print(f'rollout done in {time.time() - t0:.0f}s', flush=True)
    B = lat.shape[0]
    imgs = O.decode(rearrange(lat, 'b t c h w -> (b t) c h w'))
    st = {k: rearrange(v, '(b t) -> b t', b=B).cpu().numpy() for k, v in img_stats(imgs).items()}
    lat_mean = lat.mean((2, 3, 4)).cpu().numpy(); lat_std = lat.std((2, 3, 4)).cpu().numpy()
    change = (imgs.view(B, frames, *imgs.shape[1:])[:, 1:] - imgs.view(B, frames, *imgs.shape[1:])[:, :-1]
              ).abs().flatten(2).mean(2).cpu().numpy()
    net = resnet(dev, allow_random=a.selftest or os.environ.get('VID_ALLOW_RANDOM') == '1')
    F_gen = feats(net, imgs).cpu().numpy().reshape(B, frames, -1)
    # real reference: all frames of the sample videos (subsampled), their stats, and the VAE re-encode floor
    ref = torch.cat([read_frames(v)[::2] for v, _ in pairs])[:1200].to(dev)
    F_ref = feats(net, ref).cpu().numpy()
    zr = O.encode(ref[:256]); floor = ((O.reencode(zr) - zr).pow(2).flatten(1).sum(1) / zr.pow(2).flatten(1).sum(1)).mean().item()
    st_ref = {k: float(v.mean()) for k, v in img_stats(ref).items()}
    lat_ref = {'mean': float(zr.mean()), 'std': float(zr.std())}
    from sklearn.decomposition import PCA
    pca = PCA(64).fit(F_ref)
    R64 = pca.transform(F_ref)
    bins, fd = [], []
    for b0 in range(0, frames, 64):
        G = F_gen[:, b0:b0 + 64].reshape(-1, F_gen.shape[-1])
        if G.shape[0] >= 32:
            bins.append(b0); fd.append(frechet(pca.transform(G), R64))
    res = {'arm': a.arm, 'stab': stab, 'P': P, 'frames': frames, 'offsets': offsets, 'shift_floor_real': floor,
           'ref': {**st_ref, 'lat': lat_ref}, 'shift': shift.cpu().numpy().tolist(),
           'bright': st['bright'].tolist(), 'sat': st['sat'].tolist(), 'sharp': st['sharp'].tolist(),
           'lat_mean': lat_mean.tolist(), 'lat_std': lat_std.tolist(), 'change': change.tolist(),
           'fd_bins': bins, 'fd': fd, 'seconds': time.time() - t0}
    if a.selftest:
        print('SELFTEST OK', {k: (v if not isinstance(v, list) else np.asarray(v).shape) for k, v in res.items()
                               if k in ('shift_floor_real', 'fd', 'ref')})
        raise SystemExit
    json.dump(res, open(f_out + '.tmp', 'w')); os.replace(f_out + '.tmp', f_out)
    try:
        v = (imgs.view(B, frames, *imgs.shape[1:])[0].permute(0, 2, 3, 1).cpu() * 255).byte()
        write_mp4(f'{a.out}/{a.arm}.mp4', v)
    except Exception as e:
        print('video write failed:', e)
    print(f'[{a.arm}] shift first/last bin {np.mean(res["shift"], 0)[1:65].mean():.4f} / {np.mean(res["shift"], 0)[-64:].mean():.4f} '
          f'(real floor {floor:.4f})  FD first/last {fd[0]:.1f} / {fd[-1]:.1f}  bright last/ref '
          f'{st["bright"][:, -64:].mean() / st_ref["bright"]:.2f}', flush=True)

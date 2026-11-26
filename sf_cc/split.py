"""Two kinds of drift (Nov 26). For each long video, at 11 evenly spaced 2-s windows, project the decoded frames back
through the model's own VAE, P(x) = D(E(x)), and measure in DINOv2 feature space:
  off-manifold  off_k = 1 - cos(DINO(x), DINO(P(x)))         what the VAE round trip removes (artifacts, texture breakdown)
  on-manifold   on_k  = 1 - cos(DINO(P(x))_k, DINO(P(x))_0)  how far the CLEANED content has moved from the opening window
Growth = mean of the last two windows minus the first window. Off-share = off growth / (|off growth| + |on growth|).
Videos: CausVid and Self-Forcing at 4 min (results_sf_long; Wan VAE) and Open-Oasis (results_final, 768 frames;
Oasis VAE), each without and with re-encoding.
Criteria (fixed before the run):
  SP1 the models re-encoding helps have more off-manifold drift: off-share(CausVid, no correction) -
      off-share(Self-Forcing, no correction) > 0, bootstrap CI over videos > 0   (Oasis reported alongside; n is small)
  SP2 re-encoding acts on the off-manifold part: on CausVid, paired over prompts,
      (off growth without - with re-encoding) - (on growth without - with re-encoding) CI > 0
Writes results_split/split.json + SPLIT.md (and a per-video cache)."""
import glob, json, os, sys, numpy as np, torch, torch.nn.functional as F

ROOT = os.path.abspath('.')
OUT = os.path.join(ROOT, 'results_split'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frames_at(path, starts, n):
    """uint8 [len(starts), n, H, W, 3]: n consecutive frames from each start index"""
    import av
    want = {s + j: (i, j) for i, s in enumerate(starts) for j in range(n)}
    out = None
    with av.open(path) as c:
        for t, fr in enumerate(c.decode(video=0)):
            if t in want:
                a = fr.to_ndarray(format='rgb24')
                if out is None:
                    out = np.zeros((len(starts), n, *a.shape), np.uint8)
                i, j = want[t]; out[i, j] = a
            if t > max(want):
                break
    return out


def n_frames(path):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        return s.frames or sum(1 for _ in c.decode(video=0))


@torch.no_grad()
def dino_feats(dino, x, dev):   # x float [N,3,H,W] in [0,1]
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return F.normalize(dino((x - IN_MEAN.to(dev)) / IN_STD.to(dev)), dim=-1)


@torch.no_grad()
def wan_project(wrapper, clip, dev):
    """clip uint8 [T,H,W,3], T = 1 + 4k -> P(clip) float [T,3,H,W] in [0,1] through Wan's causal 3D VAE"""
    m = wrapper.model; cc = sys.modules[type(m).__module__].count_conv3d
    dt = next(m.parameters()).dtype
    pix = torch.from_numpy(clip).to(dev).permute(3, 0, 1, 2)[None].to(dt) / 127.5 - 1
    enc = [None] * cc(m.encoder)
    outs = [m.encoder(pix[:, :, :1], feat_cache=enc, feat_idx=[0])]
    for p in range(1, pix.shape[2], 4):
        outs.append(m.encoder(pix[:, :, p:p + 4], feat_cache=enc, feat_idx=[0]))
    mu, _ = m.conv1(torch.cat(outs, 2)).chunk(2, dim=1)
    x = m.conv2(mu); dec = [None] * cc(m.decoder)
    px = torch.cat([m.decoder(x[:, :, i:i + 1], feat_cache=dec, feat_idx=[0]) for i in range(x.shape[2])], 2).clamp(-1, 1)
    return (px[0].permute(1, 0, 2, 3).float() * 0.5 + 0.5).clamp(0, 1)


def analyse(path, project, dino, dev, n=17, K=11):
    T = n_frames(path)
    starts = [int(round(s)) for s in np.linspace(0, T - n - 1, K)]
    W = frames_at(path, starts, n)
    off, cen = [], []
    for w in W:
        x = torch.from_numpy(w).to(dev).permute(0, 3, 1, 2).float() / 255
        px = project(w)
        sel = list(range(0, n, 4))
        fx, fp = dino_feats(dino, x[sel], dev), dino_feats(dino, px[sel], dev)
        off.append(float((1 - (fx * fp).sum(-1)).mean())); cen.append(F.normalize(fp.mean(0), dim=0))
    on = [float(1 - cen[k] @ cen[0]) for k in range(K)]
    return {'off': off, 'on': on, 'off_growth': float(np.mean(off[-2:]) - off[0]), 'on_growth': float(np.mean(on[-2:]))}


def boot(x, B=4000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(x.mean())] + [float(v) for v in np.percentile(bs, [2.5, 97.5])]


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    dev = 'cuda'
    cache_f = os.path.join(OUT, 'cache.json')
    C = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(dev).eval()
    groups = {'CausVid': [('none', 'results_sf_long/causvid_sink3'), ('reenc', 'results_sf_long/causvid_sink3_reenc2_cache')],
              'Self-Forcing': [('none', 'results_sf_long/sink3'), ('reenc', 'results_sf_long/sink3_reenc2_cache')]}
    todo = []
    for model, arms in groups.items():
        for tag, d in arms:
            vids = sorted(f for f in glob.glob(os.path.join(ROOT, d, '*.mp4')) if os.path.exists(f[:-4] + '.json'))
            todo += [(model, tag, f) for f in (vids[:1] if smoke else vids)]
    if os.environ.get('SPLIT_SKIP_WAN') != '1' and any(f"{m}|{t}|{os.path.basename(f)}" not in C for m, t, f in todo):
        sys.path.insert(0, os.path.join(ROOT, 'sf_cc'))
        from sf_cc.run_sf import build
        pipe, _ = build(sink=3, ckpt='dmd'); os.chdir(ROOT)   # only the VAE is used
        del pipe.generator, pipe.text_encoder; torch.cuda.empty_cache()
        for m, t, f in todo:
            k = f"{m}|{t}|{os.path.basename(f)}"
            if k in C:
                continue
            C[k] = analyse(f, lambda w: wan_project(pipe.vae, w, dev), dino, dev)
            print('[split]', k, f"off +{C[k]['off_growth']:.4f} on +{C[k]['on_growth']:.4f}", flush=True)
            json.dump(C, open(cache_f, 'w'))
        del pipe; torch.cuda.empty_cache()
    # Oasis (its own VAE); first prompt of each run, 768 frames
    oas = [('none', 'sink4'), ('reenc', 'sink4_reenc4')]
    ofiles = [(t, f) for t, a_ in oas for f in sorted(glob.glob(os.path.join(ROOT, 'results_final', f'{a_}_v[0-9].mp4')) +
                                                    glob.glob(os.path.join(ROOT, 'results_final', f'{a_}_s1_v[0-9].mp4')))]
    if os.environ.get('SPLIT_SKIP_OASIS') != '1' and ofiles and any(f"Oasis|{t}|{os.path.basename(f)}" not in C for t, f in ofiles):
        from vid.oasis_cc import Oasis
        O = Oasis(os.path.join(ROOT, 'ext/open-oasis'), dev)
        def oproj(w):
            x = torch.from_numpy(w).to(dev).permute(0, 3, 1, 2).float() / 255
            return O.decode(O.encode(x))
        for t, f in (ofiles[:1] if smoke else ofiles):
            k = f"Oasis|{t}|{os.path.basename(f)}"
            if k in C:
                continue
            C[k] = analyse(f, oproj, dino, dev); json.dump(C, open(cache_f, 'w'))
            print('[split]', k, f"off +{C[k]['off_growth']:.4f} on +{C[k]['on_growth']:.4f}", flush=True)
    # summary
    share = lambda r: r['off_growth'] / max(abs(r['off_growth']) + abs(r['on_growth']), 1e-9)
    res, L = {}, ['# Two kinds of drift', '', 'Growth from the first to the last 2-s windows, DINOv2 cosine distance; mean [95% CI] over videos.', '',
                  '| model | arm | n | off-manifold growth | on-manifold growth | off-share |', '|---|---|---|---|---|---|']
    by = {}
    for k, r in C.items():
        m, t, v = k.split('|'); by.setdefault((m, t), {})[v] = r
    for (m, t), d in sorted(by.items()):
        rs = list(d.values())
        res[f'{m}|{t}'] = {'n': len(rs), 'off': boot([r['off_growth'] for r in rs]), 'on': boot([r['on_growth'] for r in rs]),
                           'share': boot([share(r) for r in rs])}
        q = res[f'{m}|{t}']; fm = lambda c: f'{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]'
        L.append(f"| {m} | {t} | {q['n']} | {fm(q['off'])} | {fm(q['on'])} | {q['share'][0]:+.2f} [{q['share'][1]:+.2f}, {q['share'][2]:+.2f}] |")
    if ('CausVid', 'none') in by and ('Self-Forcing', 'none') in by:
        a_ = np.array([share(r) for r in by[('CausVid', 'none')].values()]); b_ = np.array([share(r) for r in by[('Self-Forcing', 'none')].values()])
        bs = [a_[rng.integers(0, len(a_), len(a_))].mean() - b_[rng.integers(0, len(b_), len(b_))].mean() for _ in range(4000)]
        res['SP1_detail'] = [float(a_.mean() - b_.mean())] + [float(v) for v in np.percentile(bs, [2.5, 97.5])]
        res['SP1'] = bool(res['SP1_detail'][1] > 0)
    if ('CausVid', 'none') in by and ('CausVid', 'reenc') in by:
        n_, r_ = by[('CausVid', 'none')], by[('CausVid', 'reenc')]
        ids = sorted(set(n_) & set(r_))
        d = [(n_[i]['off_growth'] - r_[i]['off_growth']) - (n_[i]['on_growth'] - r_[i]['on_growth']) for i in ids]
        res['SP2_detail'] = boot(d) + [len(ids)]
        res['SP2'] = bool(res['SP2_detail'][1] > 0)
    L += ['', '## Pre-registered', '']
    for k in ('SP1', 'SP2'):
        if k in res:
            L.append(f"- {k}: {'PASS' if res[k] else 'FAIL'}  {res[k + '_detail']}")
    json.dump(res, open(os.path.join(OUT, 'split.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'SPLIT.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

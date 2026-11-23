"""FINAL video report -> <out>/FINAL_VIDEO.md + figures.  Criteria fixed before the run (see slurm/final.sh).
Unit = (video, sampling seed, prompt). A 'method' pools its seed variants (arm and arm_s1).
Primary statistic: Frechet distance (FD) of the LAST 128 frames to real Minecraft frames, ResNet-50 features (PCA-64),
ratio method / base with a paired, video-stratified bootstrap over units (2000 draws).  Secondary: ViT-B/16 features,
and KID (unbiased, polynomial kernel) for both extractors.
  F1 primary: reenc4_lf / base FD ratio (ResNet) <= 0.85 AND its 95% CI upper bound < 1; same direction (CI < 1) with ViT
  F2 colour:  colour deviation d (|log brightness| + |log saturation|, vs the prompt's own video) of reenc4_lf
              <= base + 0.10
  F3 control: a random nudge of the same size (noise4) does NOT give the gain: noise4 ratio CI includes 1 or its ratio
              is >= 0.10 above reenc4_lf's
  F4 control preserved: action-responsiveness (corr of camera/movement action magnitude with frame change) of reenc4_lf
              >= 0.8 x base
HEADLINE: F1 and F2 and F3 and F4.  Reported (no pass/fail): dose-response over re-encoding period, event-triggered and
mean-preserving variants, the model's own stabilization knob, compute overhead, drift-signal vs quality relationship."""
import json, glob, os, sys, re, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_final'
rng = np.random.default_rng(0)
runs = {}
for f in sorted(glob.glob(f'{out}/*_v[0-9].json')):
    r = json.load(open(f))
    arm = r['arm']; m = re.match(r'(.*)_s(\d+)$', arm)
    meth, seed = (m.group(1), int(m.group(2))) if m else (arm, 0)
    fz = f[:-5] + '_feats.npz'
    if os.path.exists(fz):
        r['_F'] = np.load(fz)
    runs[(meth, r['video_idx'], seed)] = r
if not runs:
    print('no results in', out); raise SystemExit
Rf = np.load(f'{out}/ref_feats.npz')
from sklearn.decomposition import PCA
P = {n: PCA(64).fit(Rf[n]) for n in ('resnet', 'vit')}
R64 = {n: P[n].transform(Rf[n]) for n in P}
mu_r = {n: R64[n].mean(0) for n in P}; C_r = {n: np.cov(R64[n], rowvar=False) for n in P}


def fd_to_ref(X, n):
    mu, C = X.mean(0), np.cov(X, rowvar=False)
    tr = np.sqrt(np.clip(np.linalg.eigvals(C @ C_r[n]).real, 0, None)).sum()
    return float(((mu - mu_r[n]) ** 2).sum() + np.trace(C) + np.trace(C_r[n]) - 2 * tr)


def kid_to_ref(X, n, m=400):
    Y = R64[n][rng.choice(len(R64[n]), min(m, len(R64[n])), replace=False)]
    X = X[rng.choice(len(X), min(m, len(X)), replace=False)]
    d = X.shape[1]; k = lambda a, b: (a @ b.T / d + 1) ** 3
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    nx, ny = len(X), len(Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1)) + (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1)) - 2 * Kxy.mean())


# units: per (method, video, seed) the last-128-frame PCA features per prompt
def unit_feats(meth, n, lastk=128):
    U = {}
    for (m_, v, s), r in runs.items():
        if m_ == meth and '_F' in r:
            F = r['_F'][n].astype(np.float32)[:, -lastk:]           # [B, k, D]
            for b in range(F.shape[0]):
                U[(v, s, b)] = P[n].transform(F[b])
    return U


meths = sorted({k[0] for k in runs})
vids = sorted({k[1] for k in runs})
base_u = {n: unit_feats('base', n) for n in P}


def boot_ratio(meth, n, metric, B=2000, ref='base'):
    U, Ub = unit_feats(meth, n), (base_u[n] if ref == 'base' else unit_feats(ref, n))
    keys = sorted(set(U) & set(Ub))
    if not keys:
        return nan3
    f = fd_to_ref if metric == 'fd' else kid_to_ref
    point = f(np.concatenate([U[k] for k in keys]), n) / f(np.concatenate([Ub[k] for k in keys]), n)
    byv = {v: [k for k in keys if k[0] == v] for v in vids}
    draws = []
    for _ in range(B if metric == 'fd' else 300):
        ks = [kk[i] for kk in byv.values() if kk for i in rng.integers(0, len(kk), len(kk))]
        draws.append(f(np.concatenate([U[k] for k in ks]), n) / f(np.concatenate([Ub[k] for k in ks]), n))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


nan3 = (float('nan'),) * 3
last = lambda arr: np.asarray(arr)[:, -64:].mean(1)        # per prompt


def per_prompt(meth, key):
    xs = []
    for (m_, v, s), r in runs.items():
        if m_ == meth:
            if key == 'd':
                rv = r['ref_video']
                xs += list(np.abs(np.log(last(r['bright']) / rv['bright'])) + np.abs(np.log(last(r['sat']) / rv['sat'])))
            elif key == 'sharp':
                xs += list(last(r['sharp']) / r['ref_video']['sharp'])
            elif key == 'resp':
                xs += list(r['resp'])
            elif key == 'shift':
                xs += list(last(r['shift']))
    return np.array(xs)


def mci(x):
    if len(x) == 0:
        return float('nan'), float('nan')
    bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return float(x.mean()), float(np.std(bs) * 1.96)


secs = lambda meth: np.mean([r['seconds_rollout'] for (m_, v, s), r in runs.items() if m_ == meth])
order = ['base', 'reenc4_lf', 'reenc4', 'noise4', 'sink4', 'sink4_reenc4', 'sink4_noise4', 'sink1', 'sink1_reenc4', 'reenc1_lf', 'reenc8_lf', 'reenc16_lf', 'evt2_lf', 'reenc4_mp', 'stab50']
order = [m for m in order if m in meths] + [m for m in meths if m not in order]
T = {}
L = ['# Course correction in a streaming video world model (Open-Oasis 500M) — final evidence', '',
     f'{len(vids)} videos; units = video x seed x prompt; last 128 of {runs[next(iter(runs))]["frames"]} frames '
     f'(training window 32). Ratios < 1 = closer to real frames than the uncorrected model. 95% paired bootstrap CIs.', '',
     '| method | units | FD ratio ResNet [95% CI] | FD ratio ViT [95% CI] | KID ratio ResNet [CI] | KID ratio ViT [CI] | colour dev d (base-rel) | sharpness vs video | action resp. | drift signal last | rollout time vs base |',
     '|---|---|---|---|---|---|---|---|---|---|---|']
sb = secs('base')
for m in order:
    nu = sum(r['_F']['resnet'].shape[0] for (m_, v, s), r in runs.items() if m_ == m and '_F' in r)
    T[m] = {k: (boot_ratio(m, n, k2) if m != 'base' else (1.0, 1.0, 1.0))
            for k, (n, k2) in {'fd_r': ('resnet', 'fd'), 'fd_v': ('vit', 'fd'), 'kid_r': ('resnet', 'kid'), 'kid_v': ('vit', 'kid')}.items()}
    d, dci = mci(per_prompt(m, 'd')); sh, _ = mci(per_prompt(m, 'sharp')); rs, rci = mci(per_prompt(m, 'resp'))
    dr, _ = mci(per_prompt(m, 'shift'))
    T[m].update(d=d, resp=rs, sharp=sh)
    c = lambda t: f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
    L.append(f"| {m} | {nu} | {c(T[m]['fd_r'])} | {c(T[m]['fd_v'])} | {c(T[m]['kid_r'])} | {c(T[m]['kid_v'])} | "
             f"{d:.2f} +- {dci:.2f} | {sh:.2f} | {rs:+.2f} +- {rci:.2f} | {dr:.4f} | x{secs(m) / sb:.2f} |")
floor = np.mean([r['shift_floor_real'] for r in runs.values()])
L += ['', f'Real-frame drift-signal floor: {floor:.4f}.', '']
# per-video primary ratio
L += ['## Primary ratio per video (ResNet FD, last 128 frames)', '', '| method | ' + ' | '.join(f'video {v}' for v in vids) + ' |',
      '|---|' + '---|' * len(vids)]
for m in order:
    if m == 'base':
        continue
    cells = []
    for v in vids:
        U, Ub = unit_feats(m, 'resnet'), base_u['resnet']
        ks = [k for k in U if k[0] == v and k in Ub]
        cells.append(f"{fd_to_ref(np.concatenate([U[k] for k in ks]), 'resnet') / fd_to_ref(np.concatenate([Ub[k] for k in ks]), 'resnet'):.2f}" if ks else '-')
    L.append(f'| {m} | ' + ' | '.join(cells) + ' |')
# drift signal vs quality: per bin, all methods/videos
xs, ys = [], []
for (m_, v, s), r in runs.items():
    sh = np.mean(r['shift'], 0)
    for i, b0 in enumerate(r['fd_bins']):
        xs.append(sh[b0:b0 + 128].mean()); ys.append(r['fd'][i])
rho = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float('nan')
L += ['', f'Drift signal vs quality: correlation between the mean re-encode shift of a 128-frame bin and that bin\'s FD, '
          f'across all runs and bins: r = {rho:+.2f} (n = {len(xs)}). (In MuJoCo the same signal was blind: |r| < 0.1.)', '']
# criteria
ok = {}
if 'reenc4_lf' in T:
    t = T['reenc4_lf']
    ok['F1'] = t['fd_r'][0] <= 0.85 and t['fd_r'][2] < 1 and t['fd_v'][2] < 1
    ok['F2'] = t['d'] <= T['base']['d'] + 0.10
    if 'noise4' in T:
        n4 = T['noise4']['fd_r']; ok['F3'] = (n4[1] <= 1 <= n4[2]) or (n4[0] >= t['fd_r'][0] + 0.10)
    ok['F4'] = t['resp'] >= 0.8 * T['base']['resp'] if T['base']['resp'] > 0 else t['resp'] >= T['base']['resp'] - 0.05
# ---- COMBINATION (attention sink x re-encoding). Criteria fixed before the combo run (slurm/combo.sh):
#   C1 primary: sink4_reenc4 / sink4 FD ratio (ResNet) CI upper < 1 (re-encoding still helps on top of the sink)
#               AND sink4_reenc4 / base point ratio < reenc4 / base point ratio (the pair beats either part)
#   C2 control: sink4_noise4 / sink4 CI includes 1 or ratio >= sink4_reenc4/sink4 + 0.10
#   C3 control preserved: action responsiveness of sink4_reenc4 >= 0.8 x base
if 'sink4_reenc4' in meths and 'sink4' in meths:
    L += ['## Combination: attention sink x re-encoding', '', '| comparison | FD ratio ResNet [CI] | FD ratio ViT [CI] | KID ratio ResNet [CI] |', '|---|---|---|---|']
    CT = {}
    for m, r_ in [('sink4', 'base'), ('sink1', 'base'), ('sink4_reenc4', 'base'), ('sink1_reenc4', 'base'),
                  ('sink4_reenc4', 'sink4'), ('sink1_reenc4', 'sink1'), ('sink4_noise4', 'sink4'), ('sink4_reenc4', 'reenc4'), ('sink4_reenc4', 'sink4_noise4')]:
        if m in meths and r_ in meths:
            CT[(m, r_)] = [boot_ratio(m, n, k2, ref=r_) for n, k2 in (('resnet', 'fd'), ('vit', 'fd'), ('resnet', 'kid'))]
            c = lambda t: f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
            L.append(f'| {m} / {r_} | ' + ' | '.join(c(t) for t in CT[(m, r_)]) + ' |')
    pr = CT.get(('sink4_reenc4', 'sink4'))
    if pr and 'reenc4' in T:
        ok['C1'] = pr[0][2] < 1 and T['sink4_reenc4']['fd_r'][0] < T['reenc4']['fd_r'][0]
    if ('sink4_noise4', 'sink4') in CT and pr:
        n_ = CT[('sink4_noise4', 'sink4')][0]; ok['C2'] = (n_[1] <= 1 <= n_[2]) or n_[0] >= pr[0][0] + 0.10
    if 'sink4_reenc4' in T:
        ok['C3'] = T['sink4_reenc4']['resp'] >= 0.8 * T['base']['resp']
    L += ['', f"**COMBINATION (C1-C3): {'SUPPORTED' if all(ok.get(k, False) for k in ('C1', 'C2', 'C3')) else 'NOT SUPPORTED'}**", '']
L += ['## Pre-registered criteria', ''] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in ok.items()]
L += ['', f"**HEADLINE (F1-F4): {'SUPPORTED' if all(ok.get(k, False) for k in ('F1', 'F2', 'F3', 'F4')) else 'NOT SUPPORTED'}**", '']
# figures
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = [m for m in ['base', 'reenc4', 'reenc4_lf', 'noise4', 'stab50', 'sink4', 'sink4_reenc4'] if m in meths]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m in show:
        rs_ = [r for (m_, v, s), r in runs.items() if m_ == m]
        bins = rs_[0]['fd_bins']
        fdm = np.array([r['fd'] for r in rs_]); ax[0].plot(np.array(bins) + 64, fdm.mean(0), marker='o', label=m)
        ax[0].fill_between(np.array(bins) + 64, fdm.min(0), fdm.max(0), alpha=0.12)
        sh_ = np.concatenate([np.asarray(r['shift'])[:, 1:] for r in rs_]); ax[1].plot(sh_.mean(0), label=m, lw=1)
        br = np.concatenate([np.asarray(r['bright']) / r['ref_video']['bright'] for r in rs_]); ax[2].plot(br.mean(0), label=m, lw=1)
    ax[1].axhline(floor, ls='--', c='k', lw=0.8, label='real-frame floor')
    ax[2].axhline(1.0, ls='--', c='k', lw=0.8)
    for a_, t_ in zip(ax, ['FD to real frames (ResNet), per 128-frame bin', 'Drift signal ||E(D(z)) - z||^2 / ||z||^2',
                           'Brightness / prompt video']):
        a_.set_title(t_, fontsize=10); a_.set_xlabel('frame'); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{out}/fig_main.png', dpi=150)
    dose = [m for m in ['reenc1_lf', 'reenc4_lf', 'reenc8_lf', 'reenc16_lf'] if m in T]
    if dose:
        plt.figure(figsize=(5, 3.5))
        xs_ = [int(re.findall(r'\d+', m)[0]) for m in dose]
        plt.errorbar(xs_, [T[m]['fd_r'][0] for m in dose], yerr=[[T[m]['fd_r'][0] - T[m]['fd_r'][1] for m in dose],
                     [T[m]['fd_r'][2] - T[m]['fd_r'][0] for m in dose]], marker='o', capsize=3)
        plt.axhline(1, ls='--', c='k', lw=0.8); plt.xscale('log', base=2); plt.xlabel('re-encode every P frames')
        plt.ylabel('FD ratio vs base (last 128 frames)'); plt.title('Dose-response'); plt.tight_layout()
        plt.savefig(f'{out}/fig_dose.png', dpi=150)
    from PIL import Image
    strips = [Image.open(f'{out}/{m}_v0_strip.jpg') for m in ['base', 'reenc4_lf'] if os.path.exists(f'{out}/{m}_v0_strip.jpg')]
    if len(strips) == 2:
        w = max(s_.width for s_ in strips); canvas = Image.new('RGB', (w, sum(s_.height for s_ in strips)))
        y = 0
        for s_ in strips:
            canvas.paste(s_, (0, y)); y += s_.height
        canvas.save(f'{out}/fig_frames_base_vs_corrected.jpg', quality=85)
    L.append('Figures: fig_main.png (FD / drift signal / brightness over time), fig_dose.png, fig_frames_base_vs_corrected.jpg '
             '(top: uncorrected, bottom: reenc4_lf; frames 0, T/4, T/2, 3T/4, T-1).')
except Exception as e:
    L.append(f'(figures failed: {e})')
txt = '\n'.join(L); open(f'{out}/FINAL_VIDEO.md', 'w').write(txt); print(txt)

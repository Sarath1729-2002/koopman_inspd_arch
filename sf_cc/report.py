"""Self-Forcing report -> results_sf/SF_REPORT.md + figures.  Criteria fixed BEFORE the run:
Windows: first 10 s and last 10 s of each 60 s video; paired over prompts (same seed and noise per prompt across arms);
95% bootstrap CIs over prompts (5000 draws).
  S1 primary: reenc1 vs base, last-10 s aesthetic difference > 0 with CI excluding 0, AND the aesthetic drop
     (first 10 s - last 10 s) shrinks by >= 30%
  S2 no collapse: reenc1 last-10 s motion (flow) >= 0.8 x base AND last-10 s CLIP text alignment >= base - 0.5
  S3 control: noise1 vs base last-10 s aesthetic difference CI includes 0 or is below 0
  S4 stacks with an attention sink: sink3_reenc1 vs sink3 last-10 s aesthetic CI > 0
HEADLINE: S1 and S2 and S3 (S4 reported).  Also reported: reenc2, DINO consistency, sharpness, brightness, overhead,
drift signal over time."""
import glob, json, os, sys, numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else 'results_sf'
rng = np.random.default_rng(0)
arms = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
D = {}
for arm in arms:
    for fj in glob.glob(f'{root}/{arm}/*.metrics.json'):
        idx = int(os.path.basename(fj)[:4])
        m = json.load(open(fj)); meta = json.load(open(fj.replace('.metrics.json', '.json')))
        D.setdefault(arm, {})[idx] = (m, meta)
if 'base' not in D:
    print('no base results'); raise SystemExit
KEYS = ['aesthetic', 'clip', 'dino_first', 'dino_adj', 'flow', 'sharp', 'bright', 'sat']


def win(m, k, which):
    t = np.asarray(m['t_sec']); v = np.asarray(m[k]); T = t.max()
    sel = t < 10 if which == 'first' else t >= T - 10
    return float(v[sel].mean())


def paired(arm, k, which, ref='base'):
    common = sorted(set(D.get(arm, {})) & set(D.get(ref, {})))
    if not common:
        return np.nan, np.nan, np.nan, 0
    f = (lambda m: win(m, k, 'last') - win(m, k, 'first')) if which == 'drift' else (lambda m: win(m, k, which))
    d = np.array([f(D[arm][i][0]) - f(D[ref][i][0]) for i in common])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(d)


def mean_win(arm, k, which):
    return float(np.mean([win(m, k, which) for m, _ in D[arm].values()]))


order = [a for a in ['base', 'reenc1', 'reenc2', 'noise1', 'sink3', 'sink3_reenc1'] if a in D] + [a for a in arms if a not in
         ['base', 'reenc1', 'reenc2', 'noise1', 'sink3', 'sink3_reenc1'] and a in D]
L = ['# Course correction in Self-Forcing (Wan2.1-T2V-1.3B), 60 s videos (~3x the training length)', '',
     f'Prompts: MovieGenVideoBench (extended prompts for generation, short prompts for CLIP scoring); '
     f'n = {len(D["base"])} paired videos per arm. Windows = first / last 10 s.', '',
     '| arm | n | aesthetic first -> last | CLIP first -> last | DINO-to-first last | motion (flow) last | sharpness last | brightness last | gen time vs base | drift signal first -> last |',
     '|---|---|---|---|---|---|---|---|---|---|']
gt_base = np.mean([meta['gen_seconds'] for _, meta in D['base'].values()])
for a in order:
    sh = [np.asarray(meta['shift']) for _, meta in D[a].values() if meta['shift']]
    s0 = np.mean([s[0, 1] for s in sh]) if sh else np.nan; s1 = np.mean([s[-1, 1] for s in sh]) if sh else np.nan
    gt = np.mean([meta['gen_seconds'] for _, meta in D[a].values()])
    L.append(f"| {a} | {len(D[a])} | {mean_win(a, 'aesthetic', 'first'):.3f} -> {mean_win(a, 'aesthetic', 'last'):.3f} | "
             f"{mean_win(a, 'clip', 'first'):.2f} -> {mean_win(a, 'clip', 'last'):.2f} | {mean_win(a, 'dino_first', 'last'):.3f} | "
             f"{mean_win(a, 'flow', 'last'):.2f} | {mean_win(a, 'sharp', 'last') * 1e3:.2f}e-3 | {mean_win(a, 'bright', 'last'):.3f} | "
             f"x{gt / gt_base:.2f} | {s0:.4f} -> {s1:.4f} |")
L += ['', '## Paired differences vs base (mean [95% CI] over prompts)', '',
      '| arm | aesthetic last 10 s | aesthetic drift (last - first) | CLIP last | motion last | DINO-to-first last |', '|---|---|---|---|---|---|']
fmt = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]'
for a in order:
    if a == 'base':
        continue
    L.append(f"| {a} | {fmt(paired(a, 'aesthetic', 'last'))} | {fmt(paired(a, 'aesthetic', 'drift'))} | "
             f"{fmt(paired(a, 'clip', 'last'))} | {fmt(paired(a, 'flow', 'last'))} | {fmt(paired(a, 'dino_first', 'last'))} |")
ok = {}
if 'reenc1' in D:
    d_last = paired('reenc1', 'aesthetic', 'last')
    drop_b = mean_win('base', 'aesthetic', 'first') - mean_win('base', 'aesthetic', 'last')
    drop_r = mean_win('reenc1', 'aesthetic', 'first') - mean_win('reenc1', 'aesthetic', 'last')
    ok['S1'] = d_last[1] > 0 and (drop_b <= 0 or drop_r <= 0.7 * drop_b)
    ok['S2'] = mean_win('reenc1', 'flow', 'last') >= 0.8 * mean_win('base', 'flow', 'last') and \
        mean_win('reenc1', 'clip', 'last') >= mean_win('base', 'clip', 'last') - 0.5
    L += ['', f'Aesthetic drop first -> last 10 s: base {drop_b:+.3f}, reenc1 {drop_r:+.3f}.']
if 'noise1' in D:
    n = paired('noise1', 'aesthetic', 'last'); ok['S3'] = n[1] <= 0
if 'sink3_reenc1' in D and 'sink3' in D:
    s = paired('sink3_reenc1', 'aesthetic', 'last', ref='sink3'); ok['S4'] = s[1] > 0
    L.append(f'Stacking with an attention sink: sink3_reenc1 - sink3, last-10 s aesthetic {fmt(s)}.')
L += ['', '## Pre-registered criteria', ''] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in ok.items()]
L += ['', f"**HEADLINE (S1, S2, S3): {'SUPPORTED' if all(ok.get(k, False) for k in ('S1', 'S2', 'S3')) else 'NOT SUPPORTED'}**", '']
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a in order:
        for j, k in enumerate(['aesthetic', 'flow']):
            V = np.array([np.asarray(m[k]) for m, _ in D[a].values()]); t = np.asarray(next(iter(D[a].values()))[0]['t_sec'])
            ax[j].plot(t, V.mean(0), label=a, lw=1.2)
        sh = [np.asarray(meta['shift']) for _, meta in D[a].values() if meta['shift']]
        if sh:
            n_ = min(len(s) for s in sh); S = np.stack([s[:n_] for s in sh])
            ax[2].plot(S[0, :, 0] * 0.75, S[:, :, 1].mean(0), label=a, lw=1.2)
    for a_, t_ in zip(ax, ['LAION aesthetic over time', 'Motion (RAFT flow) over time', 'Drift signal ||E(D(z)) - z||^2/||z||^2']):
        a_.set_title(t_, fontsize=10); a_.set_xlabel('seconds'); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{root}/fig_sf.png', dpi=150); L.append('Figure: fig_sf.png')
except Exception as e:
    L.append(f'(figure failed: {e})')
txt = '\n'.join(L); open(f'{root}/SF_REPORT.md', 'w').write(txt); print(txt)

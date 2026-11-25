"""Self-Forcing report -> results_sf/SF_REPORT.md + figures.  Criteria fixed BEFORE the run:
Windows: first 10 s and last 10 s of each 60 s video; paired over prompts (same seed and noise per prompt across arms);
95% bootstrap CIs over prompts (5000 draws).
  S1 primary: reenc1 vs base, last-10 s aesthetic difference > 0 with CI excluding 0, AND the aesthetic drop
     (first 10 s - last 10 s) shrinks by >= 30%
  S2 no collapse: reenc1 last-10 s motion (flow) >= 0.8 x base AND last-10 s CLIP text alignment >= base - 0.5
  S3 control: noise1 vs base last-10 s aesthetic difference CI includes 0 or is below 0
  S4 stacks with an attention sink: sink3_reenc1 vs sink3 last-10 s aesthetic CI > 0
HEADLINE: S1 and S2 and S3 (S4 reported).
Added after the first run (hypothesis: correction substitutes for training on own rollouts), fixed before the ODE run:
  O1: ode_reenc1 vs ode_base (same architecture, never trained on own outputs), last-10 s aesthetic CI > 0
  O2 control: ode_noise1 vs ode_base, last-10 s aesthetic CI includes 0 or is below 0
  O3 contrast: (ode_reenc1 - ode_base) - (reenc1 - base) > 0 with CI excluding 0 (correction helps more without
     self-rollout training)  Also reported: reenc2, DINO consistency, sharpness, brightness, overhead,
drift signal over time."""
import glob, json, os, sys, numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else 'results_sf'
rng = np.random.default_rng(0)
arms = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
D = {}
for arm in arms:
    for fj in glob.glob(f'{root}/{arm}/*.metrics.json'):
        idx = int(os.path.basename(fj)[:4])
        fm = fj.replace('.metrics.json', '.json')
        if not os.path.exists(fm):   # video from an interrupted run (no meta): skip it
            print('skip (no meta):', fj); continue
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


KNOWN = ['base', 'reenc1', 'reenc2', 'noise1', 'sink3', 'sink3_reenc1', 'sink3_reenc1_cache', 'sink3_reenc2_cache',
         'ode_base', 'ode_reenc1', 'ode_reenc2', 'ode_noise1',
         'causvid_base', 'causvid_reenc1', 'causvid_reenc2', 'causvid_noise1', 'causvid_sink3', 'causvid_sink3_reenc1']
order = [a for a in KNOWN if a in D] + [a for a in arms if a not in KNOWN and a in D]
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
ODE = [a for a in ('ode_reenc1', 'ode_reenc2', 'ode_noise1') if a in D]
if 'ode_base' in D and ODE:
    L += ['', '## Same model before self-rollout training (ode_init.pt), paired vs ode_base', '',
          '| arm | aesthetic last 10 s | aesthetic drift (last - first) | CLIP last | motion last | DINO-to-first last |', '|---|---|---|---|---|---|']
    for a in ODE:
        L.append(f"| {a} | {fmt(paired(a, 'aesthetic', 'last', 'ode_base'))} | {fmt(paired(a, 'aesthetic', 'drift', 'ode_base'))} | "
                 f"{fmt(paired(a, 'clip', 'last', 'ode_base'))} | {fmt(paired(a, 'flow', 'last', 'ode_base'))} | "
                 f"{fmt(paired(a, 'dino_first', 'last', 'ode_base'))} |")
    L.append(f"ode_base vs base (what self-rollout training buys): last-10 s aesthetic {fmt(paired('ode_base', 'aesthetic', 'last'))}")
    if 'ode_reenc1' in D:
        ok['O1'] = paired('ode_reenc1', 'aesthetic', 'last', 'ode_base')[1] > 0
        cm = sorted(set.intersection(*[set(D[a]) for a in ('ode_reenc1', 'ode_base', 'reenc1', 'base')]))
        w = lambda a, i: win(D[a][i][0], 'aesthetic', 'last')
        dd = np.array([(w('ode_reenc1', i) - w('ode_base', i)) - (w('reenc1', i) - w('base', i)) for i in cm])
        if len(dd):
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
            ok['O3'] = lo > 0
            L.append(f'Contrast (ode_reenc1 - ode_base) - (reenc1 - base), last-10 s aesthetic: {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(dd)})')
    if 'ode_noise1' in D:
        ok['O2'] = paired('ode_noise1', 'aesthetic', 'last', 'ode_base')[1] <= 0
# K1 (fixed before the cache run): cache-only correction beats the sink alone: sink3_reencP_cache vs sink3, last-10 s
#    aesthetic CI > 0 for P = 1 or 2 (the viewer sees the raw latents; only the attended context is corrected)
CA = [a for a in ('sink3_reenc1_cache', 'sink3_reenc2_cache') if a in D]
if CA and 'sink3' in D:
    L += ['', '## Cache-only correction, paired vs sink3 (attention sink alone)', '',
          '| arm | aesthetic last 10 s | CLIP last | motion last | DINO-to-first last |', '|---|---|---|---|---|']
    for a in CA:
        L.append(f"| {a} | {fmt(paired(a, 'aesthetic', 'last', 'sink3'))} | {fmt(paired(a, 'clip', 'last', 'sink3'))} | "
                 f"{fmt(paired(a, 'flow', 'last', 'sink3'))} | {fmt(paired(a, 'dino_first', 'last', 'sink3'))} |")
    ok['K1'] = any(paired(a, 'aesthetic', 'last', 'sink3')[1] > 0 for a in CA)
# V1-V3 (fixed before the CausVid run; adds non-collapse guards after the ODE-init lesson):
#   V1 causvid_reenc1 vs causvid_base: last-10 s aesthetic CI > 0 AND CLIP last >= -1.0 AND DINO-to-first last >= -0.05
#      (mean paired differences) AND motion last <= 2 x causvid_base's (no freeze / no flicker trade)
#   V2 control causvid_noise1 vs causvid_base: aesthetic CI includes 0 or is below 0
#   V3 stacks with the sink: causvid_sink3_reenc1 vs causvid_sink3 last-10 s aesthetic CI > 0
CV = [a for a in ('causvid_reenc1', 'causvid_reenc2', 'causvid_noise1', 'causvid_sink3', 'causvid_sink3_reenc1') if a in D]
if 'causvid_base' in D and CV:
    L += ['', '## CausVid (teacher-forced DMD student, same Wan-1.3B), paired vs causvid_base', '',
          '| arm | aesthetic last 10 s | CLIP last | motion last | DINO-to-first last |', '|---|---|---|---|---|']
    for a in CV:
        L.append(f"| {a} | {fmt(paired(a, 'aesthetic', 'last', 'causvid_base'))} | {fmt(paired(a, 'clip', 'last', 'causvid_base'))} | "
                 f"{fmt(paired(a, 'flow', 'last', 'causvid_base'))} | {fmt(paired(a, 'dino_first', 'last', 'causvid_base'))} |")
    if 'causvid_reenc1' in D:
        r = 'causvid_reenc1'
        ok['V1'] = paired(r, 'aesthetic', 'last', 'causvid_base')[1] > 0 and paired(r, 'clip', 'last', 'causvid_base')[0] >= -1.0 \
            and paired(r, 'dino_first', 'last', 'causvid_base')[0] >= -0.05 and mean_win(r, 'flow', 'last') <= 2 * mean_win('causvid_base', 'flow', 'last')
    if 'causvid_noise1' in D:
        ok['V2'] = paired('causvid_noise1', 'aesthetic', 'last', 'causvid_base')[1] <= 0
    if 'causvid_sink3_reenc1' in D and 'causvid_sink3' in D:
        s3 = paired('causvid_sink3_reenc1', 'aesthetic', 'last', 'causvid_sink3'); ok['V3'] = s3[1] > 0
        L.append(f'causvid_sink3_reenc1 vs causvid_sink3, last-10 s aesthetic: {fmt(s3)}')
    # G (fixed before the run): does the cheap teacher-forced model + inference-time fixes MATCH self-rollout training?
    #   compare to Self-Forcing (same architecture, trained on its own rollouts), paired by prompt (same prompt & seed)
    L += ['', '### CausVid + inference-time fixes vs Self-Forcing (the self-rollout-trained model), paired by prompt', '',
          '| CausVid arm | vs Self-Forcing arm | aesthetic last 10 s | CLIP last | DINO-to-first last | motion last |', '|---|---|---|---|---|---|']
    for a, r_ in (('causvid_base', 'base'), ('causvid_reenc1', 'base'), ('causvid_sink3', 'sink3'), ('causvid_sink3_reenc1', 'sink3')):
        if a in D and r_ in D:
            L.append(f"| {a} | {r_} | {fmt(paired(a, 'aesthetic', 'last', r_))} | {fmt(paired(a, 'clip', 'last', r_))} | "
                     f"{fmt(paired(a, 'dino_first', 'last', r_))} | {fmt(paired(a, 'flow', 'last', r_))} |")
    if 'causvid_sink3_reenc1' in D and 'sink3' in D:
        g = paired('causvid_sink3_reenc1', 'aesthetic', 'last', 'sink3'); gb = paired('causvid_sink3', 'aesthetic', 'last', 'sink3') if 'causvid_sink3' in D else (np.nan,) * 4
        ok['G'] = g[2] >= 0 and paired('causvid_sink3_reenc1', 'clip', 'last', 'sink3')[0] >= -1.0
        L.append(f"G: best CausVid pipeline not worse than Self-Forcing + sink (aesthetic CI upper >= 0, CLIP >= -1): "
                 f"{'PASS' if ok['G'] else 'FAIL'}; gap closed by re-encoding: {gb[0]:+.3f} -> {g[0]:+.3f}")
    L += ['', f"**CausVid test (V1, V2): {'SUPPORTED' if ok.get('V1') and ok.get('V2') else 'NOT SUPPORTED'}**"]
L += ['', '## Pre-registered criteria', ''] + [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in ok.items()]
if any(k in ok for k in ('O1', 'O2', 'O3')):
    L += ['', f"**ODE-init test (O1, O2, O3): {'SUPPORTED' if all(ok.get(k, False) for k in ('O1', 'O2', 'O3')) else 'NOT SUPPORTED'}**"]
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

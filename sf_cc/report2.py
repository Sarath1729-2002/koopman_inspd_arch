"""Campaign report (Nov 24): making course correction work in causal video generators -> <root>/SF_CAMPAIGN.md
Every arm is paired (same prompt, same seed) with its reference: the same model with the same attention sink and no
correction (sink3 for Self-Forcing, causvid_sink3 for CausVid; base / causvid_base for arms without a sink).
Windows = first / last 10 s; 95% bootstrap CIs over prompts. Criteria fixed BEFORE the run:
  NC  no-collapse guard (applies to every claimed win): CLIP last >= -0.5, DINO-to-first last >= -0.03,
      motion last >= 0.8 x reference
  G1  generator-manifold projection fixes the self-forced model: sink3_dproj_cache OR sink3_dproj vs sink3,
      last-10 s aesthetic CI > 0, with NC
  G2  control: sink3_dproj_dpn (random nudge of the projection's size) vs sink3: aesthetic CI includes 0 or is below 0
  V1  VAE projection helps the teacher-forced model once it drifts (4-min run, results_sf_long):
      causvid_sink3_reenc2_cache OR causvid_sink3_reenc2 vs causvid_sink3, last-10 s aesthetic CI > 0, with NC
  X   matching principle (60 s, cache-only, with sink): (SF dproj - SF reenc2) - (CausVid dproj - CausVid reenc2) > 0,
      last-10 s aesthetic, CI excluding 0 -- each model gains most from projecting onto the manifold its context
      was trained on (own outputs for Self-Forcing, VAE latents for CausVid)
Added Nov 24 after the first pass (sink3_dproj2_cache, a noisier re-noising level, was the best of 7 Self-Forcing arms
on prompts 0-15, +0.104 [+0.004, +0.216]); fixed BEFORE the confirmation run on FRESH prompts 16-31:
  R1  sink3_dproj2_cache vs sink3 on prompts 16-31: last-10 s aesthetic CI > 0, with NC
  R2  control sink3_dproj2_dpn vs sink3 on prompts 16-31: aesthetic CI includes 0 or is below 0
  R3  dose: sink3_dproj3_cache (noisier still) reported on all prompts, no pass/fail
Also reported, no pass/fail: partial projection (a50), drift trigger (evt2), statistics anchoring (stat), colour drift,
generation time, number of corrections, drift signal."""
import glob, json, os, sys, numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else 'results_sf'
rng = np.random.default_rng(0)
D = {}
for d in sorted(os.listdir(root)):
    for fj in glob.glob(f'{root}/{d}/*.metrics.json'):
        m = json.load(open(fj)); meta = json.load(open(fj.replace('.metrics.json', '.json')))
        D.setdefault(d, {})[int(os.path.basename(fj)[:4])] = (m, meta)


def win(m, k, which):
    t = np.asarray(m['t_sec']); v = np.asarray(m[k]); T = t.max()
    return float(v[(t < 10) if which == 'first' else (t >= T - 10)].mean())


def stat(m, k):
    if k == 'colour':   # |change in mean brightness| + |change in saturation| between first and last 10 s
        return abs(win(m, 'bright', 'last') - win(m, 'bright', 'first')) + abs(win(m, 'sat', 'last') - win(m, 'sat', 'first'))
    return win(m, k, 'last')


def paired(a, r, k, lo_idx=0):
    cm = sorted(i for i in set(D.get(a, {})) & set(D.get(r, {})) if i >= lo_idx)
    if not cm:
        return (np.nan, np.nan, np.nan, 0)
    d = np.array([stat(D[a][i][0], k) - stat(D[r][i][0], k) for i in cm])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (float(d.mean()), float(lo), float(hi), len(d))


def ref_of(a):
    pre = 'causvid_' if a.startswith('causvid') else ''
    return pre + ('sink3' if 'sink3' in a else 'base')


def mean_ref(a, r, k):
    cm = sorted(set(D.get(a, {})) & set(D.get(r, {})))
    return np.mean([stat(D[r][i][0], k) for i in cm]) if cm else np.nan


def nc(a, r):
    if a not in D or r not in D:
        return False
    return paired(a, r, 'clip')[0] >= -0.5 and paired(a, r, 'dino_first')[0] >= -0.03 and \
        paired(a, r, 'flow')[0] + mean_ref(a, r, 'flow') >= 0.8 * mean_ref(a, r, 'flow')


f = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]'
L = [f'# Making course correction work in causal video generators ({root})', '',
     'Each arm vs the same model, same sink, no correction; paired over prompts (same seed). Last 10 s unless noted.', '',
     '| arm | vs | n | aesthetic | CLIP | DINO-to-first | motion | colour drift (lower = better) | gen time x ref | corrections / video | drift last |',
     '|---|---|---|---|---|---|---|---|---|---|---|']
for a in sorted(D):
    r = ref_of(a)
    if a == r or r not in D:
        continue
    gt = np.mean([m['gen_seconds'] for _, m in D[a].values()]) / np.mean([m['gen_seconds'] for _, m in D[r].values()])
    nco = np.mean([m.get('n_corrections', np.nan) for _, m in D[a].values()])
    dl = np.mean([m['shift'][-1][1] for _, m in D[a].values() if m['shift']])
    p = paired(a, r, 'aesthetic')
    L.append(f"| {a} | {r} | {p[3]} | {f(p)} | {f(paired(a, r, 'clip'))} | {f(paired(a, r, 'dino_first'))} | "
             f"{f(paired(a, r, 'flow'))} | {f(paired(a, r, 'colour'))} | x{gt:.2f} | {nco:.0f} | {dl:.4f} |")
ok = {}
g1 = [a for a in ('sink3_dproj_cache', 'sink3_dproj') if a in D]
if g1 and 'sink3' in D:
    ok['G1'] = any(paired(a, 'sink3', 'aesthetic')[1] > 0 and nc(a, 'sink3') for a in g1)
if 'sink3_dproj_dpn' in D and 'sink3' in D:
    ok['G2'] = paired('sink3_dproj_dpn', 'sink3', 'aesthetic')[1] <= 0
v1 = [a for a in ('causvid_sink3_reenc2_cache', 'causvid_sink3_reenc2') if a in D]
if v1 and 'causvid_sink3' in D and 'long' in root:
    ok['V1'] = any(paired(a, 'causvid_sink3', 'aesthetic')[1] > 0 and nc(a, 'causvid_sink3') for a in v1)
X = ('sink3_dproj_cache', 'sink3_reenc2_cache', 'causvid_sink3_dproj_cache', 'causvid_sink3_reenc2_cache')
if all(a in D for a in X):
    cm = sorted(set.intersection(*[set(D[a]) for a in X]))
    w = lambda a, i: stat(D[a][i][0], 'aesthetic')
    d = np.array([(w(X[0], i) - w(X[1], i)) - (w(X[2], i) - w(X[3], i)) for i in cm])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    ok['X'] = lo > 0
    L += ['', f'Matching principle X: (SF dproj - SF reenc2) - (CausVid dproj - CausVid reenc2) = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)})']
if 'sink3_dproj2_cache' in D and 'sink3' in D:
    r1 = paired('sink3_dproj2_cache', 'sink3', 'aesthetic', 16)
    if r1[3]:
        nc16 = paired('sink3_dproj2_cache', 'sink3', 'clip', 16)[0] >= -0.5 and paired('sink3_dproj2_cache', 'sink3', 'dino_first', 16)[0] >= -0.03 \
            and paired('sink3_dproj2_cache', 'sink3', 'flow', 16)[0] >= -0.2 * np.mean([stat(D['sink3'][i][0], 'flow') for i in D['sink3'] if i >= 16])
        ok['R1'] = r1[1] > 0 and nc16
        L += ['', '## Confirmation on fresh prompts 16-31', '', '| arm | n | aesthetic | CLIP | DINO-to-first | motion | colour drift |', '|---|---|---|---|---|---|---|']
        for a in ('sink3_dproj2_cache', 'sink3_dproj2_dpn', 'sink3_dproj3_cache'):
            if a in D:
                p = paired(a, 'sink3', 'aesthetic', 16)
                L.append(f"| {a} | {p[3]} | {f(p)} | {f(paired(a, 'sink3', 'clip', 16))} | {f(paired(a, 'sink3', 'dino_first', 16))} | "
                         f"{f(paired(a, 'sink3', 'flow', 16))} | {f(paired(a, 'sink3', 'colour', 16))} |")
    if 'sink3_dproj2_dpn' in D and paired('sink3_dproj2_dpn', 'sink3', 'aesthetic', 16)[3]:
        ok['R2'] = paired('sink3_dproj2_dpn', 'sink3', 'aesthetic', 16)[1] <= 0
L += ['', '## Pre-registered criteria', ''] + [f'- {k}: {"PASS" if v else "FAIL"}' for k, v in ok.items()]
txt = '\n'.join(L); open(f'{root}/SF_CAMPAIGN.md', 'w').write(txt); print(txt)

"""Confirmation report -> results_wm/CONFIRM.md.  Criteria fixed BEFORE the run (after the exploration):
Correction = periodic decode->re-encode every 10 imagined steps (per10). Event triggers dropped (diagnostics: blind).
Planning (CEM-MPC, real env, 3 model seeds x 6 episodes per cell; 'cell' = env x model):
  P1 return:  per10 - none >= max(0.15*|none|, 2*SE_diff)                       [pooled over 18 episodes]
  P2 optimism: per10 gap <= 0.7 * none gap (none gap > 0)
  P3 vs ensemble (3 seeds, disagreement penalty): per10 (single model) >= ens/none
  P3b complementary: ens/per10 - ens/none >= max(0.15*|ens none|, 2*SE_diff)
  P4 vs short horizon: per10 > hp10
  A criterion PASSES if it holds in >= 2 of 3 envs for >= 4 of the 6 models.
Horizon substitution (open loop, steps 101-180, mean over 3 seeds):
  S1 kae_k10/per10 <= kae_k50/none ; koop_rdr10/per10 <= koop_rdr/none ; gru_rdr10/per10 <= gru_rdr/none  (>= 2 of 3 envs)
HEADLINE: P1 and P2."""
import json, glob, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_wm'
envs = ['cheetah', 'hopper', 'walker']
arms = ['gru_rdr', 'koop_rdr', 'gru_rdr10', 'koop_rdr10', 'kae_k10', 'kae_k50']
nan = float('nan')
P = {e: {} for e in envs}
for e in envs:
    for f in glob.glob(f'{out}/planc_{e}_*.json'):
        P[e].update(json.load(open(f)))


def pooled(e, arm, corr):
    if corr.startswith('ens'):
        x = P[e].get(f'{arm}/ens/{corr[4:]}')
        return (np.array(x['returns']), np.array(x['gaps'])) if x else (None, None)
    R, G = [], []
    for s in (0, 1, 2):
        x = P[e].get(f'{arm}/s{s}/{corr}')
        if x: R += x['returns']; G += x['gaps']
    return (np.array(R), np.array(G)) if R else (None, None)


def ms(x):
    return (x.mean(), x.std() / np.sqrt(len(x))) if x is not None and len(x) else (nan, nan)


L = ['# Confirmation: course correction in world-model planning and horizon substitution', '']
counts = {k: {e: 0 for e in envs} for k in ('P1', 'P2', 'P3', 'P3b', 'P4')}
for e in envs:
    L += [f'## Planning: {e} (random policy {P[e].get("random", {}).get("mean", nan):.1f})', '',
          '| model | none | per10 | hp10 | ens none | ens per10 | gap none -> per10 | P1 | P2 | P3 | P3b | P4 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        d = {c: pooled(e, arm, c) for c in ('none', 'per10', 'hp10', 'ens_none', 'ens_per10')}
        (mn, sn), (mp, sp_), (mh, _) = ms(d['none'][0]), ms(d['per10'][0]), ms(d['hp10'][0])
        (men, sen), (mep, sep) = ms(d['ens_none'][0]), ms(d['ens_per10'][0])
        gn, gp = (ms(d['none'][1])[0], ms(d['per10'][1])[0])
        p1 = mp - mn >= max(0.15 * abs(mn), 2 * np.hypot(sn, sp_))
        p2 = gn > 0 and gp <= 0.7 * gn
        p3 = mp >= men
        p3b = mep - men >= max(0.15 * abs(men), 2 * np.hypot(sen, sep))
        p4 = mp > mh
        for k, v in zip(('P1', 'P2', 'P3', 'P3b', 'P4'), (p1, p2, p3, p3b, p4)):
            counts[k][e] += bool(v)
        f = lambda m, s: f'{m:.1f} +- {s:.1f}'
        L.append(f"| {arm} | {f(mn, sn)} | {f(mp, sp_)} | {mh:.1f} | {f(men, sen)} | {f(mep, sep)} | {gn:.1f} -> {gp:.1f} | "
                 + ' | '.join('Y' if v else '.' for v in (p1, p2, p3, p3b, p4)) + ' |')
    L.append('')
verdict = {}
L += ['## Planning criteria (PASS = holds for >= 4 of 6 models in >= 2 of 3 envs)', '']
for k in counts:
    envs_ok = sum(counts[k][e] >= 4 for e in envs)
    verdict[k] = envs_ok >= 2
    L.append(f"- {k}: " + ', '.join(f'{e} {counts[k][e]}/6' for e in envs) + f" -> {'PASS' if verdict[k] else 'FAIL'}")

# substitution
LG = {e: [json.load(open(f)) for f in (f'{out}/eval_{e}_long.json', f'{out}/eval_{e}_long_s1.json', f'{out}/eval_{e}_long_s2.json')
          if os.path.exists(f)] for e in envs}
g = lambda e, k: float(np.mean([r[k]['b101_end'] for r in LG[e] if k in r])) if any(k in r for r in LG[e]) else nan
L += ['', '## Horizon substitution (open loop, steps 101-180, mean over seeds)', '']
s_ok = {}
for short, long_ in (('kae_k10', 'kae_k50'), ('koop_rdr10', 'koop_rdr'), ('gru_rdr10', 'gru_rdr')):
    hits = 0; parts = []
    for e in envs:
        a_, b_, c_ = g(e, f'{short}/per10'), g(e, f'{long_}/none'), g(e, f'{short}/none')
        hits += a_ <= b_
        parts.append(f'{e}: {short} none {c_:.3f} -> per10 {a_:.3f} vs {long_} none {b_:.3f}')
    s_ok[short] = hits >= 2
    L.append(f"- S1 {short}+per10 <= {long_}: {hits}/3 -> {'PASS' if hits >= 2 else 'FAIL'}   " + '; '.join(parts))
L += ['', f"**HEADLINE (P1 and P2): {'SUPPORTED' if verdict['P1'] and verdict['P2'] else 'NOT SUPPORTED'}**  "
      f"(vs ensemble P3 {'PASS' if verdict['P3'] else 'FAIL'}; complementary P3b {'PASS' if verdict['P3b'] else 'FAIL'}; "
      f"vs short horizon P4 {'PASS' if verdict['P4'] else 'FAIL'})", '']
txt = '\n'.join(L)
open(f'{out}/CONFIRM.md', 'w').write(txt)
print(txt)

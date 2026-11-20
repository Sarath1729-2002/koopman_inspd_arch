"""Exploratory report -> results_wm/EXPLORE.md : diagnostics, faithful Koopman AE, horizon substitution, planning.
Flags (fixed before the run; exploratory, not a single kill verdict):
  D  per model: 'projection helps' if proj_full/err < 0.9 at 101-180; 'on-manifold drift' if shift_full/err < 0.1;
     'projector is the bottleneck' if oracle_init10/err > 0.9 (even rebuilding from TRUE obs every 10 steps barely helps)
  F1 faithful Fathi setting: kae_k10/evt10 <= 0.85 x kae_k10/none at 101-180 (>= 2 of 3 envs)
  F2 substitution: short-trained + correction <= long-trained without:  kae_k10/evt10 <= kae_k50/none,
     gru_rdr10/{evt10,zevt10} <= gru_rdr/none, koop_rdr10/{evt10,zevt10} <= koop_rdr/none   (>= 2 of 3 envs)
  P  planning (cheetah): correction improves real return by >= max(15% of |none|, 2 SE) for the model."""
import json, glob, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else 'results_wm'
J = lambda f: json.load(open(f)) if os.path.exists(f) else {}
envs = ['cheetah', 'hopper', 'walker']
nan = float('nan')
L = ['# Exploration: why course correction fails in world models, and where it might not', '']

# ---------------- D: diagnostics ----------------
L += ['## D. Mechanism diagnostics (open-loop, steps 101-180; ratios relative to the uncorrected error)', '',
      '| env | model | err | proj_full/err | proj_z/err | shift_full/err | rec_true | oracle_init10/err | oracle_post10/err | corr(resid, growth) | flags |',
      '|---|---|---|---|---|---|---|---|---|---|---|']
for e in envs:
    D = J(f'{out}/diag_{e}.json')
    for arm, S in D.items():
        b = '101-end'
        err = S['err'][b]
        r = lambda k: S[k][b] / err
        flags = []
        if r('proj_full') < 0.9: flags.append('projection helps')
        if r('shift_full') < 0.1: flags.append('on-manifold drift')
        if r('oracle_init10') > 0.9: flags.append('projector is the bottleneck')
        L.append(f"| {e} | {arm} | {err:.4f} | {r('proj_full'):.2f} | {r('proj_z'):.2f} | {r('shift_full'):.3f} | "
                 f"{S['rec_true'][b]:.4f} | {r('oracle_init10'):.2f} | {r('oracle_post10'):.2f} | "
                 f"{S['corr_resid_future_err_growth']:+.2f} | {', '.join(flags)} |")
L.append('')

# ---------------- long-horizon table incl. new arms ----------------
LG = {e: J(f'{out}/eval_{e}_long.json') for e in envs}
g = lambda e, k, b='b101_end': LG[e].get(k, {}).get(b, nan)
arms = ['koop', 'gru', 'koop_rdr', 'gru_rdr', 'koop_rdr10', 'gru_rdr10', 'koop_rdrre', 'gru_rdrre', 'kae_k10', 'kae_k50']
L += ['## Long horizon (180 steps): error at steps 101-180, none vs selected corrections', '',
      '| env | model | none | per10 | per25 | evt10 | zevt10 | best (sched) | 26-50 none |', '|---|---|---|---|---|---|---|---|---|']
for e in envs:
    for arm in arms:
        if f'{arm}/none' not in LG[e]:
            continue
        c = [(v['b101_end'], k.split('/')[1]) for k, v in LG[e].items() if k.startswith(arm + '/')]
        best = min(c)
        L.append(f"| {e} | {arm} | {g(e, arm + '/none'):.4f} | {g(e, arm + '/per10'):.4f} | {g(e, arm + '/per25'):.4f} | "
                 f"{g(e, arm + '/evt10'):.4f} | {g(e, arm + '/zevt10'):.4f} | {best[0]:.4f} ({best[1]}) | "
                 f"{g(e, arm + '/none', 'b26_50'):.4f} |")
L.append('')

# ---------------- F1 / F2 ----------------
f1 = sum(g(e, 'kae_k10/evt10') <= 0.85 * g(e, 'kae_k10/none') for e in envs)
L += ['## F. Faithful Koopman AE and horizon substitution', '',
      f"- F1 faithful Fathi effect (kae_k10/evt10 <= 0.85x kae_k10/none): {f1}/3 -> {'PASS' if f1 >= 2 else 'FAIL'}  "
      + '; '.join(f"{e} x{g(e, 'kae_k10/evt10') / g(e, 'kae_k10/none'):.2f}" for e in envs)]
for short, long_, cc in (('kae_k10', 'kae_k50', ('evt10',)), ('gru_rdr10', 'gru_rdr', ('evt10', 'zevt10')),
                         ('koop_rdr10', 'koop_rdr', ('evt10', 'zevt10'))):
    hits, parts = 0, []
    for e in envs:
        best = min(g(e, f'{short}/{c}') for c in cc)
        ok = best <= g(e, f'{long_}/none'); hits += ok
        parts.append(f"{e} {short}+cc {best:.4f} vs {long_} {g(e, long_ + '/none'):.4f} (short none {g(e, short + '/none'):.4f})")
    L.append(f"- F2 {short}+correction <= {long_}: {hits}/3 -> {'PASS' if hits >= 2 else 'FAIL'}  " + '; '.join(parts))
L.append('')

# ---------------- P: planning ----------------
P = J(f'{out}/plan_cheetah.json')
if P:
    L += ['## P. Planning in the real HalfCheetah (CEM-MPC, 200 steps)', '',
          f"Random policy return: {P.get('random', {}).get('mean', nan):.1f}", '',
          '| model | none | per10 | zper5 | evt | flag |', '|---|---|---|---|---|---|']
    models = sorted({k.split('/')[0] for k in P if '/' in k})
    for m in models:
        cell = lambda c: (f"{P[m + '/' + c]['mean']:.1f} +- {P[m + '/' + c]['se']:.1f} (gap {P[m + '/' + c]['gap']:.1f})"
                          if m + '/' + c in P else '-')
        base = P.get(m + '/none')
        flag = ''
        if base:
            for c in ('per10', 'zper5', 'evt'):
                x = P.get(m + '/' + c)
                if x and x['mean'] - base['mean'] >= max(0.15 * abs(base['mean']), 2 * max(x['se'], base['se'])):
                    flag += f'{c} helps; '
        L.append(f"| {m} | {cell('none')} | {cell('per10')} | {cell('zper5')} | {cell('evt')} | {flag} |")
    L.append('')
txt = '\n'.join(L)
open(f'{out}/EXPLORE.md', 'w').write(txt)
print(txt)

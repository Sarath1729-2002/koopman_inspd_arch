"""Short-data report -> results_wm/SHORT.md.  Criteria fixed before the run.
Training sees only 16-step segments; deployment = 180-step open loop; metric = MSE at steps 101-180, mean of 3 seeds.
  A1 correction helps:          X_s10/per10 <= 0.80 x X_s10/none
  A2 beats noise injection:     X_s10/per10 <= X_s10n/none          (the standard short-data training fix)
  A2b complementary to noise:   X_s10n/per10 <= 0.90 x X_s10n/none
  A3 complementary to ensemble: X_s10/ens/per10 <= 0.90 x X_s10/ens/none
  A3b single+correction vs ensemble: X_s10/per10 <= X_s10/ens/none
  A4 density trigger works:     mean corr(density, error growth) >= 0.3 AND dens17 <= per10 (similar budget, ~17)
Each criterion: holds in >= 2 of 3 envs for a family; family-level PASS reported for kae, gru, koop.
HEADLINE: A1 and A2 for >= 2 of 3 families."""
import json, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_wm'
envs = ['cheetah', 'hopper', 'walker']
R = {e: json.load(open(f'{out}/short_{e}.json')) if os.path.exists(f'{out}/short_{e}.json') else {} for e in envs}
nan = float('nan')


def m(e, key, b='b101_end'):
    v = [R[e][f'{key.format(s=s)}'][b] for s in (0, 1, 2) if key.format(s=s) in R[e]]
    return float(np.mean(v)) if v else nan


def tr(e, X, k):
    v = [R[e][f'{X}_s10/s{s}/trace'][k] for s in (0, 1, 2) if f'{X}_s10/s{s}/trace' in R[e]]
    return float(np.mean(v)) if v else nan


L = ['# Short-data regime: course correction when long-horizon training is impossible', '',
     'Training sees only 16-step segments; deployment = 180-step open loop. MSE at steps 101-180, mean of 3 seeds.', '']
fam_pass = {}
for X in ('kae', 'gru', 'koop'):
    b, n = f'{X}_s10', f'{X}_s10n'
    L += [f'## {X}', '', '| env | none | per10 | per5 | res17 | dens17 | noise none | noise per10 | ens none | ens per10 | corr growth resid / dens | A1 | A2 | A2b | A3 | A3b | A4 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    cnt = dict.fromkeys(('A1', 'A2', 'A2b', 'A3', 'A3b', 'A4'), 0)
    for e in envs:
        none, p10, p5 = m(e, b + '/s{s}/none'), m(e, b + '/s{s}/per10'), m(e, b + '/s{s}/per5')
        r17, d17 = m(e, b + '/s{s}/res17'), m(e, b + '/s{s}/dens17')
        nn_, np10 = m(e, n + '/s{s}/none'), m(e, n + '/s{s}/per10')
        en, ep = R[e].get(f'{b}/ens/none', {}).get('b101_end', nan), R[e].get(f'{b}/ens/per10', {}).get('b101_end', nan)
        cr, cd = tr(e, X, 'corr_resid_growth'), tr(e, X, 'corr_dens_growth')
        f = {'A1': p10 <= 0.8 * none, 'A2': p10 <= nn_, 'A2b': np10 <= 0.9 * nn_, 'A3': ep <= 0.9 * en,
             'A3b': p10 <= en, 'A4': cd >= 0.3 and d17 <= p10}
        for k, v in f.items():
            cnt[k] += bool(v)
        L.append(f'| {e} | {none:.4f} | {p10:.4f} | {p5:.4f} | {r17:.4f} | {d17:.4f} | {nn_:.4f} | {np10:.4f} | {en:.4f} | {ep:.4f} | '
                 f'{cr:+.2f} / {cd:+.2f} | ' + ' | '.join('Y' if v else '.' for v in f.values()) + ' |')
    ok = {k: v >= 2 for k, v in cnt.items()}
    fam_pass[X] = ok
    L += ['', '- ' + ', '.join(f"{k} {cnt[k]}/3 {'PASS' if ok[k] else 'FAIL'}" for k in cnt), '']
nh = sum(fam_pass[X]['A1'] and fam_pass[X]['A2'] for X in fam_pass)
L += [f"**HEADLINE (A1 and A2 for >= 2 of 3 families): {'SUPPORTED' if nh >= 2 else 'NOT SUPPORTED'}** ({nh}/3 families)", '']
txt = '\n'.join(L)
open(f'{out}/SHORT.md', 'w').write(txt)
print(txt)

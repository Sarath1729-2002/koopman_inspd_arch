"""Aggregate results_wm/eval_*.json -> results_wm/REPORT.md with pre-registered kill criteria.
Main metric: open-loop MSE (standardized obs) over imagined steps 26-50.
MAIN ARM (fixed in advance after the CPU pilot): zevt5 = keep h, snap the discrete code z to the posterior of the
decoded observation whenever the code would change by more than a threshold calibrated (on training windows) to ~5
re-encodes per 50-step rollout.
K1 drift exists:      gru/none and koop/none error at 26-50 >= 2x error at 1-10 (>= 2 of 3 envs)
K2 re-encoding helps: koop/zevt5 <= 0.85 x koop/none (>= 2 of 3 envs)
K3 beyond the published training fix: koop_rdr/zevt5 <= 0.90 x koop_rdr/none (>= 2 of 3 envs)
PASS = K1 and K2 and K3.  Secondary: GRU generality, discrete vs continuous latent, event vs periodic."""
import json, glob, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else 'results_wm'
R = {os.path.basename(f)[5:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{out}/eval_*.json'))
     if '_' not in os.path.basename(f)[5:-5]}
envs = sorted(R)
lines = ['# Course-correcting re-encoding in a Dreamer-style world model (MuJoCo, open-loop 50 steps)', '',
         'MSE in standardized observation units; columns = imagined steps 1-10 / 11-25 / 26-50. '
         '`reenc` = mean re-encodes per rollout.', '']


def best_re(r, arm, cap=10.5):
    c = [(v['b26_50'], k.split('/')[1]) for k, v in r.items()
         if k.startswith(arm + '/') and not k.endswith('/none') and v['n_reenc'] <= cap]
    return min(c) if c else (float('nan'), '-')


for e in envs:
    r = R[e]
    lines += [f'## {e}', '', '| arm / schedule | reenc | 1-10 | 11-25 | 26-50 |', '|---|---|---|---|---|']
    for k, v in r.items():
        lines.append(f"| {k} | {v['n_reenc']:.1f} | {v['b1_10']:.4f} | {v['b11_25']:.4f} | {v['b26_50']:.4f} |")
    lines.append('')

k1 = k2 = k3 = 0
summ = ['## Kill criteria', '']
for e in envs:
    r = R[e]
    g = lambda k: r.get(k, {}).get('b26_50', float('nan'))
    drift = all(r[f'{a}/none']['b26_50'] >= 2 * r[f'{a}/none']['b1_10'] for a in ('gru', 'koop') if f'{a}/none' in r)
    bk, sk = g('koop/zevt5'), 'zevt5'; br, sr = g('koop_rdr/zevt5'), 'zevt5'
    bg, sg = g('gru/zevt5'), 'zevt5'; bc, sc = g('koop_cont/zevt5'), 'zevt5'
    gk, gr = bk / g('koop/none'), br / g('koop_rdr/none')
    k1 += drift; k2 += gk <= 0.85; k3 += gr <= 0.90
    summ.append(f"- **{e}**: drift {'yes' if drift else 'no'} (koop 1-10 {r['koop/none']['b1_10']:.4f} -> 26-50 {g('koop/none'):.4f}); "
                f"koop {g('koop/none'):.4f} -> {bk:.4f} with {sk} (x{gk:.2f}); "
                f"koop_rdr {g('koop_rdr/none'):.4f} -> {br:.4f} with {sr} (x{gr:.2f}); "
                f"gru {g('gru/none'):.4f} -> {bg:.4f} (x{bg / g('gru/none'):.2f}); "
                f"gru_rdr none {g('gru_rdr/none'):.4f}; "
                f"koop_cont {g('koop_cont/none'):.4f} -> {bc:.4f} (x{bc / g('koop_cont/none'):.2f}); "
                f"koop event vs periodic: zevt5 {g('koop/zevt5'):.4f} ({r.get('koop/zevt5', {}).get('n_reenc', float('nan')):.1f} re-enc) vs zper10 {g('koop/zper10'):.4f} / zper5 {g('koop/zper5'):.4f}; "
                f"full re-encode evt5 {g('koop/evt5'):.4f}; best-of-all koop {best_re(r, 'koop')[0]:.4f} ({best_re(r, 'koop')[1]})")
n = len(envs); need = 2 if n >= 3 else n
K1, K2, K3 = k1 >= need, k2 >= need, k3 >= need
summ += ['', f'- K1 drift exists: {k1}/{n} -> {"PASS" if K1 else "FAIL"}',
         f'- K2 re-encoding helps koop (<=0.85x): {k2}/{n} -> {"PASS" if K2 else "FAIL"}',
         f'- K3 re-encoding helps on top of RDR (<=0.90x): {k3}/{n} -> {"PASS" if K3 else "FAIL"}', '',
         f"**VERDICT: {'PASS - scale it up' if (K1 and K2 and K3) else 'KILL'}**", '']
txt = '\n'.join(summ + [''] + lines)
open(f'{out}/REPORT.md', 'w').write(txt)
print(txt)

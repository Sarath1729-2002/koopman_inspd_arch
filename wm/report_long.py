"""Long-horizon course correction (beyond the 50-step training rollout) -> results_wm/REPORT_long.md
Metric: open-loop MSE over imagined steps 101-180 (b101_end); training rollouts (RDR) cover only 50 steps.
Pre-registered (per family X in {koop, gru}; each needs >= 2 of 3 envs):
  L1 something to fix:   X_rdr/none  101-180 >= 1.5 x its 26-50 error
  L2 inference-only CC:  X_rdr/zevt10 <= 0.85 x X_rdr/none          (code snap, event-triggered)
  L3 train-time CC:      X_rdrre/per10 <= 0.85 x X_rdr/none         (re-encode in training loop, every 10 at test)
  guard (reported):      X_rdrre/per10 26-50 <= 1.10 x X_rdr/none 26-50
PASS = L1 and (L2 or L3) for at least one family."""
import json, glob, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else 'results_wm'
R = {os.path.basename(f)[5:-10]: json.load(open(f)) for f in sorted(glob.glob(f'{out}/eval_*_long.json'))}
envs = sorted(R)
nan = float('nan')
L = ['# Course correction beyond the training horizon (MuJoCo world models, open-loop 180 steps)', '']
verdict = {}
for X in ('koop', 'gru'):
    c1 = c2 = c3 = 0
    L += [f'## {X}', '', '| env | rdr none 26-50 | rdr none 101-180 | rdr zevt10 | rdrre none | rdrre per10 | rdrre per10 26-50 | best rdrre (sched) | plain none |',
          '|---|---|---|---|---|---|---|---|---|']
    for e in envs:
        r = R[e]
        g = lambda k, b='b101_end': r.get(k, {}).get(b, nan)
        base, base50 = g(f'{X}_rdr/none'), g(f'{X}_rdr/none', 'b26_50')
        inf, tr0, tr = g(f'{X}_rdr/zevt10'), g(f'{X}_rdrre/none'), g(f'{X}_rdrre/per10')
        cands = [(v['b101_end'], k.split('/')[1]) for k, v in r.items() if k.startswith(f'{X}_rdrre/')]
        best = min(cands) if cands else (nan, '-')
        c1 += base >= 1.5 * base50; c2 += inf <= 0.85 * base; c3 += tr <= 0.85 * base
        L.append(f"| {e} | {base50:.4f} | {base:.4f} | {inf:.4f} (x{inf / base:.2f}) | {tr0:.4f} (x{tr0 / base:.2f}) | "
                 f"{tr:.4f} (x{tr / base:.2f}) | {g(f'{X}_rdrre/per10', 'b26_50'):.4f} | {best[0]:.4f} ({best[1]}) | {g(f'{X}/none'):.4f} |")
    n = len(envs); need = 2 if n >= 3 else n
    v = (c1 >= need, c2 >= need, c3 >= need)
    verdict[X] = v
    L += ['', f'- L1 drift beyond training horizon: {c1}/{n} -> {"PASS" if v[0] else "FAIL"}',
          f'- L2 inference-only code snap (<=0.85x): {c2}/{n} -> {"PASS" if v[1] else "FAIL"}',
          f'- L3 train-time course correction (<=0.85x): {c3}/{n} -> {"PASS" if v[2] else "FAIL"}', '']
ok = [X for X, (a, b, c) in verdict.items() if a and (b or c)]
L += [f"**VERDICT: {'PASS for ' + ', '.join(ok) + ' - scale it up' if ok else 'KILL'}**", '']
txt = '\n'.join(L)
open(f'{out}/REPORT_long.md', 'w').write(txt)
print(txt)

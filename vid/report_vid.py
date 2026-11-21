"""Video world model report -> results_vid/VIDEO.md.  Criteria fixed before the run.
Open-Oasis 500M, trained on 32-frame windows, rolled out 512 frames (16x). Last bin = frames 448-511.
  B0 room to correct: base latent re-encode shift in the last bin >= 2x the real-frame floor AND >= 1.5x its first bin
  B1 correction reduces drift: reenc1 last-bin FD <= 0.75 x base last-bin FD
  B2 beats the existing inference-time fix: best reencP (P in 1,4,16) last-bin FD <= best stab (15, 50, 100)
  B3 not by blurring: best reenc last-bin sharpness >= 0.8 x base last-bin sharpness, and frame change >= 0.5 x base
HEADLINE: B0 and B1 and B2 (B3 reported as a guard)."""
import json, glob, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_vid'
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{out}/*.json'))}
if not R:
    print('no results'); raise SystemExit
nan = float('nan')
last = lambda r, k: float(np.mean(np.asarray(r[k])[:, -64:])) if k in r else nan
first = lambda r, k: float(np.mean(np.asarray(r[k])[:, 1:65])) if k in r else nan
fdl = lambda r: r['fd'][-1] if r.get('fd') else nan
L = ['# Course correction in a streaming video world model (Open-Oasis 500M, 512 frames = 16x training window)', '',
     '| arm | stab | P | shift first -> last (real floor) | FD first -> last | brightness last / real | saturation last / real | sharpness last / real | frame change last | sec |',
     '|---|---|---|---|---|---|---|---|---|---|']
for a, r in R.items():
    ref = r['ref']
    L.append(f"| {a} | {r['stab']} | {r['P']} | {first(r, 'shift'):.4f} -> {last(r, 'shift'):.4f} ({r['shift_floor_real']:.4f}) | "
             f"{r['fd'][0]:.1f} -> {fdl(r):.1f} | {last(r, 'bright') / ref['bright']:.2f} | {last(r, 'sat') / ref['sat']:.2f} | "
             f"{last(r, 'sharp') / ref['sharp']:.2f} | {last(r, 'change'):.4f} | {r['seconds']:.0f} |")
b = R.get('base')
v = {}
if b:
    v['B0'] = last(b, 'shift') >= 2 * b['shift_floor_real'] and last(b, 'shift') >= 1.5 * first(b, 'shift')
    v['B1'] = 'reenc1' in R and fdl(R['reenc1']) <= 0.75 * fdl(b)
    re = [a for a in R if a in ('reenc1', 'reenc4', 'reenc16')]
    sb = [a for a in R if a in ('base', 'stab50', 'stab100')]
    best_re = min(re, key=lambda a: fdl(R[a])) if re else None
    best_sb = min(sb, key=lambda a: fdl(R[a])) if sb else None
    v['B2'] = bool(best_re and best_sb and fdl(R[best_re]) <= fdl(R[best_sb]))
    v['B3'] = bool(best_re and last(R[best_re], 'sharp') >= 0.8 * last(b, 'sharp') and last(R[best_re], 'change') >= 0.5 * last(b, 'change'))
    L += ['', f'- best re-encode arm: {best_re} (last FD {fdl(R[best_re]) if best_re else nan:.1f}); best stabilization arm: '
              f'{best_sb} (last FD {fdl(R[best_sb]) if best_sb else nan:.1f})']
    L += [f"- {k}: {'PASS' if x else 'FAIL'}" for k, x in v.items()]
    L += ['', f"**HEADLINE (B0, B1, B2): {'SUPPORTED' if v['B0'] and v['B1'] and v['B2'] else 'NOT SUPPORTED'}**"
              f"  (guard B3 {'PASS' if v['B3'] else 'FAIL'})", '']
txt = '\n'.join(L)
open(f'{out}/VIDEO.md', 'w').write(txt)
print(txt)

"""Video world model v2 report -> results_vid2/VIDEO2.md.  Criteria fixed before the run.
v1 finding: re-encoding cut late FD (reenc4 -17%) and beat every stabilization setting, but repeated decode->encode
darkened frames (brightness 0.65x real) and over-saturated them -- the VAE's own reconstruction bias compounds.
v2: 3 sample videos x 4 prompts, 512 frames, last bin = frames 448-511.  Primary arm (pre-registered): evt2 =
re-encode a frame only when its latent re-encode shift exceeds 2x the real-frame floor (Fathi's residual trigger,
which in video DOES rise with drift).
  V0 off-manifold drift replicates: base last-bin shift >= 2 x floor                     (>= 2 of 3 videos)
  V1 bias-free correction: evt2 last FD <= 0.85 x base AND brightness ratio in [0.85, 1.15] AND saturation <= 1.3x
  V2 v1 replicates: reenc4 last FD <= 0.90 x base
  V3 beats latent moment matching alone: evt2 last FD <= mm last FD
HEADLINE: V0 and V1 and V3 (each in >= 2 of 3 videos)."""
import json, glob, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_vid2'
R = {}
for f in glob.glob(f'{out}/*_v[0-9].json'):
    r = json.load(open(f)); R.setdefault(r['video_idx'], {})[r['arm']] = r
vids = sorted(R)
nan = float('nan')
last = lambda r, k: float(np.mean(np.asarray(r[k])[:, -64:]))
first = lambda r, k: float(np.mean(np.asarray(r[k])[:, 1:65]))
fdl = lambda r: r['fd'][-1]
arms = ['base', 'reenc4', 'evt2', 'evt3', 'part05', 'reenc4_mm', 'mm']
L = ['# Course correction in Open-Oasis 500M, v2 (bias-aware variants; 3 videos x 4 prompts; 512 frames)', '',
     '| video | arm | re-encodes / 512 | shift first -> last (floor) | FD first -> last | FD vs base | brightness | saturation | sharpness | change |',
     '|---|---|---|---|---|---|---|---|---|---|']
cnt = dict(V0=0, V1=0, V2=0, V3=0)
for v in vids:
    D = R[v]; b = D.get('base')
    for a in arms:
        if a not in D:
            continue
        r = D[a]; ref = r['ref']
        rel = fdl(r) / fdl(b) if b else nan
        L.append(f"| {r['video'][:18]} | {a} | {np.mean(r['n_reenc']):.0f} | {first(r, 'shift'):.4f} -> {last(r, 'shift'):.4f} "
                 f"({r['shift_floor_real']:.4f}) | {r['fd'][0]:.1f} -> {fdl(r):.1f} | x{rel:.2f} | "
                 f"{last(r, 'bright') / ref['bright']:.2f} | {last(r, 'sat') / ref['sat']:.2f} | "
                 f"{last(r, 'sharp') / ref['sharp']:.2f} | {last(r, 'change'):.4f} |")
    if not b:
        continue
    col = lambda r: 0.85 <= last(r, 'bright') / r['ref']['bright'] <= 1.15 and last(r, 'sat') / r['ref']['sat'] <= 1.3
    cnt['V0'] += last(b, 'shift') >= 2 * b['shift_floor_real']
    if 'evt2' in D:
        cnt['V1'] += fdl(D['evt2']) <= 0.85 * fdl(b) and col(D['evt2'])
        cnt['V3'] += 'mm' in D and fdl(D['evt2']) <= fdl(D['mm'])
    cnt['V2'] += 'reenc4' in D and fdl(D['reenc4']) <= 0.9 * fdl(b)
need = 2 if len(vids) >= 3 else len(vids)
ok = {k: v >= need for k, v in cnt.items()}
L += ['', '- ' + ', '.join(f"{k} {cnt[k]}/{len(vids)} {'PASS' if ok[k] else 'FAIL'}" for k in cnt),
      '', f"**HEADLINE (V0, V1, V3): {'SUPPORTED' if ok['V0'] and ok['V1'] and ok['V3'] else 'NOT SUPPORTED'}**", '']
txt = '\n'.join(L)
open(f'{out}/VIDEO2.md', 'w').write(txt)
print(txt)

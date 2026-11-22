"""Video world model v3 report -> results_vid3/VIDEO3.md.  Criteria fixed before the run.
v2 finding: plain re-encoding every 4 frames cut late FD in all 3 videos (x0.83, 0.73, 0.87) but drifted colour
(brightness 0.66 / 1.50 / 0.62 of real). v3 keeps the model's own low-frequency latent (colour, lighting) and takes only
high-frequency structure from E(D(z)) (arm suffix _lf), and adds a second sampling seed to measure the FD noise floor.
Colour deviation d = |log(brightness / prompt video)| + |log(saturation / prompt video)|, last bin (frames 448-511).
  W0 replication: reenc4 last FD <= 0.90 x base
  W1 primary: reenc4_lf last FD <= 0.85 x base AND d(reenc4_lf) <= d(base) + 0.10
  W2 beyond noise: (FD base - FD reenc4_lf) > |FD base_s1 - FD base| AND reenc4_lf_s1 <= 0.90 x base_s1
  W3 colour fix works: d(reenc4_lf) < d(reenc4)
Each in >= 2 of 3 videos.  HEADLINE: W1 and W2."""
import json, glob, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_vid3'
R = {}
for f in glob.glob(f'{out}/*_v[0-9].json'):
    r = json.load(open(f)); R.setdefault(r['video_idx'], {})[r['arm']] = r
vids = sorted(R); nan = float('nan')
last = lambda r, k: float(np.mean(np.asarray(r[k])[:, -64:]))
first = lambda r, k: float(np.mean(np.asarray(r[k])[:, 1:65]))
fdl = lambda r: r['fd'][-1]
rv = lambda r: r.get('ref_video', r['ref'])
bri = lambda r: last(r, 'bright') / rv(r)['bright']
sat = lambda r: last(r, 'sat') / rv(r)['sat']
dev = lambda r: abs(np.log(bri(r))) + abs(np.log(sat(r)))
arms = ['base', 'base_s1', 'reenc4', 'reenc4_lf', 'reenc4_lf_s1', 'reenc4_mp', 'reenc1_lf', 'evt2_lf']
L = ['# Course correction in Open-Oasis 500M, v3 (structure-only re-encoding; 3 videos x 4 prompts; 512 frames)', '',
     'Colour ratios are relative to the prompt video itself; d = colour deviation (0 = matches the video).', '',
     '| video | arm | re-enc | shift last (floor) | FD first -> last | FD vs base | brightness | saturation | d | sharpness | change |',
     '|---|---|---|---|---|---|---|---|---|---|---|']
cnt = dict(W0=0, W1=0, W2=0, W3=0)
for v in vids:
    D = R[v]; b = D.get('base')
    for a in arms:
        if a not in D:
            continue
        r = D[a]
        L.append(f"| {r['video'][:18]} | {a} | {np.mean(r['n_reenc']):.0f} | {last(r, 'shift'):.4f} ({r['shift_floor_real']:.4f}) | "
                 f"{r['fd'][0]:.1f} -> {fdl(r):.1f} | x{fdl(r) / fdl(b) if b else nan:.2f} | {bri(r):.2f} | {sat(r):.2f} | "
                 f"{dev(r):.2f} | {last(r, 'sharp') / rv(r)['sharp']:.2f} | {last(r, 'change'):.4f} |")
    if not b:
        continue
    g = D.get
    if g('reenc4'): cnt['W0'] += fdl(D['reenc4']) <= 0.9 * fdl(b)
    if g('reenc4_lf'):
        lf = D['reenc4_lf']
        cnt['W1'] += fdl(lf) <= 0.85 * fdl(b) and dev(lf) <= dev(b) + 0.10
        if g('base_s1') and g('reenc4_lf_s1'):
            cnt['W2'] += (fdl(b) - fdl(lf)) > abs(fdl(D['base_s1']) - fdl(b)) and fdl(D['reenc4_lf_s1']) <= 0.9 * fdl(D['base_s1'])
        if g('reenc4'): cnt['W3'] += dev(lf) < dev(D['reenc4'])
need = 2 if len(vids) >= 3 else len(vids)
ok = {k: v >= need for k, v in cnt.items()}
L += ['', '- ' + ', '.join(f"{k} {cnt[k]}/{len(vids)} {'PASS' if ok[k] else 'FAIL'}" for k in cnt),
      '', f"**HEADLINE (W1, W2): {'SUPPORTED' if ok['W1'] and ok['W2'] else 'NOT SUPPORTED'}**", '']
txt = '\n'.join(L); open(f'{out}/VIDEO3.md', 'w').write(txt); print(txt)

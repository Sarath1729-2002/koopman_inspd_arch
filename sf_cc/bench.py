"""Benchmark table for the long-video arms (Nov 26), our own implementation of VBench-Long-style dimensions and of
ID-Forcing's two drift scores (arXiv 2610.03120). NOT leaderboard numbers: same videos, same metric code, so arms are
comparable with each other, not with published tables.
  subject consistency  100 x mean over frames of (cos to first frame + cos to previous frame) / 2, DINOv2 CLS
  aesthetic            LAION predictor, mean over the video and over the last 10 s (1-10)
  dynamic              mean RAFT flow magnitude (pixels at 240x416)
  colour drift score   100 x histogram intersection of saturation-weighted hue histograms (36 bins), first vs last 5 s
                       (100 = no colour drift)
  motion drift score   100 x min/max of mean flow over the first vs last 5 s (100 = motion kept)
Each correction arm is compared with its no-correction arm, paired over prompts: mean difference [95% CI].
Anchor arms (Nov 26; results_sf_anchor, 120 s, prompts 0-15; criteria fixed before the run):
  NC  an arm counts only if it keeps motion (dynamic >= 0.8 x the reference) and is not significantly less aesthetic
  AN1 some Self-Forcing anchor arm (hg25 / hg50 / adp2 / adt2) raises the colour drift score: paired CI > 0, with NC
  AN2 the AN1 arm with the largest colour gain also raises subject consistency: CI > 0
  AN3 the same kind of arm on CausVid (hg50 or adp2) raises the colour drift score: CI > 0, with NC
Writes results_split/BENCH.md + bench.json"""
import glob, json, os, numpy as np

rng = np.random.default_rng(0)
GROUPS = [('CausVid 4 min', 'results_sf_long', 'causvid_sink3', ['causvid_sink3_reenc2_cache']),
          ('Self-Forcing 4 min', 'results_sf_long', 'sink3', ['sink3_reenc2_cache']),
          ('CausVid 60 s', 'results_sf', 'causvid_sink3', ['causvid_sink3_reenc2_cache', 'causvid_sink3_evt2']),
          ('Self-Forcing 60 s', 'results_sf', 'sink3', ['sink3_reenc2_cache', 'sink3_dproj2_cache']),
          ('Self-Forcing 120 s, anchors', 'results_sf_anchor', 'sink3', ['sink3_hg25', 'sink3_hg50', 'sink3_adp2', 'sink3_adt2', 'sink3_reenc2_cache']),
          ('CausVid 120 s, anchors', 'results_sf_anchor', 'causvid_sink3', ['causvid_sink3_hg50', 'causvid_sink3_adp2'])]
KEYS = ['subject', 'aes', 'aes_last', 'dynamic', 'colour', 'motion']


def hue_hist(frames):
    x = frames.astype(np.float32) / 255
    mx, mn = x.max(-1), x.min(-1); d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    w = s * (s > 0.1)
    hist = np.histogram(h.ravel(), 36, (0, 1), weights=w.ravel())[0]
    return hist / max(hist.sum(), 1e-9)


def colour_score(mp4, fps=16, sec=5, every=8):
    import av
    first, last, buf = [], [], []
    with av.open(mp4) as c:
        for t, fr in enumerate(c.decode(video=0)):
            if t % every:
                continue
            a = fr.to_ndarray(format='rgb24')[::4, ::4]
            if t < fps * sec:
                first.append(a)
            buf.append(a); buf = buf[-(fps * sec // every):]
    last = buf
    return 100 * float(np.minimum(hue_hist(np.stack(first)), hue_hist(np.stack(last))).sum())


def video_scores(mj):
    m = json.load(open(mj)); t = np.asarray(m['t_sec'], float); T = t.max()
    df, da, fl, ae = (np.asarray(m[k], float) for k in ('dino_first', 'dino_adj', 'flow', 'aesthetic'))
    n = min(len(df), len(da), len(t))
    tf = t[:len(fl)]
    f1, f2 = fl[tf < 5].mean(), fl[tf >= T - 5].mean()
    return {'subject': 100 * float(((df[:n] + da[:n]) / 2).mean()), 'aes': float(ae.mean()), 'aes_last': float(ae[t[:len(ae)] >= T - 10].mean()),
            'dynamic': float(fl.mean()), 'motion': 100 * float(min(f1, f2) / max(f1, f2, 1e-9)),
            'colour': colour_score(mj.replace('.metrics.json', '.mp4'))}


def boot(x, B=4000):
    x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(x.mean())] + [float(v) for v in np.percentile(bs, [2.5, 97.5])]


if __name__ == '__main__':
    os.makedirs('results_split', exist_ok=True)
    cf = 'results_split/bench_cache.json'
    C = json.load(open(cf)) if os.path.exists(cf) else {}
    def arm_scores(root, arm):
        out = {}
        for mj in sorted(glob.glob(f'{root}/{arm}/*.metrics.json')):
            if not os.path.exists(mj.replace('.metrics.json', '.json')) or not os.path.exists(mj.replace('.metrics.json', '.mp4')):
                continue
            if mj not in C:
                C[mj] = video_scores(mj); json.dump(C, open(cf, 'w'))
            out[os.path.basename(mj)] = C[mj]
        return out
    L = ['# Long-video benchmarks (our implementation; arms comparable with each other, not with published tables)', '']
    res = {}
    for name, root, ref, arms in GROUPS:
        R = arm_scores(root, ref)
        if not R:
            continue
        L += [f'## {name}', '', '| arm | n | ' + ' | '.join(['subject consistency', 'aesthetic', 'aesthetic last 10 s', 'dynamic', 'colour drift score', 'motion drift score']) + ' |',
              '|---|---|' + '---|' * len(KEYS)]
        L.append(f'| {ref} (no correction) | {len(R)} | ' + ' | '.join(f'{np.mean([r[k] for r in R.values()]):.3f}' for k in KEYS) + ' |')
        res[name] = {ref: {k: float(np.mean([r[k] for r in R.values()])) for k in KEYS}}
        for a in arms:
            A = arm_scores(root, a); ids = sorted(set(A) & set(R))
            if not ids:
                continue
            d = {k: boot([A[i][k] - R[i][k] for i in ids]) for k in KEYS}
            res[name][a] = d
            L.append(f'| {a} − ref | {len(ids)} | ' + ' | '.join(f'{d[k][0]:+.3f} [{d[k][1]:+.3f}, {d[k][2]:+.3f}]' for k in KEYS) + ' |')
        L.append('')
    def nca(g, a):
        d = res[g][a]; ref = list(res[g].values())[0]
        return ref['dynamic'] + d['dynamic'][0] >= 0.8 * ref['dynamic'] and d['aes'][2] >= 0
    gs, gc = 'Self-Forcing 120 s, anchors', 'CausVid 120 s, anchors'
    if gs in res:
        cand = [a for a in ('sink3_hg25', 'sink3_hg50', 'sink3_adp2', 'sink3_adt2') if a in res[gs]]
        win = [a for a in cand if res[gs][a]['colour'][1] > 0 and nca(gs, a)]
        crit = {'AN1': bool(win)}
        if win:
            best = max(win, key=lambda a: res[gs][a]['colour'][0])
            crit['AN2'] = bool(res[gs][best]['subject'][1] > 0); crit['AN1_arm'] = best
            kind = 'hg50' if best.startswith('sink3_hg') else 'adp2'
            ca = f'causvid_sink3_{kind}'
            if gc in res and ca in res[gc]:
                crit['AN3'] = bool(res[gc][ca]['colour'][1] > 0 and nca(gc, ca))
        res['criteria'] = crit
        L += ['## Pre-registered (anchor arms)', ''] + [f'- {k}: {v if isinstance(v, str) else ("PASS" if v else "FAIL")}' for k, v in crit.items()]
    json.dump(res, open('results_split/bench.json', 'w'), indent=1)
    open('results_split/BENCH.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

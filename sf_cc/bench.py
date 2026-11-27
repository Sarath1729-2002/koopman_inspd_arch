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
Overnight confirmation (Nov 27; criteria fixed before the run):
  FP1 fresh prompts 16-31 (results_sf_anchor2, 120 s): Self-Forcing adp2 raises the colour drift score, CI > 0, with NC
  FP2 same on CausVid, CI > 0, with NC
  FP3 Self-Forcing adp2 on fresh prompts raises subject consistency, CI > 0
  L4  4 min (results_sf_long, prompts 0-7): Self-Forcing adp2 raises the colour drift score, CI > 0, with NC
  SEAM no visible jumps: on fresh prompts, the frame-to-frame DINO similarity across chunks where adp2 corrects, minus
       across other chunk boundaries, is not lower than for the uncorrected video by more than 0.01 (CI lower > -0.01)
  SW1 prompt switch halfway (results_sf_switch): the anchor does not block the new prompt: CLIP to the second prompt over
      the last 30 s, adp2 minus sink only, CI lower > -1
  SW2 refreshing the anchor at the switch helps the new prompt: adp2+refresh minus adp2, CLIP to the second prompt, CI > 0
  Reported: adp strength (adp1 / adp3), triggered (adt2), adp2 + VAE re-encode stacked.
Writes results_split/BENCH.md + bench.json"""
import glob, json, os, numpy as np

rng = np.random.default_rng(0)
GROUPS = [('CausVid 4 min', 'results_sf_long', 'causvid_sink3', ['causvid_sink3_reenc2_cache']),
          ('Self-Forcing 4 min', 'results_sf_long', 'sink3', ['sink3_reenc2_cache']),
          ('CausVid 60 s', 'results_sf', 'causvid_sink3', ['causvid_sink3_reenc2_cache', 'causvid_sink3_evt2']),
          ('Self-Forcing 60 s', 'results_sf', 'sink3', ['sink3_reenc2_cache', 'sink3_dproj2_cache']),
          ('Self-Forcing 120 s, anchors', 'results_sf_anchor', 'sink3', ['sink3_hg25', 'sink3_hg50', 'sink3_adp2', 'sink3_adt2', 'sink3_reenc2_cache']),
          ('CausVid 120 s, anchors', 'results_sf_anchor', 'causvid_sink3', ['causvid_sink3_hg50', 'causvid_sink3_adp2']),
          ('Self-Forcing 120 s, fresh prompts', 'results_sf_anchor2', 'sink3', ['sink3_adp1', 'sink3_adp2', 'sink3_adp3', 'sink3_adt2', 'sink3_adp2_reenc2_cache']),
          ('CausVid 120 s, fresh prompts', 'results_sf_anchor2', 'causvid_sink3', ['causvid_sink3_adp2', 'causvid_sink3_adp2_reenc2_cache']),
          ('Self-Forcing 120 s, prompt switch', 'results_sf_switch', 'sink3_sw', ['sink3_adp2_sw', 'sink3_adp2_sw_rf'])]
GROUPS[0][3].append('causvid_sink3_adp2'); GROUPS[1][3].append('sink3_adp2')
KEYS = ['subject', 'aes', 'aes_last', 'dynamic', 'colour', 'motion', 'seam', 'clip_late', 'clip2_late']
NAMES = ['subject consistency', 'aesthetic', 'aesthetic last 10 s', 'dynamic', 'colour drift score', 'motion drift score',
         'seam (adj. sim. at corrected - other chunk boundaries)', 'CLIP last 30 s', 'CLIP to 2nd prompt, last 30 s']


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
    """decodes only the first and the last `sec` seconds (seeks to the end instead of decoding the whole video)"""
    import av
    first, last = [], []
    with av.open(mp4) as c:
        st = c.streams.video[0]
        for t, fr in enumerate(c.decode(video=0)):
            if t >= fps * sec:
                break
            if t % every == 0:
                first.append(fr.to_ndarray(format='rgb24')[::4, ::4])
        dur = float(st.duration * st.time_base) if st.duration else float(c.duration / 1e6)
        c.seek(int(max(dur - sec - 1, 0) / st.time_base), stream=st, backward=True)
        buf = []
        for t, fr in enumerate(c.decode(video=0)):
            buf.append(fr.to_ndarray(format='rgb24')[::4, ::4])
        last = buf[-(fps * sec):][::every]
    return 100 * float(np.minimum(hue_hist(np.stack(first)), hue_hist(np.stack(last))).sum())


def seam(t, da, fps=16, every_chunks=4):
    """adjacent-sample DINO similarity across chunks where adp corrects (every 4th chunk) minus across other chunks.
    chunk b (3 latents) starts at video frame 12 b - 3 for b >= 1"""
    fr = np.round(t * fps).astype(int); a, o = [], []
    for b in range(1, int(fr.max() // 12) + 1):
        f0 = 12 * b - 3
        k = np.searchsorted(fr, f0)          # pair (k-1, k) straddles the boundary
        if 1 <= k < len(da):
            (a if b % every_chunks == 0 else o).append(da[k])
    return float(np.mean(a) - np.mean(o)) if a and o else float('nan')


def video_scores(mj, old=None):
    m = json.load(open(mj)); t = np.asarray(m['t_sec'], float); T = t.max()
    df, da, fl, ae = (np.asarray(m[k], float) for k in ('dino_first', 'dino_adj', 'flow', 'aesthetic'))
    n = min(len(df), len(da), len(t))
    tf = t[:len(fl)]
    f1, f2 = fl[tf < 5].mean(), fl[tf >= T - 5].mean()
    cl = np.asarray(m['clip'], float); c2 = np.asarray(m.get('clip2', [np.nan] * len(cl)), float)
    late = t[:len(cl)] >= T - 30
    return {'subject': 100 * float(((df[:n] + da[:n]) / 2).mean()), 'aes': float(ae.mean()), 'aes_last': float(ae[t[:len(ae)] >= T - 10].mean()),
            'dynamic': float(fl.mean()), 'motion': 100 * float(min(f1, f2) / max(f1, f2, 1e-9)),
            'seam': seam(t[:len(da)], da), 'clip_late': float(cl[late].mean()), 'clip2_late': float(np.nanmean(c2[late])) if np.isfinite(c2).any() else float('nan'),
            'colour': old['colour'] if old and 'colour' in old else colour_score(mj.replace('.metrics.json', '.mp4'))}


def boot(x, B=4000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float('nan')] * 3
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
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
            if mj not in C or any(k not in C[mj] for k in KEYS):
                C[mj] = video_scores(mj, C.get(mj)); json.dump(C, open(cf, 'w')); print('[bench]', mj, flush=True)
            out[os.path.basename(mj)] = C[mj]
        return out
    PV = {}
    L = ['# Long-video benchmarks (our implementation; arms comparable with each other, not with published tables)', '']
    res = {}
    for name, root, ref, arms in GROUPS:
        R = arm_scores(root, ref)
        if not R:
            continue
        PV[name] = {ref: R}
        ks = [k for k in KEYS if np.isfinite([r[k] for r in R.values()]).any()]
        L += [f'## {name}', '', '| arm | n | ' + ' | '.join(NAMES[KEYS.index(k)] for k in ks) + ' |', '|---|---|' + '---|' * len(ks)]
        L.append(f'| {ref} (no correction) | {len(R)} | ' + ' | '.join(f'{np.nanmean([r[k] for r in R.values()]):.3f}' for k in ks) + ' |')
        res[name] = {ref: {k: float(np.nanmean([r[k] for r in R.values()])) for k in KEYS}}
        for a in arms:
            A = arm_scores(root, a); ids = sorted(set(A) & set(R))
            if not ids:
                continue
            PV[name][a] = A
            d = {k: boot([A[i][k] - R[i][k] for i in ids]) for k in KEYS}
            res[name][a] = d
            L.append(f'| {a} − ref | {len(ids)} | ' + ' | '.join(f'{d[k][0]:+.3f} [{d[k][1]:+.3f}, {d[k][2]:+.3f}]' for k in ks) + ' |')
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
    c2, fpS, fpC, lS, swg = {}, 'Self-Forcing 120 s, fresh prompts', 'CausVid 120 s, fresh prompts', 'Self-Forcing 4 min', 'Self-Forcing 120 s, prompt switch'
    if fpS in res and 'sink3_adp2' in res[fpS]:
        d = res[fpS]['sink3_adp2']
        c2['FP1'] = bool(d['colour'][1] > 0 and nca(fpS, 'sink3_adp2')); c2['FP3'] = bool(d['subject'][1] > 0)
        c2['SEAM'] = bool(d['seam'][1] > -0.01)
    if fpC in res and 'causvid_sink3_adp2' in res[fpC]:
        c2['FP2'] = bool(res[fpC]['causvid_sink3_adp2']['colour'][1] > 0 and nca(fpC, 'causvid_sink3_adp2'))
    if lS in res and 'sink3_adp2' in res[lS]:
        c2['L4'] = bool(res[lS]['sink3_adp2']['colour'][1] > 0 and nca(lS, 'sink3_adp2'))
    if swg in res and 'sink3_adp2_sw' in res[swg]:
        c2['SW1'] = bool(res[swg]['sink3_adp2_sw']['clip2_late'][1] > -1)
        if 'sink3_adp2_sw_rf' in PV[swg]:
            A, B = PV[swg]['sink3_adp2_sw_rf'], PV[swg]['sink3_adp2_sw']; ids = sorted(set(A) & set(B))
            dd = boot([A[i]['clip2_late'] - B[i]['clip2_late'] for i in ids]); res['SW2_detail'] = dd
            c2['SW2'] = bool(dd[1] > 0)
    if c2:
        res['criteria_overnight'] = c2
        L += ['', '## Pre-registered (overnight confirmation)', ''] + [f'- {k}: {"PASS" if v else "FAIL"}' for k, v in c2.items()]
    json.dump(res, open('results_split/bench.json', 'w'), indent=1)
    open('results_split/BENCH.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

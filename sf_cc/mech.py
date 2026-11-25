"""What degrades in CausVid over 4 minutes, if not VAE drift? (Nov 25) CausVid's re-encode drift stays ~1.4x its floor
at 4 min, yet re-encoding helps (V1, V2). This reads the saved mp4s and tracks simple image statistics per 10-s window:
  sharp   variance of the Laplacian of the grey image (detail / blur)
  hf      share of spectral energy above 1/4 of the Nyquist frequency (texture; oversmoothing lowers it)
  contr   std of grey levels          sat   mean HSV saturation          bright  mean grey level
  flick   mean |frame_t - frame_{t-1}| (temporal flicker / motion)
Change = mean over the last 30 s minus the first 10 s, per video.
M1 (fixed before the run): at least one statistic (a) changes in uncorrected CausVid (causvid_sink3; paired CI of the
   change excludes 0) AND (b) re-encoding (causvid_sink3_reenc2_cache) shrinks that change: CI of
   |change_reenc| - |change_none| < 0, paired over prompts. Self-Forcing (sink3 vs sink3_reenc2_cache) reported alongside.
Usage: python -m sf_cc.mech [results_sf_long]  -> <root>/MECH.md, <root>/mech.json"""
import glob, json, os, sys, numpy as np
import av

rng = np.random.default_rng(0)


def stats(path, every=4, fps=16):
    out, prev = [], None
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i % every:
                continue
            x = fr.to_ndarray(format='rgb24')[::2, ::2].astype(np.float32) / 255
            g = x @ np.array([0.299, 0.587, 0.114], np.float32)
            lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
            F = np.abs(np.fft.rfft2(g - g.mean())) ** 2
            fy = np.abs(np.fft.fftfreq(g.shape[0]))[:, None]; fx = np.fft.rfftfreq(g.shape[1])[None]
            hf = F[np.sqrt(fy ** 2 + fx ** 2) > 0.125].sum() / max(F.sum(), 1e-12)
            mx, mn = x.max(-1), x.min(-1)
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()
            fl = float(np.abs(g - prev).mean()) if prev is not None else np.nan
            prev = g
            out.append((i / fps, lap.var(), hf, g.std(), sat, g.mean(), fl))
    return np.array(out)


KEYS = ['sharp', 'hf', 'contr', 'sat', 'bright', 'flick']


def change(S):
    t = S[:, 0]; T = t.max()
    first, last = S[t < 10], S[t >= T - 30]
    return np.array([np.nanmean(last[:, k + 1]) - np.nanmean(first[:, k + 1]) for k in range(len(KEYS))])


def ci(d):
    d = d[~np.isnan(d)]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return [float(d.mean())] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'results_sf_long'
    cache = f'{root}/mech_cache.json'
    C = json.load(open(cache)) if os.path.exists(cache) else {}
    for arm in ('causvid_sink3', 'causvid_sink3_reenc2_cache', 'sink3', 'sink3_reenc2_cache'):
        for f in sorted(glob.glob(f'{root}/{arm}/*.mp4')):
            k = f'{arm}/{os.path.basename(f)}'
            if k not in C:
                C[k] = change(stats(f)).tolist(); print('[mech]', k, flush=True)
                json.dump(C, open(cache, 'w'))
    L = [f'# What changes over 4 minutes ({root})', '', 'Change = last 30 s minus first 10 s, mean over prompts [95% CI]; "reenc effect" = |change with re-encode| - |change without| (negative = re-encode keeps the statistic closer to its start), paired over prompts.', '']
    res = {}
    for model, ref, arm in (('CausVid', 'causvid_sink3', 'causvid_sink3_reenc2_cache'), ('Self-Forcing', 'sink3', 'sink3_reenc2_cache')):
        ids = sorted({k.split('/')[1] for k in C if k.startswith(ref + '/')} & {k.split('/')[1] for k in C if k.startswith(arm + '/')})
        if not ids:
            continue
        A = np.array([C[f'{ref}/{i}'] for i in ids]); Bm = np.array([C[f'{arm}/{i}'] for i in ids])
        L += [f'## {model} (n={len(ids)})', '', '| statistic | change without correction | change with re-encode | reenc effect |', '|---|---|---|---|']
        r = {}
        for j, k in enumerate(KEYS):
            ca, cb, ce = ci(A[:, j]), ci(Bm[:, j]), ci(np.abs(Bm[:, j]) - np.abs(A[:, j]))
            r[k] = {'none': ca, 'reenc': cb, 'effect': ce}
            fm = lambda c: f'{c[0]:+.4g} [{c[1]:+.3g}, {c[2]:+.3g}]'
            L.append(f'| {k} | {fm(ca)} | {fm(cb)} | {fm(ce)} |')
        res[model] = r; L.append('')
    if 'CausVid' in res:
        hits = [k for k, v in res['CausVid'].items() if (v['none'][1] > 0 or v['none'][2] < 0) and v['effect'][2] < 0]
        res['M1'] = bool(hits); res['M1_stats'] = hits
        L += ['## Pre-registered', '', f"- M1: {'PASS' if hits else 'FAIL'}" + (f' ({", ".join(hits)})' if hits else '')]
    open(f'{root}/MECH.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{root}/mech.json', 'w'), indent=1)
    print('\n'.join(L))

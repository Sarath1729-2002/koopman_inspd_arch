"""How does drift grow without correction, and does the regrowth after a correction predict the best period? (Nov 24)
Uses files that already exist; CPU only, seconds.
  Oasis: results_val/ and results_final/ <arm>_v*.json   'shift' = per-frame re-encode drift [offsets, frames]
  Self-Forcing / CausVid: results_sf/ and results_sf_long/ <arm>/*.json   'shift' = [[chunk, drift], ...]
G1 growth law. Per uncorrected rollout, drift (16-frame / 1-chunk bins, from the first logged step) is fitted with
   linear a+bt, exponential A e^{bt}, power A t^p and saturating c-(c-y0)e^{-t/tau}; the law with the lowest AIC wins.
   Pass for a family if exponential wins in >= 60% of its rollouts AND the median exponential rate is > 0 (bootstrap
   CI over rollouts > 0). Pre-registered headline: G1 on Oasis without a sink (the setting where correction works).
G2 period from regrowth. In sink4_reenc8 / sink4_reenc16 (validation offsets), drift is averaged by phase k = frames
   since the last correction (frames >= 64). Predicted period P_pred = smallest k with drift(k) >= 2 x drift(1).
   Pass if P_pred is within a factor 2 of the best period from the validation FD sweep (sink4_reencP, mean ResNet FD over
   bins starting at frame >= 256; lower is better).
Writes results_growth.json"""
import glob, json, os, re, sys, numpy as np

rng = np.random.default_rng(0)


def fit_laws(t, y):
    t = np.asarray(t, float); y = np.clip(np.asarray(y, float), 1e-12, None); n = len(t); out = {}
    aic = lambda yh, k: n * np.log(max(((y - yh) ** 2).sum() / n, 1e-30)) + 2 * k
    b = np.polyfit(t, y, 1); out['linear'] = (aic(np.polyval(b, t), 2), b[0])
    b = np.polyfit(t, np.log(y), 1); out['exponential'] = (aic(np.exp(np.polyval(b, t)), 2), b[0])
    tt = np.maximum(t - t.min() + 1, 1)
    b = np.polyfit(np.log(tt), np.log(y), 1); out['power'] = (aic(np.exp(np.polyval(b, np.log(tt))), 2), b[0])
    try:
        from scipy.optimize import curve_fit
        f = lambda x, c, y0, tau: c - (c - y0) * np.exp(-(x - t.min()) / tau)
        p, _ = curve_fit(f, t, y, p0=(y.max(), y[0], (t.max() - t.min()) / 4 + 1e-3), maxfev=20000,
                         bounds=([0, 0, 1e-3], [np.inf, np.inf, np.inf]))
        out['saturating'] = (aic(f(t, *p), 3), 1 / p[2])
    except Exception:
        pass
    return min(out, key=lambda k: out[k][0]), out


def bin_(t, y, w):
    t, y = np.asarray(t), np.asarray(y); e = np.arange(t.min(), t.max() + 1, w)
    tb = [t[(t >= a) & (t < a + w)].mean() for a in e if ((t >= a) & (t < a + w)).any()]
    yb = [y[(t >= a) & (t < a + w)].mean() for a in e if ((t >= a) & (t < a + w)).any()]
    return np.array(tb), np.array(yb)


def family_summary(curves):
    wins, rates, last = [], [], []
    for t, y, floor in curves:
        if len(t) < 4:
            continue
        w, laws = fit_laws(t, y); wins.append(w); rates.append(laws['exponential'][1]); last.append(y[-1] / floor)
    if not wins:
        return None
    r = np.array(rates); bs = [np.median(r[rng.integers(0, len(r), len(r))]) for _ in range(3000)]
    frac = {k: wins.count(k) / len(wins) for k in set(wins)}
    rate_ci = [float(np.median(r))] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    return {'n': len(wins), 'winner_fraction': frac, 'exp_rate_median': rate_ci,
            'doubling_time': float(np.log(2) / np.median(r)) if np.median(r) > 0 else None,
            'drift_last_over_floor': float(np.mean(last)),
            'pass': bool(frac.get('exponential', 0) >= 0.6 and rate_ci[1] > 0)}


def oasis_curves(arm):
    cs = []
    for root in ('results_val', 'results_final', 'results_koop'):
        for f in glob.glob(f'{root}/{arm}_v*.json'):
            d = json.load(open(f)); S = np.asarray(d['shift']); fl = d['shift_floor_real']
            for row in S:
                t = np.arange(1, len(row)); tb, yb = bin_(t, row[1:], 16); cs.append((tb, yb, fl))
    return cs


def sf_curves(root, arm):
    cs = []
    for f in glob.glob(f'{root}/{arm}/*.json'):
        if f.endswith('.metrics.json'):
            continue
        d = json.load(open(f)); sh = d.get('shift') or []
        if len(sh) < 4:
            continue
        t = np.array([x[0] for x in sh], float); y = np.array([x[1] for x in sh], float)
        cs.append((t, y, float(y[:2].mean())))
    return cs


if __name__ == '__main__':
    out = {'G1': {}}
    fams = {'oasis_base': oasis_curves('base'), 'oasis_sink4': oasis_curves('sink4')}
    for root, tag in (('results_sf', '60s'), ('results_sf_long', '4min')):
        for arm in ('causvid_sink3', 'causvid_base', 'sink3', 'base'):
            fams[f'{"causvid" if arm.startswith("causvid") else "selfforcing"}_{arm.split("_")[-1]}_{tag}'] = sf_curves(root, arm)
    print('G1 growth law of uncorrected drift (winner = lowest AIC)')
    for k, cs in fams.items():
        s = family_summary(cs)
        if s is None:
            continue
        out['G1'][k] = s
        print(f"  {k:28s} n={s['n']:3d}  winners {', '.join(f'{a} {b:.0%}' for a, b in sorted(s['winner_fraction'].items()))}  "
              f"exp rate {s['exp_rate_median'][0]:+.4f} [{s['exp_rate_median'][1]:+.4f}, {s['exp_rate_median'][2]:+.4f}]/step  "
              f"last/floor {s['drift_last_over_floor']:.2f}  exp? {'PASS' if s['pass'] else 'FAIL'}")
    out['G1_pass'] = out['G1'].get('oasis_base', {}).get('pass')
    # G2
    fd = {}
    for f in glob.glob('results_val/sink4_reenc*_v*.json'):
        m = re.match(r'sink4_reenc(\d+)_v\d+\.json$', os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f)); b = np.asarray(d['fd_bins']); v = np.asarray(d['fd'])
        fd.setdefault(int(m.group(1)), []).append(float(v[b >= 256].mean()) if (b >= 256).any() else float(v[-1]))
    fd = {P: float(np.mean(v)) for P, v in sorted(fd.items())}
    out['G2'] = {'fd_by_period': fd}
    if fd:
        Pbest = min(fd, key=fd.get); out['G2']['P_best'] = Pbest
        print(f'G2 validation FD by period: {fd}  -> best P = {Pbest}')
        for P in (16, 8):
            fs = glob.glob(f'results_val/sink4_reenc{P}_v*.json')
            if not fs:
                continue
            ph = {k: [] for k in range(1, P + 1)}
            for f in fs:
                S = np.asarray(json.load(open(f))['shift'])
                for i in range(64, S.shape[1]):
                    k = i % P or P; ph[k].extend(S[:, i].tolist())
            dk = np.array([np.mean(ph[k]) for k in range(1, P + 1)])
            hit = np.where(dk >= 2 * dk[0])[0]
            Pp = int(hit[0] + 1) if len(hit) else None
            out['G2'][f'regrowth_P{P}'] = {'drift_by_phase': dk.tolist(), 'P_pred': Pp,
                                            'ratio_end_to_start': float(dk[-1] / dk[0])}
            print(f'  regrowth from sink4_reenc{P}: drift(k)/drift(1) = ' + ' '.join(f'{x / dk[0]:.2f}' for x in dk)
                  + f'  -> P_pred = {Pp if Pp else f">{P}"}')
            if 'P_pred' not in out['G2']:
                out['G2']['P_pred'] = Pp if Pp else f'>{P}'
                out['G2_pass'] = bool(Pp is not None and Pbest / 2 <= Pp <= 2 * Pbest)
    for k in ('G1_pass', 'G2_pass'):
        if k in out:
            print(k[:2], 'PASS' if out[k] else 'FAIL')
    json.dump(out, open('results_growth.json', 'w'), indent=1)

"""JEPA world-model kill-test report -> results_jwm/JWM.md.  Criteria fixed BEFORE the run:
Late horizon = second half of the open-loop rollout (steps T/2..T). Ratios = arm / none of mean late latent error,
paired bootstrap over episodes (2000 draws).
  J1 signal not blind: in the uncorrected rollout, corr(drift signal_t, latent error_t) >= 0.3 (pooled steps x episodes)
  J2 correction helps: best re-encoding period P in {1,2,4} has late-error ratio <= 0.90 with CI upper < 1
  J3 control: noise1 does not give the gain: its ratio CI includes 1, or ratio >= best re-encode ratio + 0.10
  J4 rule (training on own rollouts): the J2 ratio is lower (bigger gain) for DINO-WM (teacher-forced, 1-step) than for
     JEPA-WM (2-step rollout training) in >= 2 of the 3 environments
PROCEED TO PLANNING if J1 and J2 and J3 hold for >= 2 of 3 environments for DINO-WM."""
import glob, json, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_jwm'
rng = np.random.default_rng(0)
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{out}/*.json'))}
if not R:
    print('no results'); raise SystemExit


def late(r, arm):
    e = np.asarray(r['arms'][arm]['err']); T = e.shape[1]
    return e[:, T // 2:].mean(1)


def ratio(r, arm):
    a, b = late(r, arm), late(r, 'none')
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a)); bs.append(a[i].mean() / b[i].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a.mean() / b.mean(), lo, hi


L = ['# Course correction in JEPA world models (jepa-wms: DINO-WM, JEPA-WM) — open-loop kill test', '',
     'Latent error = ||z_pred - z_true||^2 / ||z_true||^2 on DINOv2 patch features of the true future frames (validation '
     'trajectories, logged actions, 1 real context frame). Late = second half of the rollout. Ratios < 1 = closer to the truth.', '',
     '| model | train rollout steps | T (x frameskip) | n | err step 1 | late err none | reenc1 | reenc2 | reenc4 | noise1 | oracle4 (upper bound) | drift late / floor | corr(drift, err) |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
best = {}
ok = {}
f3 = lambda t: f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
for m, r in R.items():
    e0 = np.asarray(r['arms']['none']['err']); d0 = np.asarray(r['arms']['none']['drift'])
    T = e0.shape[1]
    corr = float(np.corrcoef(d0.flatten(), e0.flatten())[0, 1])
    fl = float(np.mean(r['floor_per_step']))
    rr = {a: ratio(r, a) for a in r['arms'] if a != 'none'}
    re_ = {a: v for a, v in rr.items() if a.startswith('reenc')}
    b = min(re_.items(), key=lambda kv: kv[1][0]) if re_ else (None, (np.nan,) * 3)
    best[m] = b
    cell = lambda a: f3(rr[a]) if a in rr else '-'
    L.append(f"| {m} | {r['train_rollout_steps']} | {T} (x{r['frameskip']}) | {r['n']} | {e0[:, 0].mean():.3f} | "
             f"{e0[:, T // 2:].mean():.3f} | {cell('reenc1')} | {cell('reenc2')} | {cell('reenc4')} | {cell('noise1')} | "
             f"{cell('oracle4')} | {d0[:, T // 2:].mean() / fl:.1f}x | {corr:+.2f} |")
    j2 = b[1][0] <= 0.90 and b[1][2] < 1
    n1 = rr.get('noise1', (np.nan,) * 3)
    ok[m] = {'J1': corr >= 0.3, 'J2': bool(j2), 'J3': bool((n1[1] <= 1 <= n1[2]) or n1[0] >= b[1][0] + 0.10), 'best': b[0]}
L += ['', '## Criteria per model', '']
for m, o in ok.items():
    L.append(f"- {m}: J1 {'PASS' if o['J1'] else 'FAIL'}, J2 {'PASS' if o['J2'] else 'FAIL'} (best {o['best']}), J3 {'PASS' if o['J3'] else 'FAIL'}")
envs = sorted({m.split('_')[-1] for m in R})
j4 = sum(best.get(f'dino_wm_{e}', (0, (np.nan,)))[1][0] < best.get(f'jepa_wm_{e}', (0, (np.nan,)))[1][0] for e in envs)
L.append(f'- J4 (DINO-WM gains more than JEPA-WM): {j4}/{len(envs)} environments -> {"PASS" if j4 >= 2 else "FAIL"}')
go = sum(all(ok.get(f'dino_wm_{e}', {}).get(k, False) for k in ('J1', 'J2', 'J3')) for e in envs)
L += ['', f"**PROCEED TO PLANNING: {'YES' if go >= 2 else 'NO'}** (J1-J3 hold for DINO-WM in {go}/{len(envs)} environments)", '']
txt = '\n'.join(L); open(f'{out}/JWM.md', 'w').write(txt); print(txt)

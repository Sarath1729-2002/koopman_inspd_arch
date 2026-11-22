"""FINAL MuJoCo summary -> results_wm/FINAL_MUJOCO.md + fig_short_horizon.png (from existing result files; no compute).
1) Short-data regime (models trained on 16-step segments, deployed 180 steps): mean +- std over 3 seeds, all families.
2) Mechanism diagnostics (diag_*.json): why correction fails in recurrent-filter models (drift hidden from the decoder).
3) Planning and horizon-substitution headline numbers (planc_*.json, eval_*_long*.json) if present."""
import json, glob, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results_wm'
envs = ['cheetah', 'hopper', 'walker']
J = lambda f: json.load(open(f)) if os.path.exists(f) else {}
S = {e: J(f'{out}/short_{e}.json') for e in envs}
L = ['# MuJoCo world models — final summary', '',
     '## 1. Short-data regime: trained on 16-step segments only, deployed open-loop for 180 steps', '',
     'MSE (standardized observations) at imagined steps 101-180, mean +- std over 3 seeds.', '',
     '| env | family | no correction | re-encode every 10 | noise-injection training | noise + re-encode | 3-seed ensemble | ensemble + re-encode |',
     '|---|---|---|---|---|---|---|---|']


def ms(e, key):
    v = [S[e][key.format(s=s)]['b101_end'] for s in (0, 1, 2) if key.format(s=s) in S[e]]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


f = lambda t: f'{t[0]:.3f} +- {t[1]:.3f}' if not np.isnan(t[0]) else '-'
for e in envs:
    for X in ('kae', 'gru', 'koop'):
        b, n = f'{X}_s10', f'{X}_s10n'
        en = S[e].get(f'{b}/ens/none', {}).get('b101_end', np.nan); ep = S[e].get(f'{b}/ens/per10', {}).get('b101_end', np.nan)
        L.append(f"| {e} | {X} | {f(ms(e, b + '/s{s}/none'))} | {f(ms(e, b + '/s{s}/per10'))} | {f(ms(e, n + '/s{s}/none'))} | "
                 f"{f(ms(e, n + '/s{s}/per10'))} | {en:.3f} | {ep:.3f} |")
L += ['', 'kae = Koopman autoencoder (latent = encoder output, Fathi et al. setting); gru / koop = Dreamer-style recurrent '
          'state-space models with GRU / Koopman transition.', '']
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a_, e in zip(ax, envs):
        for key, lab in (('kae_s10/s{s}/none', 'no correction'), ('kae_s10/s{s}/per10', 're-encode every 10'),
                         ('kae_s10n/s{s}/none', 'noise-injection training')):
            curves = [np.asarray(S[e][key.format(s=s)]['mse_h']) for s in (0, 1, 2) if key.format(s=s) in S[e]]
            if curves:
                C = np.stack(curves); x = np.arange(1, C.shape[1] + 1)
                a_.plot(x, C.mean(0), label=lab); a_.fill_between(x, C.min(0), C.max(0), alpha=0.15)
        for key, lab in (('kae_s10/ens/none', 'ensemble'), ('kae_s10/ens/per10', 'ensemble + re-encode')):
            if key in S[e]:
                a_.plot(np.arange(1, len(S[e][key]['mse_h']) + 1), S[e][key]['mse_h'], ls='--', label=lab)
        a_.axvline(16, c='k', ls=':', lw=0.8); a_.set_title(f'{e}: Koopman AE trained on 16-step segments', fontsize=9)
        a_.set_xlabel('imagined step (dotted = training segment length)'); a_.set_ylabel('MSE'); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{out}/fig_short_horizon.png', dpi=150)
    L.append('Figure: fig_short_horizon.png (error vs imagined step; bands = min/max over 3 seeds).')
except Exception as ex:
    L.append(f'(figure failed: {ex})')
# diagnostics
L += ['', '## 2. Why correction fails in recurrent-filter models (seed-0 diagnostics, steps 101-180)', '',
      '| env | model | error | re-encode moves prediction (shift/err) | projected prediction error / err | rebuild from TRUE obs every 10 / err | corr(residual, future error growth) |',
      '|---|---|---|---|---|---|---|']
for e in envs:
    D = J(f'{out}/diag_{e}.json')
    for arm in ('gru_rdr', 'koop_rdr', 'kae_k10', 'kae_k50'):
        if arm in D:
            d = D[arm]; err = d['err']['101-end']
            L.append(f"| {e} | {arm} | {err:.3f} | {d['shift_full']['101-end'] / err:.3f} | {d['proj_full']['101-end'] / err:.2f} | "
                     f"{d['oracle_init10']['101-end'] / err:.2f} | {d['corr_resid_future_err_growth']:+.2f} |")
L += ['', 'Reading: decoded predictions are plausible-but-wrong states (re-encoding barely moves them, shift/err << 1), the '
          'projector itself is fine (rebuilding from true observations helps a lot), and the drift signal does not predict '
          'error growth. So the error is not off the encoder manifold in observation space.', '']
# planning headline
P = {}
for e in envs:
    for fz in glob.glob(f'{out}/planc_{e}_*.json'):
        P.setdefault(e, {}).update(J(fz))
if P:
    L += ['## 3. Planning (CEM-MPC in the real environments; 3 seeds x 6 episodes) — negative result', '',
          '| env | model | no correction | re-encode every 10 | 3-seed ensemble (disagreement penalty) |', '|---|---|---|---|---|']
    for e in envs:
        for arm in ('gru_rdr', 'kae_k10', 'kae_k50'):
            R = lambda c: np.mean(sum((P[e].get(f'{arm}/s{s}/{c}', {}).get('returns', []) for s in (0, 1, 2)), []) or [np.nan])
            ens = np.mean(P[e].get(f'{arm}/ens/none', {}).get('returns', [np.nan]))
            L.append(f'| {e} | {arm} | {R("none"):.0f} | {R("per10"):.0f} | {ens:.0f} |')
    L += ['', 'Re-encoding does not reliably improve planning return; ensembles are the stronger fix there.', '']
txt = '\n'.join(L); open(f'{out}/FINAL_MUJOCO.md', 'w').write(txt); print(txt)

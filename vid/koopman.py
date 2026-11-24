"""A Koopman view of Oasis drift (Nov 24). Needs latents saved by VID_SAVE_LAT=1 python -m vid.oasis_cc --out results_koop.
Fit, on HALF the rollouts (even rows) and only inside the training window (frames 0..31), a linear operator with
action input on PCA(64) coordinates of the latent:  y_{t+1} ~ K y_t + B a_{t+1} + c   (ridge least squares).
Then, on the OTHER half, for every frame of the 768-frame rollout:
  koopman residual   e_t = ||y_{t+1} - K y_t - B a_{t+1} - c||^2 / mean ||y||^2 (in-window)   -- a Koopman-native drift
  subspace residual  q_t = relative energy of z_t outside the in-window PCA subspace
  re-encode drift    s_t = ||E(D(z)) - z||^2 / ||z||^2 (logged by oasis_cc)
Criteria (fixed before the run):
  K1  the Koopman residual sees the drift: Spearman(binned e, binned s) over 32-frame bins pooled across held-out rollouts
      >= 0.5 AND mean e over the last 64 frames >= 2 x mean e inside the window
  K2  the operator predicts the growth law: spectral radius rho(K) > 1.01 -> exponential, < 0.99 -> saturating,
      otherwise linear; pass if it matches the AIC winner (majority) for the same rollouts in vid/growth.py
Also reported: rho, the eigenvalue magnitudes, within-bin Spearman (across rollouts at the same time; no time trend).
Usage: python -m vid.koopman [--root results_koop --arm base]  -> results_koop/koopman_<arm>.json"""
import argparse, glob, json, os, numpy as np
from scipy.stats import spearmanr
from vid.growth import fit_laws, bin_

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results_koop'); ap.add_argument('--arm', default='base')
    ap.add_argument('--r', type=int, default=64); ap.add_argument('--win', type=int, default=32)
    a = ap.parse_args()
    Z, A, S = [], [], []
    for f in sorted(glob.glob(f'{a.root}/{a.arm}_v*_lat.npy')):
        tag = os.path.basename(f)[:-8]
        z = np.load(f).astype(np.float32); Z.append(z.reshape(z.shape[0], z.shape[1], -1))
        A.append(np.load(f'{a.root}/{tag}_feats.npz')['actions'].astype(np.float32))
        S.append(np.asarray(json.load(open(f'{a.root}/{tag}.json'))['shift'], np.float32))
    if not Z:
        raise SystemExit(f'no latents in {a.root} for arm {a.arm} (run oasis_cc with VID_SAVE_LAT=1)')
    Z, A, S = np.concatenate(Z), np.concatenate(A), np.concatenate(S)
    T = min(Z.shape[1], A.shape[1] + 0, S.shape[1]); Z, S = Z[:, :T], S[:, :T]
    if A.shape[1] < T:   # actions saved for frames 0..T-1
        A = np.concatenate([A, np.zeros((A.shape[0], T - A.shape[1], A.shape[2]), np.float32)], 1)
    fit, ev = np.arange(len(Z)) % 2 == 0, np.arange(len(Z)) % 2 == 1
    W = a.win
    Xw = Z[fit, :W].reshape(-1, Z.shape[-1])
    mu = Xw.mean(0); U = np.linalg.svd(Xw - mu, full_matrices=False)[2][:a.r]      # [r, D]
    Y = (Z - mu) @ U.T                                                               # [N, T, r]
    def design(y, act):
        return np.concatenate([y, act, np.ones((*y.shape[:-1], 1), np.float32)], -1)
    Xf = design(Y[fit, :W - 1], A[fit, 1:W]).reshape(-1, a.r + A.shape[-1] + 1)
    Yf = Y[fit, 1:W].reshape(-1, a.r)
    lam = 1e-3 * np.trace(Xf.T @ Xf) / Xf.shape[1]
    M = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ Yf)            # [r+A+1, r]
    K = M[:a.r].T
    eig = np.abs(np.linalg.eigvals(K)); rho = float(eig.max())
    ny = float((Y[fit, :W] ** 2).sum(-1).mean())
    pred = design(Y[ev, :-1], A[ev, 1:]) @ M
    e = ((Y[ev, 1:] - pred) ** 2).sum(-1) / ny                                       # [n_ev, T-1], e[:, t] predicts t+1
    zc = Z[ev] - mu
    q = ((zc - (zc @ U.T) @ U) ** 2).sum(-1) / (Z[ev] ** 2).sum(-1)                  # [n_ev, T]
    s = S[ev][:, 1:]
    nb = (T - 1) // W
    eb = e[:, :nb * W].reshape(len(e), nb, W).mean(-1); sb = s[:, :nb * W].reshape(len(s), nb, W).mean(-1)
    qb = q[:, 1:nb * W + 1].reshape(len(q), nb, W).mean(-1)
    rho_es = float(spearmanr(eb.ravel(), sb.ravel())[0]); rho_qs = float(spearmanr(qb.ravel(), sb.ravel())[0])
    within = float(np.nanmean([spearmanr(eb[:, j], sb[:, j])[0] for j in range(nb)])) if len(eb) > 2 else None
    ratio = float(e[:, -64:].mean() / e[:, :W - 1].mean())
    wins = []
    for row in S[ev]:
        tb, yb = bin_(np.arange(1, T), row[1:], 16); wins.append(fit_laws(tb, yb)[0])
    law = max(set(wins), key=wins.count)
    pred_law = 'exponential' if rho > 1.01 else ('saturating' if rho < 0.99 else 'linear')
    res = {'arm': a.arm, 'n_fit': int(fit.sum()), 'n_eval': int(ev.sum()), 'rho': rho, 'eig_top10': sorted(eig.tolist())[-10:],
           'in_window_fit_residual': float(e[:, :W - 1].mean()), 'late_over_window': ratio,
           'spearman_koopman_vs_reencode': rho_es, 'spearman_subspace_vs_reencode': rho_qs, 'within_bin_spearman': within,
           'koopman_resid_by_bin': eb.mean(0).tolist(), 'reencode_by_bin': sb.mean(0).tolist(),
           'law_observed': law, 'law_counts': {k: wins.count(k) for k in set(wins)}, 'law_predicted': pred_law}
    res['K1'] = bool(rho_es >= 0.5 and ratio >= 2)
    res['K2'] = bool(pred_law == law)
    print(f"[koopman] {a.arm}: rho(K) {rho:.3f}  in-window residual {res['in_window_fit_residual']:.3f}  late/window {ratio:.2f}  "
          f"Spearman(koopman, re-encode) {rho_es:+.2f}  (subspace {rho_qs:+.2f}, within-bin {within})  "
          f"law observed {law} predicted {pred_law}")
    print('K1', 'PASS' if res['K1'] else 'FAIL', ' K2', 'PASS' if res['K2'] else 'FAIL')
    json.dump(res, open(f'{a.root}/koopman_{a.arm}.json', 'w'), indent=1)

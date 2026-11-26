"""Sparse-Koopman trigger for Oasis, step 1 (GPU, ~15 min): fit the sparse Koopman autoencoder on REAL Minecraft
latents (Oasis VAE, consecutive frames of every sample video), then
  SK1 (diagnostic, fixed before the run; the sparse-Koopman rerun of K1): on the saved validation rollouts in
      results_koop (base, sink4), support novelty (1 - best Jaccard overlap with any real-frame support) tracks the
      re-encode drift: Spearman over 32-frame bins pooled across rollouts >= 0.5 AND mean novelty over the last 64
      frames >= 2 x over the first 32. Also reported: the one-step Koopman prediction residual in the same test.
  calibration: the novelty threshold is set so the trigger fires as often as the VAE-drift trigger sink4_evt2 did on
      the validation offsets (results_val), so step 2 compares triggers at the same correction budget.
Writes results_final/skae_oasis.pt (model, PCA space, support bank, threshold, rate) and results_final/skae_oasis.json.
Step 2 = arms sink4_sk / sink4_rnd / sink4_evt2 on the TEST offsets (vid.oasis_cc), scored by vid/report_skae.py."""
import glob, json, os, sys, numpy as np, torch
from scipy.stats import spearmanr
from koop.skae import Space, train_skae, supports, novelty, pred_residual
from vid.oasis_cc import Oasis, read_frames

if __name__ == '__main__':
    oasis_dir = sys.argv[1] if len(sys.argv) > 1 else 'ext/open-oasis'
    smoke = '--smoke' in sys.argv
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(0)
    O = Oasis(oasis_dir, dev)
    vids = [v for v in sorted(glob.glob(os.path.join(oasis_dir, '**', '*.mp4'), recursive=True))
            if not os.path.basename(v).startswith(('video', 'out'))]
    Z = []
    for v in vids:
        fr = read_frames(v, 64 if smoke else 1200)
        zs = torch.cat([O.encode(fr[i:i + 64].to(dev)).cpu() for i in range(0, len(fr), 64)])
        Z.append(zs.flatten(1)); print('[skae_oasis] encoded', os.path.basename(v), tuple(zs.shape), flush=True)
    W = 16
    seqs = torch.cat([z[:len(z) // W * W].view(-1, W, z.shape[1]) for z in Z])
    allz = torch.cat(Z)
    sp = Space(allz[::2], r=128 if smoke else 256)
    Y = sp(seqs.flatten(0, 1)).view(seqs.shape[0], W, -1)
    m, k = 256, 16
    net = train_skae(Y, Y.shape[-1], m=m, k=k, steps=200 if smoke else 4000, dev=dev,
                     log=lambda s: print(s, flush=True))
    bank = supports(net, sp(allz).to(dev))
    bank = bank[torch.randperm(len(bank))[:4096]]
    nov_real = novelty(supports(net, sp(allz[1::7]).to(dev)), bank, k)
    print(f'[skae_oasis] real-frame novelty mean {nov_real.mean():.3f}', flush=True)
    res = {'n_real_frames': len(allz), 'm': m, 'k': k, 'novelty_real_mean': float(nov_real.mean())}
    # SK1 on the saved validation rollouts
    sk1 = {}
    for arm in ('base', 'sink4'):
        E, S, R = [], [], []
        for f in sorted(glob.glob(f'results_koop/{arm}_v*_lat.npy')):
            lat = torch.from_numpy(np.load(f).astype(np.float32)); B, T = lat.shape[:2]
            sh = np.asarray(json.load(open(f.replace('_lat.npy', '.json')))['shift'])
            X = sp(lat.flatten(2).flatten(0, 1)).to(dev)
            nv = novelty(supports(net, X), bank, k).view(B, T).cpu().numpy()
            Xv = X.view(B, T, -1)
            pr = pred_residual(net, Xv[:, :-1].flatten(0, 1), Xv[:, 1:].flatten(0, 1)).view(B, T - 1).cpu().numpy()
            E.append(nv); S.append(sh); R.append(pr)
        if not E:
            continue
        E, S, R = np.concatenate(E), np.concatenate(S), np.concatenate(R)
        nb = (E.shape[1] - 1) // 32
        b = lambda A, off=1: A[:, off:off + nb * 32].reshape(len(A), nb, 32).mean(-1)
        rho_n = float(spearmanr(b(E).ravel(), b(S).ravel())[0]); rho_r = float(spearmanr(b(R, 0).ravel(), b(S).ravel())[0])
        ratio = float(E[:, -64:].mean() / E[:, 1:33].mean())
        sk1[arm] = {'spearman_novelty_vs_drift': rho_n, 'spearman_predresid_vs_drift': rho_r, 'late_over_early': ratio,
                    'novelty_by_bin': b(E).mean(0).tolist(), 'pass': bool(rho_n >= 0.5 and ratio >= 2)}
        print(f'[skae_oasis] SK1 {arm}: Spearman novelty/drift {rho_n:+.2f} (pred-residual {rho_r:+.2f}), late/early {ratio:.2f}', flush=True)
        if arm == 'sink4':
            res['_sink4_nov'] = E[:, 1:].ravel()
    res['SK1'] = sk1
    res['SK1_pass'] = bool(sk1.get('base', {}).get('pass'))
    # calibration to the evt2 budget
    ev = [json.load(open(f)) for f in glob.glob('results_val/sink4_evt2_v*.json')]
    rate = float(np.mean([np.mean(e['n_reenc']) / (e['frames'] - 1) for e in ev])) if ev else 0.10
    nov_s4 = res.pop('_sink4_nov', nov_real.cpu().numpy())
    tau = float(np.quantile(nov_s4, 1 - rate))
    res.update({'rate': rate, 'tau': tau})
    print(f'[skae_oasis] evt2 budget {rate:.3f} of frames -> novelty threshold {tau:.3f}', flush=True)
    os.makedirs('results_final', exist_ok=True)
    torch.save({'net': net.state_dict(), 'd': Y.shape[-1], 'm': m, 'k': k, 'space': sp.state(), 'bank': bank,
                'tau': tau, 'rate': rate}, 'results_final/skae_oasis' + ('_smoke' if smoke else '') + '.pt')
    json.dump(res, open('results_final/skae_oasis' + ('_smoke' if smoke else '') + '.json', 'w'), indent=1)
    print('SK1', 'PASS' if res['SK1_pass'] else 'FAIL')

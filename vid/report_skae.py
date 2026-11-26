"""Sparse-Koopman trigger on Oasis, step 2 report -> results_final/SKAE.md. Criteria fixed before the run:
  SKV1 the trigger helps: sink4_sk / sink4 FD ratio (ResNet, last 128 frames, paired video-stratified bootstrap) CI upper < 1
  SKV2 it is a better trigger than the VAE-drift one at the same budget: sink4_sk / sink4_evt2 CI upper < 1
  SKV3 WHEN matters, not just how often: sink4_sk / sink4_rnd (random frames, same rate) CI upper < 1
Also reported: the fraction of frames each trigger corrected, ViT FD, KID, and sink4_reenc4 (full budget) for scale.
Reuses the statistics of vid/report_final.py (its definitions are executed, its report is not written)."""
import sys, numpy as np
src = open('vid/report_final.py').read().split('\nT = {}\n')[0]
sys.argv = ['report_final', 'results_final']
g = {'__name__': 'report_final_defs'}; exec(src, g)
boot, runs = g['boot_ratio'], g['runs']
f = lambda r: f'{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]'
L = ['# Sparse-Koopman re-encoding trigger (Open-Oasis, test offsets)', '',
     'Ratios < 1 = closer to real frames. Last 128 of 768 frames, 3 videos x 8 prompts, 95% paired bootstrap CIs.', '',
     '| comparison | FD ratio ResNet | FD ratio ViT | KID ratio ResNet | frames corrected |', '|---|---|---|---|---|']
rate = lambda m: np.mean([np.mean(r['n_reenc']) / (r['frames'] - 1) for (m_, v, s), r in runs.items() if m_ == m]) if any(k[0] == m for k in runs) else float('nan')
ok, R = {}, {}
for a, ref in (('sink4_sk', 'sink4'), ('sink4_evt2', 'sink4'), ('sink4_rnd', 'sink4'), ('sink4_reenc4', 'sink4'),
               ('sink4_sk', 'sink4_evt2'), ('sink4_sk', 'sink4_rnd')):
    if not any(k[0] == a for k in runs) or not any(k[0] == ref for k in runs):
        continue
    R[(a, ref)] = [boot(a, n, k2, ref=ref) for n, k2 in (('resnet', 'fd'), ('vit', 'fd'), ('resnet', 'kid'))]
    L.append(f'| {a} / {ref} | {f(R[(a, ref)][0])} | {f(R[(a, ref)][1])} | {f(R[(a, ref)][2])} | {rate(a):.1%} |')
for k, key in (('SKV1', ('sink4_sk', 'sink4')), ('SKV2', ('sink4_sk', 'sink4_evt2')), ('SKV3', ('sink4_sk', 'sink4_rnd'))):
    if key in R:
        ok[k] = bool(R[key][0][2] < 1)
L += ['', '## Pre-registered', ''] + [f'- {k}: {"PASS" if v else "FAIL"}' for k, v in ok.items()]
open('results_final/SKAE.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

"""Run Deep Forcing (Yi et al., ICML 2026; github.com/cvlab-kaist/DeepForcing; training-free deep sink + participative
compression on Self-Forcing) on our prompts and seeds, and file the videos like run_sf.py does so metrics / bench can
score them as arm 'deepforcing'.  Usage (from the repo root):
  python -m sf_cc.df_run --prompts 32-35 --latents 480 --out results_sf_anchor3
Needs ext/DeepForcing (cloned on the login node) with checkpoints/ and wan_models/ linked from ext/Self-Forcing."""
import argparse, glob, json, os, shutil, subprocess, sys

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', default='32-47'); ap.add_argument('--latents', type=int, default=480)
    ap.add_argument('--out', default='results_sf_anchor3'); ap.add_argument('--config', default='configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    root = os.getcwd(); DF = os.path.join(root, 'ext/DeepForcing'); out = os.path.join(root, a.out, 'deepforcing')
    os.makedirs(out, exist_ok=True)
    SFP = os.path.join(root, os.environ.get('SF_DIR', 'ext/Self-Forcing'), 'prompts')   # same prompt files as run_sf.py
    ext = [l.strip() for l in open(os.path.join(SFP, 'MovieGenVideoBench_extended.txt'), encoding='utf-8')]
    short = [l.strip() for l in open(os.path.join(SFP, 'MovieGenVideoBench.txt'), encoding='utf-8')]
    lo, hi = (int(x) for x in a.prompts.split('-'))
    for idx in range(lo, hi + 1):
        f = os.path.join(out, f'{idx:04d}.json')
        if os.path.exists(f) and not a.smoke:
            continue
        tmp = os.path.join(out, f'_tmp_{idx}'); os.makedirs(tmp, exist_ok=True)
        pf = os.path.join(tmp, 'prompt.txt'); open(pf, 'w', encoding='utf-8').write(ext[idx] + '\n')
        n_lat = 21 if a.smoke else a.latents
        cmd = [sys.executable, os.path.join(root, 'sf_cc', 'df_launch.py'), '--config_path', a.config, '--output_folder', tmp,
               '--checkpoint_path', 'checkpoints/self_forcing_dmd.pt', '--data_path', pf, '--use_ema',
               '--save_with_index', '--seed', str(1000 + idx), '--num_output_frames', str(n_lat)]
        import time; t0 = time.time()
        r = subprocess.run(cmd, cwd=DF)
        mp4s = glob.glob(os.path.join(tmp, '*.mp4'))
        if r.returncode or not mp4s:
            print(f'[deepforcing] prompt {idx} FAILED (exit {r.returncode})', flush=True); sys.exit(1)
        shutil.move(mp4s[0], os.path.join(out, f'{idx:04d}.mp4')); shutil.rmtree(tmp)
        json.dump({'arm': 'deepforcing', 'idx': idx, 'prompt': short[idx], 'latents': n_lat, 'gen_seconds': time.time() - t0,
                   'config': a.config, 'shift': []}, open(f, 'w'))
        print(f'[deepforcing] prompt {idx} done in {time.time() - t0:.0f}s', flush=True)
        if a.smoke:
            print('SMOKE OK'); break

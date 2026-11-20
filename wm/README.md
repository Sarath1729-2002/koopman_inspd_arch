# wm: course-correcting re-encoding in a Dreamer-style world model (kill test)

Setting: MuJoCo HalfCheetah / Walker2d / Hopper (Gymnasium v5, proprioceptive), random correlated actions.
World model: RSSM with discrete latents (DreamerV3-style 16x16) and either a GRU or a Koopman-style
linear transition (rotation-scaling blocks + bilinear action term, as in Koopman Dreamer, arXiv 2607.19719).
Course correction during open-loop imagination: decode -> re-encode (h <- init(enc(o_hat)), z <- posterior).
Baseline training fix: Rollout-Decoded Reconstruction (arXiv 2608.25017), arms *_rdr.

One-time setup on the login node (needs internet):
    source ~/venvs/ouro/bin/activate && pip install "mujoco>=3.2,<4" "gymnasium[mujoco]"
    python -c "import gymnasium as g; g.make('Hopper-v5').reset(); print('ok')"
Run:   sbatch slurm/wm_kill.sh            (STEPS=10000 default; resumable)
Out:   results_wm/REPORT.md   (kill criteria K1-K3 at the top)

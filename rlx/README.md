# rlvr_koop — course-correcting Koopman extrapolation of GRPO training

RELEX (2026): RLVR weight trajectories are ~rank-1 and smooth, so a line fitted to early steps extrapolates the run,
but one-shot extrapolation destabilises at long horizons. Here: Koopman dynamics on each tensor's top-3 trajectory
modes (velocity operator, ridge-shrunk to RELEX's constant velocity, spectral radius <= 1) and Fathi-style course
correction (extrapolate k steps, then a few REAL GRPO steps re-anchor the trajectory). Verdict: rlx/report.py.

    cd ~/koopman_inspd_arch && tar xzf rlvr_koop.tgz && mkdir -p logs
    module load gcc arrow python/3.11 && source ~/venvs/ouro/bin/activate && export HF_HOME=$SCRATCH/hf
    python -m rlx.prep                         # login node: model + GSM8K
    OMP_NUM_THREADS=4 bash tests/smoke_rlx.sh  # ~2 min
    sbatch slurm/rlvr_koop.sh                  # -> results_rlx/REPORT.md

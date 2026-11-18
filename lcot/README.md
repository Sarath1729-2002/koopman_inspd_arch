# latent_snap — Coconut + periodic re-encoding of continuous thoughts through the vocabulary

Precondition from the literature: SIM-CoT (ICLR 2026) shows Coconut (GPT-2, GSM8k-Aug) collapses as latent
thoughts grow, with thoughts drifting away from the vocabulary embedding space. Our drift test showed snapping
to a codebook fixes exactly that kind of drift. Arms (all from the same CoT stage-0 checkpoint):
  coconut | soft2 (soft re-encode every 2 thoughts = every reasoning step) | hard2 (hard, straight-through) | soft1 (every thought)
5 stages x c=2 -> 10 thoughts at the end (past where SIM-CoT reports collapse). Verdict: lcot/report.py.

    # login node (internet), once:
    cd ~/koopman_inspd_arch && tar xzf latent_snap.tgz && mkdir -p logs
    module load gcc arrow python/3.11 && source ~/venvs/ouro/bin/activate && export HF_HOME=$SCRATCH/hf
    python -m lcot.prep                        # GSM8k-Aug -> data_lcot/, GPT-2 -> HF cache
    OMP_NUM_THREADS=4 bash tests/smoke_lcot.sh # ~2 min
    sbatch slurm/latent_snap.sh                # ~1-1.5 h -> results_lcot/REPORT.md

#!/bin/bash
# Submit everything: Self-Forcing (pre-self-rollout checkpoint + cache-only correction) and Oasis (controls, seed 2, sweep).
cd "$(dirname "$0")/.."
[ -s ext/Self-Forcing/checkpoints/ode_init.pt ] || { echo "ode_init.pt missing: finish the download first"; exit 1; }
ARMS="ode_reenc1 ode_noise1 sink3_reenc1_cache sink3_reenc2_cache ode_base" PROMPTS=0-15 GEN_MIN=230 \
  sbatch --time=4:30:00 slurm/sf.sh
sbatch slurm/oasis_all.sh
squeue -u $USER --start

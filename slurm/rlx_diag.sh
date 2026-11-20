#!/bin/bash
# Diagnose the saved GRPO trajectory (no training): why extrapolation failed.   -> results_rlx/DIAG.md
#SBATCH --job-name=rlx_diag
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=0:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${RLX_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8
CK="${CK:-$SCRATCH/ckpt_rlx}"
[ -f "$CK/deltas/step40.pt" ] && [ -f "$CK/deltas/step150.pt" ] || { echo "saved trajectory missing in $CK (was it deleted?)"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python -m rlx.diag --ckpt "$CK" --out results_rlx

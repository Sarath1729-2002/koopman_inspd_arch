#!/bin/bash
# Viability run on a trained TRM: rollout (32 ACT steps, 2x trained depth) -> diagnose -> explore.
# Usage: CKPT=/path/to/step_XXXX sbatch slurm/analyze.sh
#SBATCH --job-name=ktrm_analyze
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
module load gcc arrow python/3.11 cuda
source ~/venvs/ktrm/bin/activate
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
: "${CKPT:?set CKPT=/path/to/TRM/step_N}"
OUT="${OUT:-runs/arc1}"
COMMON="--trm_dir third_party/TinyRecursiveModels --data_dir data/arc1concept-aug-1000 --ckpt $CKPT --out $OUT"
nvidia-smi -L
python -m ktrm.rollout  $COMMON --n_examples ${N_EX:-1024} --n_act ${N_ACT:-32} --batch_size 128
python -m ktrm.diagnose --out $OUT
python -m ktrm.explore  $COMMON --n_examples ${N_EXPLORE:-512} --rollouts 8 --sigmas 0.05,0.1,0.2 --batch_size 128
echo "== done: $OUT/REPORT.md  $OUT/EXPLORE.md  $OUT/diagnose.png"

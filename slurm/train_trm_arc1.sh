#!/bin/bash
# Train TRM on ARC-AGI-1 with the paper's settings (README: ~3 days on 4x H100 for 100k epochs).
# Only needed if find_checkpoints.py turns up nothing usable.
#   EPOCHS=100000 sbatch slurm/train_trm_arc1.sh          # full run
#   EPOCHS=20000 EVAL_INTERVAL=5000 sbatch ...            # shorter viability model (~1/5 the time, weaker)
# If TamIA's wall-time limit is below the run length, lower EPOCHS: pretrain.py can reload weights
# (load_checkpoint=...) but NOT optimizer state or the LR schedule, so a resumed run is not identical.
#SBATCH --job-name=trm_arc1
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
module load gcc arrow python/3.11 cuda
source ~/venvs/ktrm/bin/activate
export WANDB_MODE=offline
EPOCHS="${EPOCHS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
RUN="${RUN:-trm_arc1_e${EPOCHS}}"
DATA="$(pwd)/data/arc1concept-aug-1000"
CKDIR="$(pwd)/checkpoints/$RUN"
cd third_party/TinyRecursiveModels
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm data_paths="[$DATA]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  epochs=$EPOCHS eval_interval=$EVAL_INTERVAL \
  +run_name=$RUN +checkpoint_path=$CKDIR ema=True
ls -la "$CKDIR"

#!/bin/bash
# Iteration 2: correct-without-forgetting arms + long-memory recall test.
#   GPU0 130m,370m | GPU1 790m | GPU2 1.4b | GPU3 2.8b      -> results_mcc2/REPORT.md
#SBATCH --job-name=mamba_cc2
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 HF_HOME="${HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="${OUT:-results_mcc2}"; mkdir -p logs/mcc2 "$OUT"
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
run() { CUDA_VISIBLE_DEVICES=$1 python -m mcc.run2 --model state-spaces/mamba-$2-hf --out "$OUT" ${EXTRA:-} > logs/mcc2/$2.log 2>&1 \
          && grep -h "recall\|done" logs/mcc2/$2.log || echo "FAILED $2 (logs/mcc2/$2.log)"; }
( run 0 130m; run 0 370m ) & run 1 790m & run 2 1.4b & run 3 2.8b &
wait
echo "===== $(date '+%F %T') runs done"
python -m mcc.report2 --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

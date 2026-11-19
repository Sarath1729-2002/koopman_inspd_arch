#!/bin/bash
# Iteration 3 (final): learned course-correction gate on frozen pretrained Mamba, then the pre-registered decision.
#   GPU0 130m,370m | GPU1 790m | GPU2 1.4b | GPU3 2.8b       -> results_mcc3/REPORT.md
#SBATCH --job-name=mamba_cc3
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 HF_HOME="${HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="${OUT:-results_mcc3}"; mkdir -p logs/mcc3 "$OUT"
for f in mcc/run.py mcc/run2.py mcc/run3.py data/train.bin data/eval.bin; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
run() { CUDA_VISIBLE_DEVICES=$1 python -m mcc.run3 --model state-spaces/mamba-$2-hf --out "$OUT" ${EXTRA:-} > logs/mcc3/$2.log 2>&1 \
          && grep -h "train batch\|recall\|done" logs/mcc3/$2.log || echo "FAILED $2 (logs/mcc3/$2.log)"; }
( run 0 130m; run 0 370m ) & run 1 790m & run 2 1.4b & run 3 2.8b &
wait
echo "===== $(date '+%F %T') runs done"
python -m mcc.report3 --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

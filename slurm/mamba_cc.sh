#!/bin/bash
# Iteration 1: pretrained Mamba (trained at 2k) evaluated to 16k with re-encoding-from-observation arms.
#   GPU0 130m | GPU1 370m | GPU2 790m | GPU3 1.4b then 2.8b       -> results_mcc/REPORT.md
#   sbatch slurm/mamba_cc.sh      (needs data/train.bin + data/eval.bin from the lm_drift prep, and python -m mcc.prep)
#SBATCH --job-name=mamba_cc
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
OUT="${OUT:-results_mcc}"; mkdir -p logs/mcc "$OUT"
for f in data/train.bin data/eval.bin; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
run() { CUDA_VISIBLE_DEVICES=$1 python -m mcc.run --model state-spaces/mamba-$2-hf --out "$OUT" ${EXTRA:-} > logs/mcc/$2.log 2>&1 \
          && tail -1 logs/mcc/$2.log || echo "FAILED $2 (logs/mcc/$2.log)"; }
run 0 130m & run 1 370m & run 2 790m & ( run 3 1.4b; run 3 2.8b ) &
wait
echo "===== $(date '+%F %T') runs done"
python -m mcc.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

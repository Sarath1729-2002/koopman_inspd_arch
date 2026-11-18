#!/bin/bash
# Instruction re-encoding inside reasoning traces. 4 GPUs in parallel:
#   GPU0 base (+ residual states for the probe) | GPU1 re64,re512 | GPU2 re128 | GPU3 re256   -> results_rif/REPORT.md
#   sbatch slurm/reason_reenc.sh        (MODEL=... to override; must be cached: python -m rif.prep on the login node)
#SBATCH --job-name=reason_reenc
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
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"; OUT="${OUT:-results_rif}"
mkdir -p logs/rif "$OUT"
[ -f data/gsm8k_test.jsonl ] || { echo "data/gsm8k_test.jsonl missing: run python -m rif.prep on the login node"; exit 1; }
echo "===== $(date '+%F %T') node $(hostname) model $MODEL"; nvidia-smi -L
g=0
for arms in base re64,re512 re128 re256; do
  ( CUDA_VISIBLE_DEVICES=$g python -m rif.gen --model "$MODEL" --arms $arms --out "$OUT" ${EXTRA:-} > logs/rif/gpu$g.log 2>&1 \
      && tail -1 logs/rif/gpu$g.log || echo "FAILED $arms (see logs/rif/gpu$g.log)" ) &
  g=$((g+1))
done
wait
echo "===== $(date '+%F %T') generation done"
python -m rif.report --root "$OUT" --model "$MODEL"
echo "===== DONE -> $OUT/REPORT.md"

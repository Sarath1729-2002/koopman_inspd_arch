#!/bin/bash
# Depth Koopman kill test on a pretrained LLM: budgets 6/9/12 replaced layers on GPUs 0-2, full model on GPU 3.
#   sbatch slurm/depth_koop.sh          (MODEL=... to override; must already be in $HF_HOME, compute nodes are offline)
#SBATCH --job-name=depth_koop
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
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"; OUT="${OUT:-results_depth}"
mkdir -p logs/depth "$OUT"
for f in data/train.bin data/eval.bin; do [ -f $f ] || { echo "$f missing (made by the lm_drift prep)"; exit 1; }; done
[ -f data/gsm8k_test.jsonl ] || echo "WARNING: data/gsm8k_test.jsonl missing -> perplexity only"
echo "===== $(date '+%F %T') node $(hostname) model $MODEL"; nvidia-smi -L
g=0
for m in 6 9 12; do
  ( CUDA_VISIBLE_DEVICES=$g python -m dkoop.run --model "$MODEL" --budgets $m --out "$OUT" ${EXTRA:-} > logs/depth/m$m.log 2>&1 \
      || echo "FAILED m=$m (see logs/depth/m$m.log)" ) &
  g=$((g+1))
done
( CUDA_VISIBLE_DEVICES=3 python -m dkoop.run --model "$MODEL" --budgets 6 --full --out "$OUT/_full" ${EXTRA:-} > logs/depth/full.log 2>&1 \
    && cp "$OUT/_full/full.json" "$OUT/" || echo "FAILED full (see logs/depth/full.log)" ) &
wait
echo "===== $(date '+%F %T') runs done"; grep -h "ppl\|per-layer" logs/depth/*.log
python -m dkoop.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

#!/bin/bash
# Course-correcting Koopman extrapolation of GRPO (Qwen2.5-0.5B-Instruct, GSM8K).
#   GPU0 reference GRPO run, 150 steps (saves prefix deltas 1..40 + steps 60/100/150 to $SCRATCH)
#   GPU1 one-shot extrapolations (relex / line3 / linerec3 / koop3) once the prefix exists
#   GPU2 course-corrected koop3 | GPU3 course-corrected line3          -> results_rlx/REPORT.md
#SBATCH --job-name=rlvr_koop
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${RLX_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 HF_HOME="$SCRATCH/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CK="${CK:-$SCRATCH/ckpt_rlx}"; OUT="${OUT:-results_rlx}"
rm -rf "$CK"; mkdir -p "$CK" logs/rlx "$OUT"
for f in data/gsm8k_train.jsonl data/gsm8k_test.jsonl rlx/common.py; do [ -f $f ] || { echo "$f missing (python -m rlx.prep)"; exit 1; }; done
echo "===== $(date '+%F %T') node $(hostname)  ckpt $CK"; nvidia-smi -L
X="--ckpt $CK --out $OUT ${EXTRA:-}"
CUDA_VISIBLE_DEVICES=0 python -m rlx.train_ref $X > logs/rlx/ref.log 2>&1 || { echo "FAILED ref (logs/rlx/ref.log)"; touch "$CK/REF_FAILED"; tail -25 logs/rlx/ref.log; } &
CUDA_VISIBLE_DEVICES=1 python -m rlx.extrap --mode oneshot $X > logs/rlx/oneshot.log 2>&1 || echo "FAILED oneshot" &
CUDA_VISIBLE_DEVICES=2 python -m rlx.extrap --mode cc --method koop3 $X > logs/rlx/cc_koop3.log 2>&1 || echo "FAILED cc koop3" &
CUDA_VISIBLE_DEVICES=3 python -m rlx.extrap --mode cc --method line3 $X > logs/rlx/cc_line3.log 2>&1 || echo "FAILED cc line3" &
wait
echo "===== $(date '+%F %T') all done"; grep -h "acc\|weight error" logs/rlx/*.log
python -m rlx.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md   (delete $CK afterwards: ~45 GB)"

#!/bin/bash
# Deep Forcing baseline on our prompts (sf_cc/df_run.py), then metrics.  OUT / PROMPTS / LAT as in sf.sh.
#SBATCH --job-name=deepF
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset VIRTUAL_ENV PYTHONHOME; PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/venvs/' | paste -sd: -)
module load gcc arrow python/3.11 cuda
source "${SF_VENV:-$HOME/venvs/sf}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
export HF_HOME="${SF_HF_HOME:-$SCRATCH/hf_sf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
OUT=${OUT:-results_sf_anchor3}; PROMPTS=${PROMPTS:-32-47}; LAT=${LAT:-480}
mkdir -p logs/sf
[ -d ext/DeepForcing ] || { echo "MISSING ext/DeepForcing (clone it on the login node)"; exit 1; }
echo "===== $(date '+%F %T') node $(hostname)"
CUDA_VISIBLE_DEVICES=0 python -m sf_cc.df_run --prompts ${PROMPTS%-*}-${PROMPTS%-*} --smoke --out results_sf_smoke > logs/sf/df_smoke.log 2>&1
grep -q "SMOKE OK" logs/sf/df_smoke.log || { echo "SMOKE FAILED"; grep -v Warning logs/sf/df_smoke.log | tail -30; exit 1; }
lo=${PROMPTS%-*}; hi=${PROMPTS#*-}; n=$((hi - lo + 1))
for g in 0 1 2 3; do
  a0=$((lo + g * n / 4)); a1=$((lo + (g + 1) * n / 4 - 1))
  CUDA_VISIBLE_DEVICES=$g python -m sf_cc.df_run --prompts $a0-$a1 --latents $LAT --out $OUT > logs/sf/df_$a0-$a1.log 2>&1 \
    || { echo "[deepF] $a0-$a1 FAILED"; grep -v Warning logs/sf/df_$a0-$a1.log | tail -10; } &
done
wait
CUDA_VISIBLE_DEVICES=0 python -m sf_cc.metrics --root $OUT --arms deepforcing > logs/sf/metrics_deepforcing.log 2>&1
grep -h "Error\|Traceback" logs/sf/metrics_deepforcing.log | head
echo "===== $(date '+%F %T') DONE: $(ls $OUT/deepforcing/*.mp4 | wc -l) Deep Forcing videos"

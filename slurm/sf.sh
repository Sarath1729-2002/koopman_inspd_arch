#!/bin/bash
# Course correction in Self-Forcing (Wan2.1-T2V-1.3B): 32 MovieGenVideoBench prompts x 6 arms, 60 s videos (240 latent
# frames, ~3x training length) -> metrics -> results_sf/SF_REPORT.md + fig_sf.png. Needs `bash sf_cc/prep.sh` once.
# Arms: base | reenc1 | reenc2 | noise1 (equal-size random nudge control) | sink3 (attention sink) | sink3_reenc1. Resumable.
#SBATCH --job-name=sf_cc
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${SF_VENV:-$HOME/venvs/sf}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${SF_HF_HOME:-$SCRATCH/hf_sf}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
PROMPTS="${PROMPTS:-0-31}"; LAT="${LAT:-240}"; OUT=results_sf
ARMS="${ARMS:-base reenc1 noise1 sink3 sink3_reenc1 reenc2}"
mkdir -p logs/sf $OUT
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
NG=$(nvidia-smi -L | wc -l); NL=$((2 * NG))
SM=""; for a in $ARMS; do [[ $a == *reenc* ]] && { SM=$a; break; }; done; SM=${SM:-${ARMS%% *}}
CUDA_VISIBLE_DEVICES=0 python -m sf_cc.run_sf --arm $SM --smoke --out results_sf_smoke > logs/sf/smoke_${SLURM_JOB_ID:-0}.log 2>&1
grep -q "SMOKE OK" logs/sf/smoke_${SLURM_JOB_ID:-0}.log || { echo "SMOKE FAILED"; tail -40 logs/sf/smoke_${SLURM_JOB_ID:-0}.log; exit 1; }
grep "checkpoint\|generator load\|^\[" logs/sf/smoke_${SLURM_JOB_ID:-0}.log
lo=${PROMPTS%-*}; hi=${PROMPTS#*-}
NA=$(echo $ARMS | wc -w); CH=$(( (NL + NA - 1) / NA )); [ $CH -lt 2 ] && CH=2   # split prompts so jobs ~ lanes
JOBS=()
for arm in $ARMS; do
  n=$((hi - lo + 1)); for ((c = 0; c < CH; c++)); do
    a0=$((lo + c * n / CH)); a1=$((lo + (c + 1) * n / CH - 1)); [ $a1 -ge $a0 ] && JOBS+=("$arm:$a0-$a1")
  done
done
echo "${#JOBS[@]} jobs on $NL lanes: ${JOBS[*]}"
GEN=${GEN_MIN:+timeout ${GEN_MIN}m}   # optional wall-clock cap on each generation process so metrics + report still run
# 2 lanes per GPU (each process ~30 GB), lanes run their share of jobs sequentially
for ((lane = 0; lane < NL; lane++)); do
  g=$((lane % NG))
  ( for ((k = lane; k < ${#JOBS[@]}; k += NL)); do
      arm=${JOBS[$k]%:*}; pr=${JOBS[$k]#*:}
      CUDA_VISIBLE_DEVICES=$g $GEN python -m sf_cc.run_sf --arm $arm --prompts $pr --latents $LAT --out $OUT \
        > logs/sf/${arm}_${pr}.log 2>&1 || { echo "[sf] $arm $pr FAILED (exit $?)"; grep -v Warning logs/sf/${arm}_${pr}.log | tail -8; }
    done ) &
done
wait
grep -h "^\[" logs/sf/*.log | tail -20
echo "===== $(date '+%F %T') generation done: $(ls $OUT/*/*.mp4 | wc -l) videos"
g=0
for arm in $ARMS; do   # metrics skip videos already scored
  CUDA_VISIBLE_DEVICES=$((g % NG)) python -m sf_cc.metrics --root $OUT --arms $arm > logs/sf/metrics_$arm.log 2>&1 &
  g=$((g + 1))
done
wait
grep -h "Error\|Traceback" logs/sf/metrics_*.log | head
python -m sf_cc.report $OUT
echo "===== DONE -> $OUT/SF_REPORT.md  $OUT/fig_sf.png  videos in $OUT/<arm>/"

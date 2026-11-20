#!/bin/bash
# FOLLOW-UP KILL TEST (~1 h): course correction BEYOND the training horizon + course correction IN training.
#   trains koop_rdrre / gru_rdrre (3 envs) -> evaluates all RDR/plain arms at 180 imagined steps -> REPORT_long.md
# Reuses data_wm/ and ckpt_wm/ from wm_kill. Resumable.
#SBATCH --job-name=wm_long
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda mujoco/3.3.0
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
STEPS="${STEPS:-10000}"; mkdir -p logs/wm
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
ls data_wm/*.npz ckpt_wm/*.pt | wc -l
i=0
for env in cheetah walker hopper; do
  for arm in koop_rdrre gru_rdrre; do
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -m wm.train --env $env --arm $arm --steps $STEPS > logs/wm/${env}_${arm}.log 2>&1 &
    i=$((i + 1))
  done
done
# meanwhile: long-horizon eval of the already-trained arms on GPU 3
( for env in cheetah walker hopper; do
    CUDA_VISIBLE_DEVICES=3 python -m wm.evaluate --env $env --horizon 180 --tag long --arms koop,gru,koop_rdr,gru_rdr > logs/wm/long_pre_${env}.log 2>&1
  done ) &
wait
grep -h "saved\|non-finite\|Error" logs/wm/*rdrre.log
echo "===== $(date '+%F %T') training done"
g=0
for env in cheetah walker hopper; do
  CUDA_VISIBLE_DEVICES=$g python -m wm.evaluate --env $env --horizon 180 --tag long --arms koop_rdrre,gru_rdrre > logs/wm/long_${env}.log 2>&1 &
  g=$((g + 1))
done
wait
cat logs/wm/long_pre_*.log logs/wm/long_[chw]*.log | grep "\] \|Traceback\|Error"
python -m wm.report_long results_wm
echo "===== DONE -> results_wm/REPORT_long.md"

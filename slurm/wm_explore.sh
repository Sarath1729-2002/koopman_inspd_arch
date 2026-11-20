#!/bin/bash
# EXPLORATION (~1.5 h): why course correction fails in world models, and where it might not.
#   GPUs 0-2: train kae_k10, kae_k50 (faithful Fathi Koopman AE), gru_rdr10, koop_rdr10 for 3 envs (12 models)
#   GPU 3   : mechanism diagnostics on all existing models, then CEM planning in real HalfCheetah (existing models)
#   then    : diagnostics + 180-step eval + planning for the new models -> results_wm/EXPLORE.md
# Reuses data_wm/ and ckpt_wm/. Resumable (every stage skips finished work).
#SBATCH --job-name=wm_explore
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda mujoco/3.3.0
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
STEPS="${STEPS:-10000}"; mkdir -p logs/wm results_wm
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
i=0
for env in cheetah walker hopper; do
  for arm in kae_k10 kae_k50 gru_rdr10 koop_rdr10; do
    CUDA_VISIBLE_DEVICES=$((i % 3)) python -m wm.train --env $env --arm $arm --steps $STEPS > logs/wm/${env}_${arm}.log 2>&1 &
    i=$((i + 1))
  done
done
( for env in cheetah walker hopper; do
    CUDA_VISIBLE_DEVICES=3 python -m wm.diag --env $env > logs/wm/diag_pre_${env}.log 2>&1
  done
  CUDA_VISIBLE_DEVICES=3 python -m wm.plan --arms gru,gru_rdr,koop,koop_rdr > logs/wm/plan_pre.log 2>&1 ) &
wait
grep -h "saved\|non-finite\|Error" logs/wm/*_kae_k*.log logs/wm/*_rdr10.log
echo "===== $(date '+%F %T') phase A done"
g=0
for env in cheetah walker hopper; do
  ( CUDA_VISIBLE_DEVICES=$g python -m wm.diag --env $env > logs/wm/diag_${env}.log 2>&1
    CUDA_VISIBLE_DEVICES=$g python -m wm.evaluate --env $env --horizon 180 --tag long \
        --arms kae_k10,kae_k50,gru_rdr10,koop_rdr10 > logs/wm/long_new_${env}.log 2>&1 ) &
  g=$((g + 1))
done
CUDA_VISIBLE_DEVICES=3 python -m wm.plan --arms gru_rdr10,kae_k10,kae_k50 > logs/wm/plan_new.log 2>&1 &
wait
cat logs/wm/diag_pre_*.log logs/wm/diag_[chw]*.log logs/wm/plan_*.log | grep "\] \|random\|Traceback\|Error"
python -m wm.report_explore results_wm
echo "===== DONE -> results_wm/EXPLORE.md"

#!/bin/bash
# CONFIRMATION (overnight, ~4-5 h): course correction in world-model planning (3 envs, 3 seeds, ensemble and
# short-horizon baselines) + horizon substitution across seeds.  Criteria: wm/report_confirm.py docstring.
# Reuses data_wm/ and seed-0 models in ckpt_wm/. Resumable.
#SBATCH --job-name=wm_confirm
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda mujoco/3.3.0
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
STEPS="${STEPS:-10000}"; MAXJ="${MAXJ:-24}"; mkdir -p logs/wm results_wm
ARMS="gru_rdr koop_rdr gru_rdr10 koop_rdr10 kae_k10 kae_k50"
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done; }
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
# 1) extra seeds
i=0
for seed in 1 2; do for env in cheetah walker hopper; do for arm in $ARMS; do
  throttle
  CUDA_VISIBLE_DEVICES=$((i % 4)) python -m wm.train --env $env --arm $arm --seed $seed --steps $STEPS \
      > logs/wm/${env}_${arm}_s${seed}.log 2>&1 &
  i=$((i + 1))
done; done; done
wait
grep -h "saved\|non-finite\|Error" logs/wm/*_s[12].log | sort | uniq -c | sort -rn | head -5
ls ckpt_wm/*_s[12].pt | wc -l
echo "===== $(date '+%F %T') training done"
# 2) open-loop 180-step eval for the new seeds
g=0
for seed in 1 2; do for env in cheetah walker hopper; do
  CUDA_VISIBLE_DEVICES=$((g % 4)) python -m wm.evaluate --env $env --seed $seed --horizon 180 --tag long_s$seed \
      --arms ${ARMS// /,} > logs/wm/long_s${seed}_${env}.log 2>&1 &
  g=$((g + 1))
done; done
wait
echo "===== $(date '+%F %T') eval done"
# 3) planning: one process per (env, model); seeds 0-2 x {none, per10, hp10} + ensemble {none, per10}
g=0
for env in cheetah walker hopper; do for arm in $ARMS; do
  CUDA_VISIBLE_DEVICES=$((g % 4)) python -m wm.plan --env $env --arms $arm --seeds 0,1,2 --corrs none,per10,hp10 \
      --ensemble --episodes 6 --fname planc_${env}_${arm} > logs/wm/planc_${env}_${arm}.log 2>&1 &
  g=$((g + 1))
done; done
wait
grep -h "Traceback\|Error" logs/wm/planc_*.log | head
echo "===== $(date '+%F %T') planning done"
python -m wm.report_confirm results_wm
echo "===== DONE -> results_wm/CONFIRM.md"

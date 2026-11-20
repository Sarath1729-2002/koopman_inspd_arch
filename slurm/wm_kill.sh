#!/bin/bash
# KILL TEST (~1 h): course-correcting re-encoding in a Dreamer-style world model on MuJoCo.
#   data (CPU, ~5 min) -> 15 world models (3 envs x 5 arms) on 4 GPUs -> open-loop eval -> results_wm/REPORT.md
# Needs gymnasium[mujoco] in the venv (install once on the login node, see wm/README.md). Resumable.
#SBATCH --job-name=wm_kill
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore MUJOCO_GL=disable
STEPS="${STEPS:-10000}"; mkdir -p logs/wm
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
OMP_NUM_THREADS=1 python -m wm.data --procs 30 2>&1 | grep -v -i warn | grep -v '^$' || { echo DATA FAILED; exit 1; }
echo "===== $(date '+%F %T') data done"
export OMP_NUM_THREADS=2
i=0
for env in cheetah walker hopper; do
  for arm in koop koop_rdr gru gru_rdr koop_cont; do
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -m wm.train --env $env --arm $arm --steps $STEPS > logs/wm/${env}_${arm}.log 2>&1 &
    i=$((i + 1))
  done
done
wait
grep -h "saved\|non-finite\|Error" logs/wm/*.log
echo "===== $(date '+%F %T') training done"
g=0
for env in cheetah walker hopper; do
  CUDA_VISIBLE_DEVICES=$g python -m wm.evaluate --env $env > logs/wm/eval_${env}.log 2>&1 &
  g=$((g + 1))
done
wait
cat logs/wm/eval_*.log
python -m wm.report results_wm
echo "===== DONE -> results_wm/REPORT.md"

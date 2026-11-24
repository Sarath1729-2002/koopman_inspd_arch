#!/bin/bash
# JEPA world-model kill test (~1-2 h): DINO-WM vs JEPA-WM on Push-T / Wall / PointMaze, open-loop rollouts with
# decode->re-encode correction, noise control and oracle. Needs `bash jwm/prep.sh` once on the login node.
#SBATCH --job-name=jwm
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge 2>/dev/null; module load python/3.10
source "${JWM_VENV:-$SCRATCH/venvs/jwm2}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export JEPAWM_DSET=$SCRATCH/jepa_dset JEPAWM_LOGS=$SCRATCH/jepa_logs JEPAWM_CKPT=$SCRATCH/jepa_ckpt JEPAWM_OSSCKPT=$SCRATCH/jepa_ckpt
export TORCH_HOME=$SCRATCH/torch_jwm HF_HOME=$SCRATCH/hf_jwm HF_HUB_OFFLINE=1
mkdir -p logs/jwm results_jwm
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
CUDA_VISIBLE_DEVICES=0 python -m jwm.drift --model dino_wm_pusht --smoke > logs/jwm/smoke.log 2>&1
grep -q "SMOKE OK" logs/jwm/smoke.log || { echo "SMOKE FAILED"; grep -v Warning logs/jwm/smoke.log | tail -30; exit 1; }
grep "^model\|^\[" logs/jwm/smoke.log
M=(dino_wm_pusht jepa_wm_pusht dino_wm_wall jepa_wm_wall dino_wm_pointmaze jepa_wm_pointmaze)
for g in 0 1 2 3; do
  ( for ((k = g; k < ${#M[@]}; k += 4)); do
      CUDA_VISIBLE_DEVICES=$g python -m jwm.drift --model ${M[$k]} ${JWM_ARGS:-} > logs/jwm/${M[$k]}.log 2>&1 \
        || { echo "[jwm] ${M[$k]} FAILED"; grep -v Warning logs/jwm/${M[$k]}.log | tail -8; }
    done ) &
done
wait
grep -h "^model\|^\[" logs/jwm/*.log | grep -v smoke
python -m jwm.report results_jwm
echo "===== $(date '+%F %T') DONE -> results_jwm/JWM.md"

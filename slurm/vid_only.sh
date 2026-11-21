#!/bin/bash
# Part B only: course correction in the Open-Oasis 500M video world model (8 arms x 512 frames) -> results_vid/VIDEO.md
#SBATCH --job-name=vid_only
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
mkdir -p logs/vid results_vid
python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest 2>&1 | tail -5 || { echo "SELFTEST FAILED"; exit 1; }
ARMS_B=(base reenc1 stab1 reenc4 stab50 reenc16 stab100 reenc1_stab50)
for g in 0 1 2 3; do
  ( for k in 0 1; do arm=${ARMS_B[$((g * 2 + k))]}
      CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --frames 512 \
          --out results_vid > logs/vid/$arm.log 2>&1 || { echo "[B] $arm FAILED"; tail -5 logs/vid/$arm.log; }
    done ) &
done
wait
grep -h "^\[" logs/vid/*.log
python -m vid.report_vid results_vid

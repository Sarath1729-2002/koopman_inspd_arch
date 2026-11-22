#!/bin/bash
# Video world model v3 (~1-1.5 h): 8 arms x 3 sample videos (4 prompts each), 512 frames -> results_vid3/VIDEO2.md
# Arms: base, base_s1 (seed noise) | reenc4 (replication) | reenc4_lf (primary: structure-only re-encode) |
# reenc4_lf_s1 | reenc4_mp (mean-preserving) | reenc1_lf | evt2_lf
#SBATCH --job-name=vid3
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
mkdir -p logs/vid3 results_vid3
python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest 2>&1 | grep -v Warning | tail -3 || { echo "SELFTEST FAILED"; exit 1; }
RUNS=()
for v in 0 1 2; do for arm in base reenc4_lf base_s1 reenc4_lf_s1 reenc4 reenc4_mp reenc1_lf evt2_lf; do RUNS+=("$arm:$v"); done; done
for g in 0 1 2 3; do
  ( for ((k = g; k < ${#RUNS[@]}; k += 4)); do
      arm=${RUNS[$k]%:*}; v=${RUNS[$k]#*:}
      CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames 512 \
          --out results_vid3 > logs/vid3/${arm}_v$v.log 2>&1 || { echo "[v2] $arm v$v FAILED"; tail -4 logs/vid3/${arm}_v$v.log; }
    done ) &
done
wait
grep -h "^\[" logs/vid3/*.log
python -m vid.report_vid3 results_vid3

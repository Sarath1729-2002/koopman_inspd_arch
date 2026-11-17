#!/bin/bash
# Phase 1 grid: 3 tasks x 4 arms x 3 seeds = 36 small runs, 8 at a time (2 per H100), then the report.
# Resumable: finished runs (their json exists) are skipped.
#   sbatch slurm/phase1.sh
#SBATCH --job-name=koop_phase1
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ktrm}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2
TASKS="${TASKS:-parity copy addition}"; ARMS="${ARMS:-plain gainctl koopman koopman_re}"; SEEDS="${SEEDS:-0 1 2}"
PAR="${PAR:-8}"
mkdir -p runs logs/phase1
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L

jobs=()
for s in $SEEDS; do for t in $TASKS; do for a in $ARMS; do
  [ -f "runs/$t/${a}_s$s.json" ] && continue
  jobs+=("$t $a $s")
done; done; done
echo "runs to do: ${#jobs[@]}"
i=0
for j in "${jobs[@]}"; do
  set -- $j
  gpu=$(( i % 4 )); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu python -m klg.train --task $1 --arm $2 --seed $3 ${EXTRA:-} > logs/phase1/$1_$2_s$3.log 2>&1 \
      && tail -1 logs/phase1/$1_$2_s$3.log || echo "FAILED $1 $2 s$3 (see logs/phase1/$1_$2_s$3.log)" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo "===== $(date '+%F %T') training done"
python -m klg.report --root runs
echo "===== DONE -> runs/REPORT.md runs/phase1.png"

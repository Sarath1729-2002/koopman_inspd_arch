#!/bin/bash
# Drift kill test: 2 tasks x 8 arms x 3 seeds = 48 tiny runs, 16 at a time on one H100 node (~30-45 min).
# Resumable (finished runs are skipped).   sbatch slurm/run.sh
#SBATCH --job-name=drift_kill
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ktrm}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2
TASKS="${TASKS:-parity mod5}"; ARMS="${ARMS:-lstm rot unit norm16 re4 re16 re32 ure16}"; SEEDS="${SEEDS:-0 1 2}"; PAR="${PAR:-16}"
OUT="${OUT:-runs_drift}"
mkdir -p "$OUT" logs/drift
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
jobs=()
for s in $SEEDS; do for t in $TASKS; do for a in $ARMS; do
  [ -f "$OUT/$t/${a}_s$s.json" ] && continue
  jobs+=("$t $a $s")
done; done; done
echo "runs to do: ${#jobs[@]}"
i=0
for j in "${jobs[@]}"; do
  set -- $j
  gpu=$(( i % 4 )); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu python -m dk.train --task $1 --arm $2 --seed $3 --out "$OUT" ${EXTRA:-} > logs/drift/$1_$2_s$3.log 2>&1 \
      && tail -1 logs/drift/$1_$2_s$3.log || echo "FAILED $1 $2 s$3 (see logs/drift/$1_$2_s$3.log)" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo "===== $(date '+%F %T') training done"
python -m dk.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

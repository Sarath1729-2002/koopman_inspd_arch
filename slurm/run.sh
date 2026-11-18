#!/bin/bash
# S5 kill test: 7 arms x 3 seeds = 21 small runs, 8 at a time on one H100 node, then the report.
# Resumable (finished runs are skipped).   sbatch slurm/run.sh
#SBATCH --job-name=bnd_kill
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ktrm}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2
ARMS="${ARMS:-lstm deltaprod lrnn bnd2 bnd4 bnd8 bnd16}"; SEEDS="${SEEDS:-0 1 2}"; PAR="${PAR:-8}"
GROUP="${GROUP:-s5}"; OUT="${OUT:-runs_$GROUP}"
mkdir -p "$OUT" logs/runs
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
jobs=()
for s in $SEEDS; do for a in $ARMS; do
  [ -f "$OUT/${a}_s$s.json" ] && continue
  jobs+=("$a $s")
done; done
echo "runs to do: ${#jobs[@]}"
i=0
for j in "${jobs[@]}"; do
  set -- $j
  gpu=$(( i % 4 )); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu python -m bk.train --arm $1 --seed $2 --group "$GROUP" --out "$OUT" ${EXTRA:-} > logs/runs/$1_s$2.log 2>&1 \
      && tail -1 logs/runs/$1_s$2.log || echo "FAILED $1 s$2 (see logs/runs/$1_s$2.log)" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo "===== $(date '+%F %T') training done"
python -m bk.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

#!/bin/bash
# OVERNIGHT PROOF OF CONCEPT: Koopman course correction vs the seriality gap (generative, 3 seeds).
#   GPU0: block48 s0-2 -> featdiff s0     GPU1: block8 s0-2 -> featdiff s1
#   GPU2: block1  s0-2 -> featdiff s2     GPU3: koopbase s0-2 -> koopdiff s0-2
#   then evaluation of seeds 0/1/2 in parallel, then the report + figure.     -> results_sg2/REPORT.md, poc.png
# Resumable: resubmitting skips finished checkpoints.     sbatch slurm/poc_overnight.sh
#SBATCH --job-name=poc_koop
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${POC_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4
CK="${CK:-ckpt_sg2}"; OUT="${OUT:-results_sg2}"; mkdir -p logs/poc "$CK" "$OUT"
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
tr() { CUDA_VISIBLE_DEVICES=$1 python -m sg2.train --ckpt "$CK" ${EXTRA:-} "${@:2}" >> logs/poc/gpu$1.log 2>&1 \
         || echo "FAILED training on GPU$1: ${*:2} (logs/poc/gpu$1.log)"; tail -1 logs/poc/gpu$1.log; }
( for s in 0 1 2; do tr 0 --kind block --k 48 --seed $s; done; tr 0 --kind featdiff --seed 0 ) &
( for s in 0 1 2; do tr 1 --kind block --k 8  --seed $s; done; tr 1 --kind featdiff --seed 1 ) &
( for s in 0 1 2; do tr 2 --kind block --k 1  --seed $s; done; tr 2 --kind featdiff --seed 2 ) &
( for s in 0 1 2; do tr 3 --kind koopbase --seed $s; done; for s in 0 1 2; do tr 3 --kind koopdiff --seed $s; done ) &
wait
echo "===== $(date '+%F %T') training done"; ls -la "$CK"
for s in 0 1 2; do
  ( CUDA_VISIBLE_DEVICES=$s python -m sg2.evaluate --seed $s --ckpt "$CK" --out "$OUT" > logs/poc/eval_s$s.log 2>&1 \
      || { echo "FAILED eval seed $s"; tail -20 logs/poc/eval_s$s.log; } ) &
done
wait
echo "===== $(date '+%F %T') evaluation done"; grep -h "passes" logs/poc/eval_s0.log | head -40
python -m sg2.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md  $OUT/poc.png"

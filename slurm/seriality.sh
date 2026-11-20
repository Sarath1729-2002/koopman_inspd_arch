#!/bin/bash
# Seriality gap vs course-correcting Koopman (hard spheres, state space). 4 arms train in parallel, then eval + report.
#   sbatch slurm/seriality.sh        -> results_sg/REPORT.md
#SBATCH --job-name=seriality
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${SG_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4
CK="${CK:-ckpt_sg}"; OUT="${OUT:-results_sg}"; mkdir -p logs/sg "$CK" "$OUT"
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
g=0
for arm in parallel ar koop koop1; do
  ( CUDA_VISIBLE_DEVICES=$g python -m sg.train --arm $arm --ckpt "$CK" ${EXTRA:-} > logs/sg/$arm.log 2>&1 \
      && tail -1 logs/sg/$arm.log || { echo "FAILED $arm"; tail -20 logs/sg/$arm.log; } ) &
  g=$((g+1))
done
wait
echo "===== $(date '+%F %T') training done"
CUDA_VISIBLE_DEVICES=0 python -m sg.evaluate --ckpt "$CK" --out "$OUT" 2>&1 | tee logs/sg/eval.log | grep -v Warn
python -m sg.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

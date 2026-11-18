#!/bin/bash
# LM drift test: 4 byte LMs (rglru/mamba x base/re64) trained at context 512 in parallel (1 per GPU),
# then evaluated at 8192 with test-time arms.  Needs data/ from scripts/prep_data.py (login node).  sbatch slurm/run.sh
#SBATCH --job-name=lm_drift
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
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4
mkdir -p logs/lm ckpt results
[ -f data/train.bin ] || { echo "data/train.bin missing: run python scripts/prep_data.py on the login node"; exit 1; }
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
g=0
for core in rglru mamba; do for k in 0 64; do
  arm=$([ $k = 0 ] && echo base || echo re$k)
  if [ ! -f ckpt/${core}_${arm}.pt ]; then
    ( CUDA_VISIBLE_DEVICES=$g python -m lmd.train --core $core --reenc_every $k ${EXTRA:-} > logs/lm/train_${core}_${arm}.log 2>&1 \
        && tail -1 logs/lm/train_${core}_${arm}.log || echo "FAILED train $core $arm" ) &
  fi
  g=$((g+1))
done; done
wait
echo "===== $(date '+%F %T') training done"; tail -n 2 logs/lm/train_*.log
g=0
for c in rglru_base rglru_re64 mamba_base mamba_re64; do
  ( CUDA_VISIBLE_DEVICES=$g python -m lmd.evaluate --only $c > logs/lm/eval_$c.log 2>&1 && grep -v Warn logs/lm/eval_$c.log | grep "_" \
      || echo "FAILED eval $c" ) &
  g=$((g+1))
done
wait
python -m lmd.report --root results
echo "===== DONE -> results/REPORT.md"

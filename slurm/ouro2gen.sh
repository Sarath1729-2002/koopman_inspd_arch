#!/bin/bash
# U2 rerun (rdiag/odir2_gen.py): GSM8k answer accuracy with/without steering; GPU0 Ouro-1.4B, GPU1 Ouro-2.6B. ~3 h
#SBATCH --job-name=ouro2gen
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset VIRTUAL_ENV PYTHONHOME; PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/venvs/' | paste -sd: -)
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME="${RD_HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs/ouro2
echo "===== $(date '+%F %T') node $(hostname)"
CUDA_VISIBLE_DEVICES=0 python -m rdiag.odir2_gen --smoke > logs/ouro2/gen_smoke.log 2>&1 && echo "SMOKE OK" \
  || { echo "SMOKE FAILED"; tail -20 logs/ouro2/gen_smoke.log; exit 1; }
( CUDA_VISIBLE_DEVICES=0 python -m rdiag.odir2_gen > logs/ouro2/gen.log 2>&1 || { echo "[gen] 1.4B FAILED"; tail -15 logs/ouro2/gen.log; } ) &
( CUDA_VISIBLE_DEVICES=1 python -m rdiag.odir2_gen --model ByteDance/Ouro-2.6B --tag 26 --bs 8 > logs/ouro2/gen_26.log 2>&1 \
    || { echo "[gen] 2.6B FAILED"; tail -15 logs/ouro2/gen_26.log; } ) &
wait
grep -h "^\[odir2_gen\]\|^U2" logs/ouro2/gen*.log
python -m vid.checks_report > /dev/null; echo "===== $(date '+%F %T') DONE -> CHECKS.md"

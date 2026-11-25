#!/bin/bash
# Ouro damage-direction validation (rdiag/odir2.py; criteria N1-N3, U1-U2, KP1-KP2, A1-A2, OH2, G1o in its docstring)
#   GPU0 Ouro-1.4B   -> results_rdiag/odir2.json
#   GPU1 Ouro-2.6B   -> results_rdiag/odir2_26.json   (skipped if not in the HF cache)
# Other-text check uses wikitext-103 test if cached (prep on the login node, see below), else it is skipped.
#SBATCH --job-name=ouro2
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# drop any venv inherited from the submitting shell (sbatch copies the environment)
unset VIRTUAL_ENV PYTHONHOME; PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/venvs/' | paste -sd: -)
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME="${RD_HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs/ouro2 results_rdiag
echo "===== $(date '+%F %T') node $(hostname)"
for f in data_lcot/test.jsonl data_lcot/train.jsonl; do [ -e $f ] || echo "MISSING $f"; done
( [ -s results_rdiag/odir2.json ] || CUDA_VISIBLE_DEVICES=0 python -m rdiag.odir2 > logs/ouro2/odir2.log 2>&1 \
    || { echo "[ouro2] 1.4B FAILED"; tail -15 logs/ouro2/odir2.log; } ) &
( [ -s results_rdiag/odir2_26.json ] || CUDA_VISIBLE_DEVICES=1 python -m rdiag.odir2 --model ByteDance/Ouro-2.6B --tag 26 --bs 8 \
    > logs/ouro2/odir2_26.log 2>&1 || { echo "[ouro2] 2.6B FAILED/skipped"; tail -6 logs/ouro2/odir2_26.log; } ) &
wait
grep -h "^\[odir2\]\|PASS\|FAIL" logs/ouro2/*.log
python -m vid.checks_report > /dev/null; echo "===== $(date '+%F %T') DONE -> CHECKS.md"

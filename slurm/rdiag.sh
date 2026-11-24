#!/bin/bash
# Reasoning-side drift diagnostics (~2-3 h): does the "errors leave the manifold" condition hold in
#   GPU0: Ouro-1.4B looped past its trained depth (4 -> 12 loops)      -> results_rdiag/ouro.json
#   GPU1: Coconut continuous thoughts (trains the plain Coconut arm once from ckpt_lcot/stage0.pt, then 10/14/20
#         thoughts with and without vocabulary re-encoding)            -> results_rdiag/coconut.json
#   GPU2: Ouro-2.6B, if it is in the HF cache                          -> results_rdiag/ouro26.json
# Needs lcot/, data_lcot/, ckpt_lcot/stage0.pt from the earlier latent_snap run, and Ouro in $SCRATCH/hf.
#SBATCH --job-name=rdiag
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 HF_HOME="${RD_HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs/rdiag results_rdiag
echo "===== $(date '+%F %T') node $(hostname)"
for f in data_lcot/test.jsonl ckpt_lcot/stage0.pt; do [ -e $f ] || echo "MISSING $f"; done
[[ "${ONLY:-all}" =~ ^(all|ouro)$ ]] && CUDA_VISIBLE_DEVICES=0 python -m rdiag.ouro > logs/rdiag/ouro.log 2>&1 || { echo "[rdiag] ouro FAILED"; tail -15 logs/rdiag/ouro.log; } &
[[ "${ONLY:-all}" =~ ^(all|coconut)$ ]] && CUDA_VISIBLE_DEVICES=1 python -m rdiag.coconut > logs/rdiag/coconut.log 2>&1 || { echo "[rdiag] coconut FAILED"; tail -15 logs/rdiag/coconut.log; } &
[[ "${ONLY:-all}" == all ]] && ( CUDA_VISIBLE_DEVICES=2 python -m rdiag.ouro --model ByteDance/Ouro-2.6B --out results_rdiag/o26 > logs/rdiag/ouro26.log 2>&1 \
    && mv results_rdiag/o26/ouro.json results_rdiag/ouro26.json || echo "[rdiag] ouro-2.6B skipped/failed (see logs/rdiag/ouro26.log)" ) &
wait
grep -h "^\[\|PASS\|FAIL" logs/rdiag/ouro.log logs/rdiag/coconut.log logs/rdiag/ouro26.log
echo "===== $(date '+%F %T') DONE -> results_rdiag/"

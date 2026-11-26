#!/bin/bash
# Two kinds of drift + benchmark table, on the videos we already have (~2-3 h, mostly video decoding).
#   1. sf_cc/split.py  (sf venv, Wan VAE)     CausVid / Self-Forcing at 4 min, without and with re-encoding -> SP1, SP2
#   2. sf_cc/split.py  (ouro venv, Oasis VAE) Oasis sink4 / sink4_reenc4                              -> reported
#   3. sf_cc/bench.py  (CPU)                  VBench-Long-style dims + ID-Forcing colour / motion drift scores
# Results: results_split/SPLIT.md, results_split/BENCH.md
#SBATCH --job-name=split
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset VIRTUAL_ENV PYTHONHOME; PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/venvs/' | paste -sd: -)
module load gcc arrow python/3.11 cuda
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs/split results_split
echo "===== $(date '+%F %T') node $(hostname)"
( source "${SF_VENV:-$HOME/venvs/sf}/bin/activate"; export HF_HOME="${SF_HF_HOME:-$SCRATCH/hf_sf}"
  python sf_cc/bench.py > logs/split/bench.log 2>&1 || { echo "[split] bench FAILED"; tail -10 logs/split/bench.log; } ) &
( source "${SF_VENV:-$HOME/venvs/sf}/bin/activate"; export HF_HOME="${SF_HF_HOME:-$SCRATCH/hf_sf}"
  CUDA_VISIBLE_DEVICES=0 SPLIT_SKIP_OASIS=1 python -m sf_cc.split --smoke > logs/split/smoke.log 2>&1 \
    || { echo "[split] SMOKE FAILED"; tail -20 logs/split/smoke.log; exit 1; }
  echo "[split] smoke ok"
  CUDA_VISIBLE_DEVICES=0 SPLIT_SKIP_OASIS=1 python -m sf_cc.split > logs/split/wan.log 2>&1 \
    || { echo "[split] Wan part FAILED"; tail -20 logs/split/wan.log; } ) &
( source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
  CUDA_VISIBLE_DEVICES=1 SPLIT_SKIP_WAN=1 python -m sf_cc.split > logs/split/oasis.log 2>&1 \
    || { echo "[split] Oasis part FAILED"; tail -20 logs/split/oasis.log; } ) &
wait
source "${SF_VENV:-$HOME/venvs/sf}/bin/activate"
SPLIT_SKIP_WAN=1 SPLIT_SKIP_OASIS=1 python -m sf_cc.split | tail -14
cat results_split/BENCH.md
echo "===== $(date '+%F %T') DONE"

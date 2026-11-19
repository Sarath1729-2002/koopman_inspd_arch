#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1
T=$(mktemp -d)
python tests/make_tiny_mamba.py $T
python -m mcc.run3 --model $T/m --data $T --W 64 --every 16 --seg 8 --train_len 256 --train_seqs 2 --train_batches 1 \
  --L 512 --n_seq 2 --n_recall 3 --gaps 16,96 --n_calib 8 --out $T/res --device cpu | tail -4
python -m mcc.report3 --root $T/res | tail -4
echo "SMOKE OK ($T)"

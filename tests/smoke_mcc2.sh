#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1
T=$(mktemp -d)
python tests/make_tiny_mamba.py $T
python -m mcc.run2 --model $T/m --data $T --W 64 --L 512 --n_seq 2 --n_recall 3 --gaps 16,96 --n_calib 8 --every 16 --out $T/res --device cpu | tail -3
python -m mcc.report2 --root $T/res | tail -6
echo "SMOKE OK ($T)"

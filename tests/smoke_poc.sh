#!/bin/bash
# CPU smoke (~3 min): every model kind trains a few steps, every sampler/calibration path runs, report + figure.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
T=$(mktemp -d)
for spec in "block --k 48" "block --k 8" "block --k 1" "koopbase" "koopdiff" "featdiff"; do
  python -m sg2.train --kind $spec --seed 0 --n_train 400 --steps 20 --batch 16 --d 32 --layers 2 --ckpt $T/ck --device cpu | tail -1
done
python -m sg2.evaluate --seed 0 --ckpt $T/ck --n_test 30 --n_val 16 --N 3 --N_sweep 2 --budgets 2,4 --bs 15 --d 32 --layers 2 --out $T/res --device cpu | tail -2
python -m sg2.report --root $T/res | tail -4
echo "SMOKE OK ($T)"

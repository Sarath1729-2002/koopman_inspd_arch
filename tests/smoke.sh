#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
OUT=$(mktemp -d)
for a in lstm rot unit norm16 re16 ure16; do
  python -m dk.train --task mod5 --arm $a --steps 30 --batch 32 --train_len 16 --eval_lengths 16,64 --n_eval 64 --device cpu --out "$OUT" | tail -1
done
python -m dk.report --root "$OUT" | tail -4
echo "SMOKE OK ($OUT)"

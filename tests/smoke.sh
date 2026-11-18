#!/bin/bash
# CPU smoke test (~1-2 min): every arm trains a few steps, evaluates, and the report runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
OUT=$(mktemp -d)
for a in lstm deltaprod lrnn bnd4 bnd8; do
  python -m bk.train --arm $a --steps 40 --batch 32 --train_len 16 --eval_lengths 16,32,64 --n_eval 64 \
    --device cpu --out "$OUT" | tail -1
done
python -m bk.report --root "$OUT" | tail -8
echo "SMOKE OK ($OUT)"

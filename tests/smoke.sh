#!/bin/bash
# CPU check of the whole pipeline (~1-2 min): 2 tasks x 4 arms x 1 seed, tiny models, then the report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4
R=runs_smoke; rm -rf $R
for t in parity addition; do for a in plain gainctl koopman koopman_re; do
  python -m klg.train --task $t --arm $a --steps 60 --d 32 --device cpu --n_eval 16 --train_max 4 --eval_lengths 3,6 --out $R | tail -1
done; done
python -m klg.report --root $R | tail -3
rm -rf $R; echo "SMOKE OK"

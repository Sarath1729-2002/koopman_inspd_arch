#!/bin/bash
# CPU smoke (~5 min): tiny training of all 4 arms, evaluation with every schedule, report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
T=$(mktemp -d)
for arm in parallel ar koop koop1; do python -m sg.train --arm $arm --n_train 1000 --steps 60 --batch 32 --ckpt $T/ck --device cpu | tail -1; done
python -m sg.evaluate --ckpt $T/ck --n_test 60 --taus 0.2,0.5 --out $T/res --device cpu | tail -3
python -m sg.report --root $T/res | tail -6
echo "SMOKE OK ($T)"

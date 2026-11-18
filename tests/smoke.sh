#!/bin/bash
# CPU smoke (~1 min) on synthetic bytes: trains 2 tiny models, runs every eval arm and the report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
T=$(mktemp -d)
python -c "
import numpy as np, sys
t = b''.join(open(f,'rb').read() for f in ['lmd/model.py','lmd/evaluate.py','lmd/train.py']) * 300
np.frombuffer(t[:-60000], dtype=np.uint8).tofile('$T/train.bin'); np.frombuffer(t[-60000:], dtype=np.uint8).tofile('$T/eval.bin')"
python -m lmd.train --core rglru --steps 20 --batch 4 --d 64 --layers 2 --data $T --out $T/ck --device cpu | tail -1
python -m lmd.train --core mamba --reenc_every 64 --steps 20 --batch 4 --d 64 --layers 2 --data $T --out $T/ck --device cpu | tail -1
python -m lmd.evaluate --ckpt $T/ck --data $T --eval_len 2048 --n_seq 4 --out $T/res --device cpu 2>&1 | grep -v Warn
python -m lmd.report --root $T/res | tail -3
echo "SMOKE OK ($T)"

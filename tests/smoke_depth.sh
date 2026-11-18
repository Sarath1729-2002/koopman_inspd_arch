#!/bin/bash
# CPU smoke test (~1 min): tiny random Qwen2, fits, all arms, report. Self-contained.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1
T=$(mktemp -d)
python tests/make_tiny.py $T/tiny
mkdir -p $T/data && python -c "
import numpy as np
t=open('dkoop/run.py','rb').read()*60
np.frombuffer(t,dtype=np.uint8).tofile('$T/data/train.bin'); np.frombuffer(t[:200000],dtype=np.uint8).tofile('$T/data/eval.bin')"
printf '{"question": "What is 2+3?", "answer": "2+3=5\\n#### 5"}\n%.0s' 1 2 3 4 > $T/gsm.jsonl
python -m dkoop.run --model $T/tiny --budgets 2,3 --n_calib 16 --n_ppl 8 --n_gsm 4 --T 64 --bs 4 --gsm8k $T/gsm.jsonl --data $T/data \
  --full --out $T/res --device cpu
python -m dkoop.report --root $T/res | tail -4
echo "SMOKE OK ($T)"

#!/bin/bash
# CPU smoke (~1-2 min): tiny random GPT-2 + synthetic data; stage 0, all 4 arms (2 stages), report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1
T=$(mktemp -d)
python tests/make_tiny_gpt2.py $T
X="--model $T/m --data $T/data --stages 2 --stage0_epochs 1 --epochs_per_stage 1 --batch 16 --eval_bs 32 --device cpu --out $T/res --stage0_ckpt $T/ck/s0.pt"
python -m lcot.train --arm cot --do_stage0 $X | tail -1
for spec in "coconut none 2" "soft2 soft 2" "hard2 hard 2" "soft1 soft 1"; do set -- $spec
  python -m lcot.train --arm $1 --snap $2 --k $3 $X | tail -1; done
python -m lcot.report --root $T/res | tail -12
echo "SMOKE OK ($T)"

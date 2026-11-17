#!/bin/bash
# CPU end-to-end run on a mock looped model + fake data (~1 min). Numbers are meaningless by design.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
SP=runs/smoke_data
python tests/make_fake_data.py $SP >/dev/null
A="--mock --data_dir $SP --out runs/smoke --t_train 3 --t_max 7 --device cpu --batch_size 8"
python -m kl.check $A | grep -E "check|inspect\]"
python -m kl.fit $A --n_train 40 --tok_per_seq 16 --rank 8 --kae_epochs 3 | grep "\[fit\]"
python -m kl.evaluate $A --n_gsm 16 --n_arc 12 | tail -1
python -m kl.generate $A --n 8 --shots 1 --max_new 6 | tail -3
echo "SMOKE OK"

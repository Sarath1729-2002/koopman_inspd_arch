#!/bin/bash
# CPU smoke (~1-2 min): tiny random Mamba; checks step decoding == full forward, then all arms + report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1
T=$(mktemp -d)
python tests/make_tiny_mamba.py $T
python - <<PY
import torch
from mcc.run import load, make_cache, step
tok, m = load("$T/m", "cpu", torch.float32)
x = torch.randint(0, m.config.vocab_size, (3, 40))
with torch.no_grad(): full = m(x).logits
c = make_cache(m, 3, "cpu", torch.float32)
st = torch.stack([step(m, c, x[:, t]) for t in range(40)], 1)
print("step decode vs full forward, max |logit diff|:", (st - full).abs().max().item())
assert (st - full).abs().max().item() < 1e-3
PY
python -m mcc.run --model $T/m --data $T --W 64 --L 512 --n_seq 3 --n_calib 8 --every 16 --out $T/res --device cpu | tail -1
python -m mcc.report --root $T/res | tail -8
echo "SMOKE OK ($T)"

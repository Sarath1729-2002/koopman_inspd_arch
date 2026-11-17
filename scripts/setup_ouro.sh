#!/bin/bash
# LOGIN node, one time (needs internet): env, Ouro-1.4B weights + remote code, GSM8K / ARC-C as JSONL.
set -euo pipefail
cd "$(dirname "$0")/.."
module load gcc arrow python/3.11 cuda
[ -d ~/venvs/ouro ] || virtualenv --no-download ~/venvs/ouro
source ~/venvs/ouro/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch
# Ouro's remote code needs transformers < 4.56 (model card recommends 4.54.1)
pip install --prefer-binary "transformers==4.54.1" accelerate datasets numpy scipy huggingface_hub
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
MODEL="${MODEL:-ByteDance/Ouro-1.4B}"
hf download "$MODEL"
# load config + tokenizer once so the remote code lands in $HF_HOME/modules (compute nodes are offline)
python - "$MODEL" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer
m = sys.argv[1]
cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
AutoTokenizer.from_pretrained(m, trust_remote_code=True)
print("config:", {k: getattr(cfg, k, None) for k in ("num_hidden_layers", "hidden_size", "total_ut_steps", "early_exit_threshold")})
PY
mkdir -p data
python - <<'PY'
from datasets import load_dataset
for s in ("train", "test"):
    load_dataset("openai/gsm8k", "main", split=s).to_json(f"data/gsm8k_{s}.jsonl")
load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").to_json("data/arc_challenge_test.jsonl")
PY
wc -l data/*.jsonl
python -c "import torch, transformers; print('env ok', torch.__version__, transformers.__version__)"

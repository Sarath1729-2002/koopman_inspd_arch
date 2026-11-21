#!/bin/bash
# ONE-TIME SETUP on the LOGIN NODE (needs internet): Open-Oasis code + weights + ResNet-50 weights + python deps.
# Usage (from ~/koopman_inspd_arch):  bash vid/prep.sh
set -uo pipefail
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
mkdir -p ext && cd ext
[ -d open-oasis ] || git clone https://github.com/etched-ai/open-oasis.git || { echo "!! git clone failed"; exit 1; }
cd open-oasis
for p in einops timm av diffusers safetensors scikit-learn scipy; do
  pip install --no-index "$p" >/dev/null 2>&1 || pip install "$p" >/dev/null 2>&1 || echo "!! could not install $p"
done
python -m pip show huggingface_hub >/dev/null 2>&1 || pip install --no-index huggingface_hub || pip install huggingface_hub
# weights: official (gated: accept the terms at https://huggingface.co/Etched/oasis-500m and `huggingface-cli login`)
# fall back to the public mirror camenduru/oasis-500m
get() { python - "$@" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, names = sys.argv[1], sys.argv[2:]
for n in names:
    try:
        print(hf_hub_download(repo, n, local_dir='.')); break
    except Exception as e:
        print('  miss', repo, n, type(e).__name__)
PY
}
ls oasis500m.* >/dev/null 2>&1 || get Etched/oasis-500m oasis500m.safetensors oasis500m.pt
ls oasis500m.* >/dev/null 2>&1 || get camenduru/oasis-500m oasis500m.safetensors oasis500m.pt
ls vit-l-20.* >/dev/null 2>&1 || get Etched/oasis-500m vit-l-20.safetensors vit-l-20.pt
ls vit-l-20.* >/dev/null 2>&1 || get camenduru/oasis-500m vit-l-20.safetensors vit-l-20.pt
ls -la oasis500m.* vit-l-20.* sample_data/ 2>&1 | head -20
cd ../..
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2'); print('resnet50 weights ok')"
python -c "import einops, timm, av, safetensors, sklearn, scipy; print('deps ok')"
echo "PREP DONE. Optional GPU check (on a compute node): python -m vid.oasis_cc --selftest"

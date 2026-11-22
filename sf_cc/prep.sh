#!/bin/bash
# ONE-TIME SETUP on the TamIA LOGIN NODE (needs internet + HF_TOKEN):  export HF_TOKEN=hf_...; bash sf_cc/prep.sh
# Creates venv ~/venvs/sf, puts Self-Forcing + Wan2.1-T2V-1.3B weights (~20 GB) on $SCRATCH, caches metric models.
set -uo pipefail
cd "$(dirname "$0")/.."
module load gcc arrow python/3.11 cuda
VENV="${SF_VENV:-$HOME/venvs/sf}"
export HF_HOME="${SF_HF_HOME:-$SCRATCH/hf_sf}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=0
[ -d "$VENV" ] || virtualenv --no-download "$VENV" || python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null 2>&1
inst() { pip install --no-index "$1" >/dev/null 2>&1 || pip install "$1" >/dev/null 2>&1 || echo "!! could not install $1"; }
inst torch; inst torchvision
for p in "diffusers==0.31.0" "transformers>=4.49" accelerate ftfy easydict omegaconf einops imageio imageio-ffmpeg av \
         sentencepiece open_clip_torch matplotlib "huggingface_hub[cli]" scipy opencv-python-headless; do inst "$p"; done
pip install --no-index --force-reinstall --no-deps torch torchvision >/dev/null 2>&1 || true   # keep a matched pair
pip install --no-index flash_attn >/dev/null 2>&1 && echo "flash-attn installed" || echo "flash-attn not available: PyTorch SDPA fallback"
mkdir -p "$SCRATCH/sf" ext
[ -d "$SCRATCH/sf/Self-Forcing" ] || git clone https://github.com/guandeh17/Self-Forcing.git "$SCRATCH/sf/Self-Forcing"
ln -sfn "$SCRATCH/sf/Self-Forcing" ext/Self-Forcing
cd ext/Self-Forcing
python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
tok = os.environ.get('HF_TOKEN')
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='wan_models/Wan2.1-T2V-1.3B', token=tok)
hf_hub_download('gdhe17/Self-Forcing', 'checkpoints/self_forcing_dmd.pt', local_dir='.', token=tok)
print('weights ok')
PY
ls -la wan_models/Wan2.1-T2V-1.3B checkpoints | head -20
cd ../..
python -c "from sf_cc.metrics import load_models; load_models('cpu'); print('metric models cached')"
python -c "import torch, torchvision, torchvision.ops; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" \
  || echo "!! torch/torchvision mismatch: pip uninstall -y torch torchvision && pip install --no-index torch torchvision"
ls ext/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors ext/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  >/dev/null && echo "Self-Forcing weights present (GPU import check happens in the job's smoke test)"
echo "PREP DONE"

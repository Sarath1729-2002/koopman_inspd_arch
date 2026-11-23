#!/bin/bash
# ONE-TIME setup on the LOGIN node (compute nodes are offline): venv, Meta's jepa-wms code, datasets, weights.
# Run inside tmux or with nohup (downloads take a while):   nohup bash jwm/prep.sh > logs/jwm_prep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export JEPAWM_DSET=$SCRATCH/jepa_dset JEPAWM_LOGS=$SCRATCH/jepa_logs JEPAWM_CKPT=$SCRATCH/jepa_ckpt JEPAWM_OSSCKPT=$SCRATCH/jepa_ckpt
export TORCH_HOME=$SCRATCH/torch_jwm HF_HOME=$SCRATCH/hf_jwm
mkdir -p $JEPAWM_DSET $JEPAWM_LOGS $JEPAWM_CKPT $TORCH_HOME
module load gcc arrow python/3.10 cuda 2>/dev/null || module load gcc arrow python/3.11 cuda
python --version
V=${JWM_VENV:-$SCRATCH/venvs/jwm}; export PIP_CACHE_DIR=$SCRATCH/pip_cache; mkdir -p $SCRATCH/venvs
[ -d $V ] || virtualenv --no-download $V
source $V/bin/activate
pip install -q --upgrade pip
pip install -q --no-index torch torchvision 2>/dev/null || pip install -q torch torchvision
for p in tensordict timm einops decord scipy tqdm h5py matplotlib pandas imageio imageio-ffmpeg pyyaml omegaconf \
         huggingface_hub datasets clusterscope opencv-python-headless termcolor "ruamel.yaml" lpips; do
  pip install -q "$p" 2>&1 | tail -1; python -c "import importlib.util,sys; m=sys.argv[1].split(\"[\")[0].replace(\"-\",\"_\").split(\"=\")[0]; m={\"pyyaml\":\"yaml\",\"opencv_python_headless\":\"cv2\",\"huggingface_hub\":\"huggingface_hub\",\"ruamel.yaml\":\"ruamel\",\"imageio_ffmpeg\":\"imageio_ffmpeg\"}.get(m,m); sys.exit(importlib.util.find_spec(m) is None)" "$p" || echo "!! pip $p failed"
done
[ -d $SCRATCH/jepa-wms/.git ] || git clone -q https://github.com/facebookresearch/jepa-wms $SCRATCH/jepa-wms
mkdir -p ext; ln -sfn $SCRATCH/jepa-wms ext/jepa-wms
pip install -q --no-deps -e ext/jepa-wms 2>&1 | tail -1
python -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__)"
(cd ext/jepa-wms && HF_HUB_OFFLINE=0 python src/scripts/download_data.py --dataset pusht wall pointmaze) 2>&1 | tail -8
ls $JEPAWM_DSET
# cache weights (checkpoints, decoder heads, DINOv2 via torch.hub) by loading each model once on CPU
for m in dino_wm_pusht jepa_wm_pusht dino_wm_wall jepa_wm_wall dino_wm_pointmaze jepa_wm_pointmaze; do
  CUDA_VISIBLE_DEVICES= python -m jwm.drift --model $m --load_only 2>&1 | grep -E "LOADED|Error|error|No module" | tail -3
done
echo "PREP DONE"

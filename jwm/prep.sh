#!/bin/bash
# ONE-TIME setup on the LOGIN node (compute nodes are offline): venv, Meta's jepa-wms code, datasets, weights.
# Uses plain PyPI wheels (not the cluster wheelhouse, whose opencv/pyarrow are placeholders and whose torch is too old
# for jepa-wms), pinned to the versions jepa-wms asks for. Needs HF access to the gated dataset facebook/jepa-wms
# and HF_TOKEN exported.        nohup bash jwm/prep.sh > logs/jwm_prep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export JEPAWM_DSET=$SCRATCH/jepa_dset JEPAWM_LOGS=$SCRATCH/jepa_logs JEPAWM_CKPT=$SCRATCH/jepa_ckpt JEPAWM_OSSCKPT=$SCRATCH/jepa_ckpt
export TORCH_HOME=$SCRATCH/torch_jwm HF_HOME=$SCRATCH/hf_jwm HF_HUB_OFFLINE=0
export PIP_CONFIG_FILE=/dev/null PIP_CACHE_DIR=$SCRATCH/pip_cache   # plain PyPI, cache off home
mkdir -p $JEPAWM_DSET $JEPAWM_LOGS $JEPAWM_CKPT $TORCH_HOME $SCRATCH/venvs
module purge 2>/dev/null; module load python/3.10
python --version
V=${JWM_VENV:-$SCRATCH/venvs/jwm2}
[ -d $V ] || python -m venv $V
source $V/bin/activate
python --version
pip install -q --upgrade pip
pip install -q "torch==2.7.1" "torchvision==0.22.1" "tensordict>=0.9.1" 2>&1 | tail -2
pip install -q timm einops decord scipy tqdm h5py matplotlib pandas imageio imageio-ffmpeg pyyaml omegaconf \
    huggingface_hub datasets clusterscope opencv-python-headless termcolor ruamel.yaml lpips 2>&1 | tail -3
python -c "import torch, torchvision, tensordict, cv2, decord, datasets; print('torch', torch.__version__, 'tv', torchvision.__version__, 'tensordict', tensordict.__version__, 'cv2', cv2.__version__, 'cuda build', torch.version.cuda)" \
  || echo "!! core imports failed"
[ -d $SCRATCH/jepa-wms/.git ] || git clone -q https://github.com/facebookresearch/jepa-wms $SCRATCH/jepa-wms
mkdir -p ext; ln -sfn $SCRATCH/jepa-wms ext/jepa-wms
[ -n "${HF_TOKEN:-}" ] || echo "!! HF_TOKEN not set: the gated dataset download will fail"
(cd ext/jepa-wms && python src/scripts/download_data.py --dataset pusht wall pointmaze) 2>&1 | grep -v "^$" | tail -8
ls $JEPAWM_DSET
# cache weights (checkpoints, decoder heads, DINOv2 via torch.hub) by loading each model once on CPU
for m in dino_wm_pusht jepa_wm_pusht dino_wm_wall jepa_wm_wall dino_wm_pointmaze jepa_wm_pointmaze; do
  CUDA_VISIBLE_DEVICES= python -m jwm.drift --model $m --load_only 2>&1 | grep -E "LOADED|Error|error|No module" | tail -3
done
echo "PREP DONE"

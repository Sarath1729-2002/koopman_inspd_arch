#!/bin/bash
# ONE-TIME setup on the LOGIN node (compute nodes are offline). Installs only the extra packages jepa-wms needs into an
# EXISTING venv (default ~/venvs/sf, Python 3.11), with torch/torchvision/numpy pinned to what that venv already has so
# nothing it uses changes. opencv and pyarrow come from cluster modules (the wheelhouse versions are placeholders).
# jepa-wms itself is not pip-installed: it is put on sys.path. Needs HF_TOKEN + access to the gated facebook/jepa-wms.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export JEPAWM_DSET=$SCRATCH/jepa_dset JEPAWM_LOGS=$SCRATCH/jepa_logs JEPAWM_CKPT=$SCRATCH/jepa_ckpt JEPAWM_OSSCKPT=$SCRATCH/jepa_ckpt
export TORCH_HOME=$SCRATCH/torch_jwm HF_HOME=$SCRATCH/hf_jwm HF_HUB_OFFLINE=0 PIP_CACHE_DIR=$SCRATCH/pip_cache
mkdir -p $JEPAWM_DSET $JEPAWM_LOGS $JEPAWM_CKPT $TORCH_HOME
module load gcc arrow python/3.11 cuda opencv
V=${JWM_VENV:-$HOME/venvs/sf}
source $V/bin/activate || { echo "!! no venv at $V"; exit 1; }
python --version
python -c "import torch, torchvision, numpy; print('existing: torch', torch.__version__, 'tv', torchvision.__version__, 'numpy', numpy.__version__)"
C=$SCRATCH/jwm_constraints.txt
python -c "import torch, torchvision, numpy; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\nnumpy=={numpy.__version__}')" > $C
for p in tensordict timm einops decord scipy tqdm h5py matplotlib pandas imageio imageio-ffmpeg pyyaml omegaconf \
         huggingface_hub datasets clusterscope termcolor ruamel.yaml lpips; do
  pip install -q -c $C "$p" 2>&1 | grep -v "^$" | tail -1
done
python -c "import torch, torchvision, tensordict, cv2, decord, datasets, timm, huggingface_hub; print('OK imports: torch', torch.__version__, 'tensordict', tensordict.__version__, 'cv2', cv2.__version__)" \
  || echo "!! core imports failed"
[ -d $SCRATCH/jepa-wms/.git ] || git clone -q https://github.com/facebookresearch/jepa-wms $SCRATCH/jepa-wms
mkdir -p ext; ln -sfn $SCRATCH/jepa-wms ext/jepa-wms
[ -n "${HF_TOKEN:-}" ] || echo "!! HF_TOKEN not set: the gated dataset download will fail"
(cd ext/jepa-wms && python src/scripts/download_data.py --dataset pusht wall pointmaze) 2>&1 | grep -v "^$" | tail -8
ls $JEPAWM_DSET
for m in dino_wm_pusht jepa_wm_pusht dino_wm_wall jepa_wm_wall dino_wm_pointmaze jepa_wm_pointmaze; do
  CUDA_VISIBLE_DEVICES= python -m jwm.drift --model $m --load_only 2>&1 | grep -E "LOADED|Error|error|No module" | tail -3
done
echo "PREP DONE"

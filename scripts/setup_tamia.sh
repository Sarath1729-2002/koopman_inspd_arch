#!/bin/bash
# One-time setup on a TamIA LOGIN node (needs internet). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

module load gcc arrow python/3.11 cuda
if [ ! -d ~/venvs/ktrm ]; then virtualenv --no-download ~/venvs/ktrm; fi
source ~/venvs/ktrm/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch
pip install --prefer-binary numpy scipy scikit-learn matplotlib omegaconf hydra-core einops argdantic pydantic coolname huggingface_hub tqdm
# training-only extras (skip if you only analyse a downloaded checkpoint)
pip install --prefer-binary wandb numba || true
pip install --no-cache-dir --no-build-isolation adam-atan2 || echo "[warn] adam-atan2 failed to build: only needed to TRAIN TRM"

mkdir -p third_party
if [ ! -d third_party/TinyRecursiveModels ]; then
  git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git third_party/TinyRecursiveModels
fi

# ARC-AGI-1 data, exactly the README command (seed 42, 1000 augmentations) -> matches official checkpoints
if [ ! -f data/arc1concept-aug-1000/test/dataset.json ]; then
  ( cd third_party/TinyRecursiveModels && python -m dataset.build_arc_dataset \
      --input-file-prefix kaggle/combined/arc-agi \
      --output-dir ../../data/arc1concept-aug-1000 \
      --subsets training evaluation concept \
      --test-set-name evaluation )
fi
du -sh data/arc1concept-aug-1000
python -c "import torch, omegaconf, sklearn, matplotlib; print('env ok', torch.__version__)"

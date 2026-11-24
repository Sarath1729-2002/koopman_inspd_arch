#!/bin/bash
# Login node: download the 6 models' weights (checkpoint, decoder head, DINOv2) into the caches the job reads offline.
# Does NOT load the datasets (too heavy for the login node; the job's smoke test checks data loading on the GPU node).
cd "$(dirname "$0")/.."
module load gcc arrow python/3.11 cuda opencv; source "${JWM_VENV:-$HOME/venvs/sf}/bin/activate"
export JEPAWM_DSET=$SCRATCH/jepa_dset JEPAWM_LOGS=$SCRATCH/jepa_logs JEPAWM_CKPT=$SCRATCH/jepa_ckpt JEPAWM_OSSCKPT=$SCRATCH/jepa_ckpt
export TORCH_HOME=$SCRATCH/torch_jwm HF_HOME=$SCRATCH/hf_jwm HF_HUB_OFFLINE=0 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
for m in dino_wm_pusht jepa_wm_pusht dino_wm_wall jepa_wm_wall dino_wm_pointmaze jepa_wm_pointmaze; do
  CUDA_VISIBLE_DEVICES= python -m jwm.drift --model $m --cache_only 2>&1 | grep -E "CACHED|Error|error|No module" | tail -3
done
echo "CACHE DONE"

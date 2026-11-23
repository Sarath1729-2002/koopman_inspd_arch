#!/bin/bash
# Resumable CausVid download (login node; compute nodes have no internet), then submits the CausVid run.
# Run inside tmux so a dropped SSH session does not kill it:   tmux new -s dl 'bash sf_cc/get_causvid.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
module load gcc arrow python/3.11 cuda
source "${SF_VENV:-$HOME/venvs/sf}/bin/activate"
export HF_HOME=$SCRATCH/hf_sf HF_HUB_OFFLINE=0
[ -d ext/CausVid/.git ] || git clone -q https://github.com/tianweiy/CausVid ext/CausVid
DST=ext/Self-Forcing/checkpoints/causvid/model.pt
mkdir -p "$(dirname "$DST")"
if [ ! -s "$DST" ]; then
  for try in 1 2 3 4 5; do   # hf_hub_download resumes the partial file left by an interrupted attempt
    python -c "from huggingface_hub import hf_hub_download as d; print(d('tianweiy/CausVid','autoregressive_checkpoint/model.pt',local_dir='$SCRATCH/cv_dl'))" && break
    echo "attempt $try failed, retrying in 30 s"; sleep 30
  done
  mv "$SCRATCH/cv_dl/autoregressive_checkpoint/model.pt" "$DST" || { echo "DOWNLOAD FAILED"; exit 1; }
fi
SZ=$(stat -c %s "$DST"); echo "model.pt: $SZ bytes"
python -c "import torch; sd=torch.load('$DST', map_location='cpu', mmap=True); print('keys:', list(sd)[:5])" || { echo "model.pt unreadable (truncated?) -- delete it and rerun"; exit 1; }
grep -n "denoising_step_list\|num_frame_per_block\|warp_denoising_step\|context_noise" ext/CausVid/configs/wan_causal_dmd.yaml
deactivate
ARMS="causvid_reenc1 causvid_noise1 causvid_sink3_reenc1 causvid_base causvid_sink3" PROMPTS=0-15 GEN_MIN=230 \
  sbatch --time=4:30:00 slurm/sf.sh
squeue -u $USER --start

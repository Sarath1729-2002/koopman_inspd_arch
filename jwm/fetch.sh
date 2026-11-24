#!/bin/bash
# Login node: fetch the 6 world-model checkpoints from Hugging Face (facebook/jepa-wms) and the 2 image decoders with
# 16 parallel connections each (single-connection downloads are throttled here). Resumable; skips finished files.
cd "$(dirname "$0")/.."
CK=$SCRATCH/jepa_ckpt; DC=$SCRATCH/torch_jwm/hub/checkpoints; mkdir -p $CK $DC
for m in dino_wm_pusht jepa_wm_pusht dino_wm_wall jepa_wm_wall dino_wm_pointmaze jepa_wm_pointmaze; do
  [ -s $CK/$m.pth.tar ] && { echo "have $m"; continue; }   # pget writes the file only once all pieces are in
  bash sf_cc/pget.sh https://huggingface.co/facebook/jepa-wms/resolve/main/$m.pth.tar $CK/$m.pth.tar - 16 | tail -2
done
for f in vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar; do
  [ -s $DC/$f ] || bash sf_cc/pget.sh https://dl.fbaipublicfiles.com/jepa-wms/$f $DC/$f - 16 | tail -2
done
ls -la $CK $DC
echo "FETCH DONE"

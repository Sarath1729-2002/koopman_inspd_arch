#!/bin/bash
# Latent-CoT re-encoding kill test. Stage 0 (CoT SFT) once on GPU 0, then 4 arms in parallel from it:
#   coconut | soft2 (soft vocab re-encode every 2 thoughts = every step) | hard2 (hard, straight-through) | soft1 (every thought)
#   sbatch slurm/latent_snap.sh        (needs data_lcot/ from: python -m lcot.prep on the login node)
#SBATCH --job-name=latent_snap
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 HF_HOME="${HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="${OUT:-results_lcot}"; CK="${CK:-ckpt_lcot}"
mkdir -p logs/lcot "$OUT" "$CK"
[ -f data_lcot/train.jsonl ] || { echo "data_lcot/ missing: run python -m lcot.prep on the login node"; exit 1; }
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L
if [ ! -f "$CK/stage0.pt" ]; then
  CUDA_VISIBLE_DEVICES=0 python -m lcot.train --arm cot --do_stage0 --stage0_ckpt "$CK/stage0.pt" --out "$OUT" ${EXTRA:-} \
    > logs/lcot/stage0.log 2>&1 || { echo "stage 0 FAILED (logs/lcot/stage0.log)"; tail -20 logs/lcot/stage0.log; exit 1; }
fi
grep "CoT (stage 0)" logs/lcot/stage0.log
echo "===== $(date '+%F %T') stage 0 done"
g=0
for spec in "coconut none 2" "soft2 soft 2" "hard2 hard 2" "soft1 soft 1"; do
  set -- $spec
  ( CUDA_VISIBLE_DEVICES=$g python -m lcot.train --arm $1 --snap $2 --k $3 --stage0_ckpt "$CK/stage0.pt" --out "$OUT" ${EXTRA:-} \
      > logs/lcot/$1.log 2>&1 || echo "FAILED $1 (logs/lcot/$1.log)" ) &
  g=$((g+1))
done
wait
echo "===== $(date '+%F %T') arms done"; grep -h "acc" logs/lcot/*.log | grep -v "step"
python -m lcot.report --root "$OUT"
echo "===== DONE -> $OUT/REPORT.md"

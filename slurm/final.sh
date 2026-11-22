#!/bin/bash
# FINAL EVIDENCE RUN (~6-8 h): course correction in Open-Oasis 500M, powered and controlled.
#   3 videos x 8 prompts, 768 frames (24x the 32-frame training window), ResNet-50 + ViT-B/16 features.
#   Core arms x 2 sampling seeds: base, reenc4_lf (primary), noise4 (equal-size random nudge control), reenc4
#   Single seed: reenc1_lf, reenc8_lf, reenc16_lf (dose-response), evt2_lf, reenc4_mp, stab50 (model's own knob)
#   -> results_final/FINAL_VIDEO.md (+ figures) and results_wm/FINAL_MUJOCO.md. Resumable: finished runs are skipped.
#SBATCH --job-name=final
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
OUT=results_final; FR=768; OFFS=100,200,300,400,500,600,700,800
mkdir -p logs/final $OUT
echo "===== $(date '+%F %T') node $(hostname)"
python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest 2>&1 | grep -v Warning | tail -3 | cut -c1-300 \
  || { echo "SELFTEST FAILED (ViT weights downloaded? see README in slurm/final.sh)"; exit 1; }
RUNS=()
for s in "" _s1; do for v in 0 1 2; do for arm in base reenc4_lf noise4 reenc4; do RUNS+=("$arm$s:$v"); done; done; done
for v in 0 1 2; do for arm in reenc1_lf reenc8_lf reenc16_lf evt2_lf reenc4_mp stab50; do RUNS+=("$arm:$v"); done; done
echo "${#RUNS[@]} runs"
for g in 0 1 2 3; do
  ( for ((k = g; k < ${#RUNS[@]}; k += 4)); do
      arm=${RUNS[$k]%:*}; v=${RUNS[$k]#*:}
      CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames $FR \
          --offsets $OFFS --out $OUT > logs/final/${arm}_v$v.log 2>&1 || { echo "[final] $arm v$v FAILED"; tail -4 logs/final/${arm}_v$v.log; }
    done ) &
done
wait
grep -h "^\[" logs/final/*.log | cut -c1-260
python -m vid.report_final $OUT
python -m wm.report_final_wm results_wm
echo "===== $(date '+%F %T') DONE -> $OUT/FINAL_VIDEO.md  results_wm/FINAL_MUJOCO.md  figures in both"

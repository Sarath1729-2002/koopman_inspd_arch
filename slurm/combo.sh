#!/bin/bash
# COMBINATION RUN (~3 h): attention sink x decode->re-encode in Open-Oasis 500M. Same 3 videos x 8 prompts x 768 frames,
# seeds and offsets as slurm/final.sh, written into results_final so every arm is paired with the existing base/reenc4.
#   sink4 (first 4 frames, incl. the real prompt frame, stay in the 32-frame context), sink4_reenc4 (primary),
#   sink4_noise4 (equal-size random nudge control), then sink1 / sink1_reenc4 (real prompt frame only).
# Criteria C1-C3 are written in vid/report_final.py before the run. Resumable.
#SBATCH --job-name=combo
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=5:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
OUT=results_final; FR=768; OFFS=100,200,300,400,500,600,700,800
mkdir -p logs/combo $OUT
echo "===== $(date '+%F %T') node $(hostname)"
python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest --arm sink4_reenc4 2>&1 | grep -v Warning | tail -3 | cut -c1-300 || { echo "SELFTEST FAILED"; exit 1; }
RUNS=()
for arm in sink4 sink4_reenc4 sink4_noise4 sink1 sink1_reenc4; do for v in 0 1 2; do RUNS+=("$arm:$v"); done; done
echo "${#RUNS[@]} runs"
for g in 0 1 2 3; do
  ( for ((k = g; k < ${#RUNS[@]}; k += 4)); do
      arm=${RUNS[$k]%:*}; v=${RUNS[$k]#*:}
      CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames $FR \
          --offsets $OFFS --out $OUT > logs/combo/${arm}_v$v.log 2>&1 || { echo "[combo] $arm v$v FAILED"; tail -4 logs/combo/${arm}_v$v.log; }
      echo "[combo] $(date '+%T') $arm v$v done"
    done ) &
done
wait
python -m vid.report_final $OUT
echo "===== $(date '+%F %T') DONE -> $OUT/FINAL_VIDEO.md (section 'Combination')"

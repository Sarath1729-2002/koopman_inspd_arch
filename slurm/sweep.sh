#!/bin/bash
# PERIOD SWEEP on VALIDATION prompts (start offsets different from the test offsets in final.sh / combo.sh):
# base, sink4, sink4 + re-encode every P in {1,2,4,8,16}, and drift-triggered sink4_evt2 (re-encode a frame when its
# shift > 2x the real-frame floor). Choose P here, then confirm the chosen P on the test offsets. ~2.5 h. Resumable.
#SBATCH --job-name=sweep
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
OUT=${OUT:-results_val}; FR=768; OFFS=${OFFS:-50,150,250,350,450,550,650,750}
mkdir -p logs/sweep $OUT
echo "===== $(date '+%F %T') node $(hostname)"
python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest --arm sink4_reenc4 2>&1 | grep -v Warning | tail -3 | cut -c1-300 || { echo "SELFTEST FAILED"; exit 1; }
RUNS=()
for arm in ${SWEEP_ARMS:-base sink4 sink4_reenc1 sink4_reenc2 sink4_reenc4 sink4_reenc8 sink4_reenc16 sink4_evt2}; do for v in ${VIDS:-0 1}; do RUNS+=("$arm:$v"); done; done
echo "${#RUNS[@]} runs"
for g in 0 1 2 3; do
  ( for ((k = g; k < ${#RUNS[@]}; k += 4)); do
      arm=${RUNS[$k]%:*}; v=${RUNS[$k]#*:}
      CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames $FR \
          --offsets $OFFS --out $OUT > logs/sweep/${arm}_v$v.log 2>&1 || { echo "[sweep] $arm v$v FAILED"; tail -4 logs/sweep/${arm}_v$v.log; }
      echo "[sweep] $(date '+%T') $arm v$v done"
    done ) &
done
wait
python -m vid.report_final $OUT
echo "===== $(date '+%F %T') DONE -> $OUT/FINAL_VIDEO.md (sections 'Re-encoding period sweep' and 'FD over time')"

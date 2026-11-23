#!/bin/bash
# OASIS, everything left, in priority order (resumable; resubmit to continue):
#   1. Mechanism controls on the TEST offsets (results_final): post (re-encode output frames after the rollout, no
#      feedback), reenc4_ctx and sink4_reenc4_ctx (correction feeds only the model's context; the viewer sees raw frames),
#      sink4_post. Criteria M1-M2 in vid/report_final.py.
#   2. Second sampling seed for the combination (sink4_s1, sink4_reenc4_s1, sink4_noise4_s1).
#   3. Period sweep + drift trigger on VALIDATION offsets (results_val), to choose P without touching the test set.
#SBATCH --job-name=oasis_all
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
FR=768; TEST=100,200,300,400,500,600,700,800; VAL=50,150,250,350,450,550,650,750
mkdir -p logs/oasis_all results_final results_val
echo "===== $(date '+%F %T') node $(hostname)"
for a in sink4_reenc4_ctx post; do
  python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest --arm $a 2>&1 | grep -v Warning | tail -2 | cut -c1-300 \
    || { echo "SELFTEST $a FAILED"; exit 1; }
done
RUNS=()
for arm in sink4_reenc4_ctx post reenc4_ctx sink4_post; do for v in 0 1 2; do RUNS+=("$arm:$v:results_final:$TEST"); done; done
for arm in sink4_reenc4_s1 sink4_s1 sink4_noise4_s1; do for v in 0 1 2; do RUNS+=("$arm:$v:results_final:$TEST"); done; done
for arm in base sink4 sink4_reenc4 sink4_reenc2 sink4_reenc8 sink4_evt2 sink4_reenc1 sink4_reenc16; do
  for v in 0 1; do RUNS+=("$arm:$v:results_val:$VAL"); done; done
echo "${#RUNS[@]} runs"
for g in 0 1 2 3; do
  ( for ((k = g; k < ${#RUNS[@]}; k += 4)); do
      IFS=: read -r arm v out offs <<< "${RUNS[$k]}"
      CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames $FR \
          --offsets $offs --out $out > logs/oasis_all/${out}_${arm}_v$v.log 2>&1 \
          || { echo "[oasis] $arm v$v $out FAILED"; tail -4 logs/oasis_all/${out}_${arm}_v$v.log; }
      echo "[oasis] $(date '+%T') $arm v$v $out done"
    done ) &
done
wait
python -m vid.report_final results_final > /dev/null
python -m vid.report_final results_val > /dev/null
echo "===== $(date '+%F %T') DONE -> results_final/FINAL_VIDEO.md  results_val/FINAL_VIDEO.md"

#!/bin/bash
# Every Nov 24 check in one job (~6-7 h). Criteria are in each script's docstring; summary -> CHECKS.md
#   GPU0  Coconut: error direction + steering (rdiag.cdir), then retrain Coconut 3 epochs/stage (C3) and rerun
#         C1/C2 + direction/steering on it
#   GPU1  Ouro-1.4B: damage direction, steering, drift as a stopping signal (rdiag.odir)
#   GPU2/3 Oasis base + sink4 rollouts with saved latents (validation offsets) -> Koopman fit (vid.koopman)
#   CPU   growth law + period-from-regrowth on existing Oasis / CausVid / Self-Forcing files (vid.growth)
# Resumable: finished parts are skipped. ONLY=coconut|ouro|oasis|cpu runs one part.
#SBATCH --job-name=checks
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=9:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${RD_HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
O="${ONLY:-all}"; mkdir -p logs/checks results_rdiag results_koop
echo "===== $(date '+%F %T') node $(hostname)"
for f in data_lcot/test.jsonl ckpt_lcot/stage0.pt ckpt_lcot/coconut_final.pt ext/open-oasis; do [ -e $f ] || echo "MISSING $f"; done
run() { local g=$1 log=$2; shift 2; CUDA_VISIBLE_DEVICES=$g "$@" > logs/checks/$log.log 2>&1 || { echo "[checks] $log FAILED"; tail -12 logs/checks/$log.log; return 1; }; }
if [[ $O =~ ^(all|coconut)$ ]]; then (
  [ -s results_rdiag/cdir.json ] || run 0 cdir python -m rdiag.cdir
  [ -s results_rdiag/coconut_e3.json ] || run 0 coconut_e3 python -m rdiag.coconut --eps 3
  [ -s results_rdiag/cdir_e3.json ] || run 0 cdir_e3 python -m rdiag.cdir --eps 3 ) & fi
if [[ $O =~ ^(all|ouro)$ ]]; then ( [ -s results_rdiag/odir.json ] || run 1 odir python -m rdiag.odir ) & fi
if [[ $O =~ ^(all|oasis)$ ]]; then
  VAL=50,150,250,350,450,550,650,750
  for v in 0 1; do (
    for arm in base sink4; do
      [ -s results_koop/${arm}_v${v}_lat.npy ] || run $((2 + v)) oasis_${arm}_v$v env VID_SAVE_LAT=1 \
        python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames 768 --offsets $VAL --out results_koop
    done ) &
  done
fi
wait
if [[ $O =~ ^(all|cpu|oasis)$ ]]; then
  python -m vid.growth > logs/checks/growth.log 2>&1 || { echo "[checks] growth FAILED"; tail -12 logs/checks/growth.log; }
  for arm in base sink4; do python -m vid.koopman --arm $arm > logs/checks/koopman_$arm.log 2>&1 || { echo "[checks] koopman $arm FAILED"; tail -8 logs/checks/koopman_$arm.log; }; done
fi
grep -h "^\[\|PASS\|FAIL" logs/checks/*.log 2>/dev/null | grep -v "^\[coconut\] stage"
python -m vid.checks_report > /dev/null && echo "===== $(date '+%F %T') DONE -> CHECKS.md"

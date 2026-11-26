#!/bin/bash
# Sparse-Koopman reruns, both tracks in one job (~4-5 h). Criteria in each script's docstring.
#   GPU0 Ouro-1.4B  rdiag.oskae        SO1-SO3  (regime-aware rerun of KP1/KP2 and G1o)
#   GPU1 Ouro-2.6B  rdiag.oskae --tag 26
#   GPU2 Oasis: fit the sparse Koopman AE on real latents + SK1 (vid.skae_oasis), then GPU2/3 run the trigger arms on the
#        TEST offsets: sink4_sk (support-novelty trigger), sink4_rnd (random, same rate), sink4_evt2 (VAE-drift trigger)
#        -> SKV1-SKV3 in results_final/SKAE.md
#SBATCH --job-name=skae
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset VIRTUAL_ENV PYTHONHOME; PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/venvs/' | paste -sd: -)
module load gcc arrow python/3.11 cuda
source "${VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME="${RD_HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p logs/skae results_rdiag results_final
echo "===== $(date '+%F %T') node $(hostname)"
TEST=100,200,300,400,500,600,700,800
( [ -s results_rdiag/oskae.json ] || CUDA_VISIBLE_DEVICES=0 python -m rdiag.oskae > logs/skae/oskae.log 2>&1 \
    || { echo "[skae] Ouro-1.4B FAILED"; tail -15 logs/skae/oskae.log; } ) &
( [ -s results_rdiag/oskae_26.json ] || CUDA_VISIBLE_DEVICES=1 python -m rdiag.oskae --model ByteDance/Ouro-2.6B --tag 26 --bs 8 \
    > logs/skae/oskae_26.log 2>&1 || { echo "[skae] Ouro-2.6B FAILED"; tail -15 logs/skae/oskae_26.log; } ) &
(
  if [ ! -s results_final/skae_oasis.pt ]; then
    CUDA_VISIBLE_DEVICES=2 python -m vid.skae_oasis ext/open-oasis > logs/skae/skae_oasis.log 2>&1 \
      || { echo "[skae] Oasis SKAE fit FAILED"; tail -15 logs/skae/skae_oasis.log; exit 1; }
  fi
  grep -h "SK1\|threshold" logs/skae/skae_oasis.log
  CUDA_VISIBLE_DEVICES=2 python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest --arm sink4_sk 2>&1 | grep -v Warning | tail -2 \
    || { echo "[skae] selftest sink4_sk FAILED"; exit 1; }
  RUNS=(); for arm in sink4_sk sink4_rnd sink4_evt2; do for v in 0 1 2; do RUNS+=("$arm:$v"); done; done
  for g in 2 3; do
    ( for ((k = g - 2; k < ${#RUNS[@]}; k += 2)); do
        IFS=: read -r arm v <<< "${RUNS[$k]}"
        [ -s results_final/${arm}_v$v.json ] && continue
        CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --video_idx $v --frames 768 \
            --offsets $TEST --out results_final > logs/skae/${arm}_v$v.log 2>&1 \
            || { echo "[skae] $arm v$v FAILED"; tail -4 logs/skae/${arm}_v$v.log; }
        echo "[skae] $(date '+%T') $arm v$v done"
      done ) &
  done
  wait
  python -m vid.report_skae > logs/skae/report.log 2>&1 && tail -5 results_final/SKAE.md || { echo "[skae] report FAILED"; tail -8 logs/skae/report.log; }
) &
wait
grep -h "^\[oskae\]\|SO[123]" logs/skae/oskae*.log
python -m vid.checks_report > /dev/null; echo "===== $(date '+%F %T') DONE -> CHECKS.md"

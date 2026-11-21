#!/bin/bash
# OVERNIGHT 2: two explorations in parallel on 4 GPUs.
#   A (MuJoCo, short-data regime): 54 small models trained only on 16-step segments -> 180-step deployment
#     vs noise injection, ensembles, density trigger                         -> results_wm/SHORT.md
#   B (Open-Oasis 500M streaming video world model): 8 inference arms x 512 frames (16x training window),
#     decode->re-encode of generated latents vs the model's own stabilization knob -> results_vid/VIDEO.md
# B needs `bash vid/prep.sh` run ONCE on the login node first; if B's selftest fails, A still runs. Resumable.
#SBATCH --job-name=overnight2
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda mujoco/3.3.0
source "${WM_VENV:-$HOME/venvs/ouro}/bin/activate"
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}" HF_HUB_OFFLINE=1
STEPS="${STEPS:-10000}"; MAXJ="${MAXJ:-24}"; mkdir -p logs/wm logs/vid results_wm results_vid
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done; }
echo "===== $(date '+%F %T') node $(hostname)"; nvidia-smi -L

# ---------------- B: video world model (background, 2 arms per GPU, sequential per GPU)
( if python -m vid.oasis_cc --oasis_dir ext/open-oasis --selftest > logs/vid/selftest.log 2>&1; then
    echo "[B] selftest OK"; tail -1 logs/vid/selftest.log
    ARMS_B=(base reenc1 stab1 reenc4 stab50 reenc16 stab100 reenc1_stab50)
    for g in 0 1 2 3; do
      ( for k in 0 1; do arm=${ARMS_B[$((g * 2 + k))]}
          CUDA_VISIBLE_DEVICES=$g python -m vid.oasis_cc --oasis_dir ext/open-oasis --arm $arm --frames 512 \
              --out results_vid > logs/vid/$arm.log 2>&1 || echo "[B] $arm FAILED (logs/vid/$arm.log)"
        done ) &
    done
    wait
    grep -h "^\[" logs/vid/*.log | grep -v selftest
    python -m vid.report_vid results_vid
  else
    echo "[B] SELFTEST FAILED -- see logs/vid/selftest.log (did you run bash vid/prep.sh?)"; tail -15 logs/vid/selftest.log
  fi ) > logs/vid/B_summary.log 2>&1 &
BPID=$!

# ---------------- A: short-data MuJoCo
i=0
for seed in 0 1 2; do for env in cheetah walker hopper; do
  for arm in kae_s10 gru_s10 koop_s10 kae_s10n gru_s10n koop_s10n; do
    throttle
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -m wm.train --env $env --arm $arm --seed $seed --steps $STEPS \
        > logs/wm/${env}_${arm}_s${seed}.log 2>&1 &
    i=$((i + 1))
  done
done; done
# wait for A's training jobs only (B runs on)
while [ "$(jobs -rp | grep -v "^$BPID\$" | wc -l)" -gt 0 ]; do sleep 30; done
echo "===== $(date '+%F %T') A training done: $(ls ckpt_wm/*_s10*_s?.pt 2>/dev/null | wc -l)/54 models"
grep -l "non-finite\|Error" logs/wm/*_s10*.log 2>/dev/null | head
g=0
for env in cheetah walker hopper; do
  CUDA_VISIBLE_DEVICES=$g python -m wm.short_eval --env $env > logs/wm/short_${env}.log 2>&1 &
  g=$((g + 1))
done
while [ "$(jobs -rp | grep -v "^$BPID\$" | wc -l)" -gt 0 ]; do sleep 30; done
grep -h "^\[" logs/wm/short_*.log
python -m wm.report_short results_wm
echo "===== $(date '+%F %T') A done -> results_wm/SHORT.md ; waiting for B"
wait $BPID
cat logs/vid/B_summary.log
echo "===== $(date '+%F %T') ALL DONE -> results_wm/SHORT.md  results_vid/VIDEO.md  results_vid/*.mp4"

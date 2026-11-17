#!/bin/bash
# Full proof of concept on Ouro-1.4B, one log:  check -> fit -> evaluate -> generate.
#   sbatch slurm/ouro_poc.sh
#SBATCH --job-name=koopman_ouro
#SBATCH --account=aip-oberman
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow python/3.11 cuda
source ~/venvs/ouro/bin/activate
export HF_HOME="${HF_HOME:-$SCRATCH/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${MODEL:-ByteDance/Ouro-1.4B}"
OUT="${OUT:-runs/ouro14}"
A="--model $MODEL --data_dir data --out $OUT --t_train 4 --t_max 12"
stamp(){ echo; echo "===== $(date '+%F %T')  $* ====="; }
stamp "node $(hostname)"; nvidia-smi -L

stamp "0/3 check hooks"
python -m kl.check $A --batch_size 8
python - "$OUT/inspect.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if r["exact_t_train_top1_agree"] < 0.99:
    sys.exit("readout mismatch: depth-d readout != real d-loop model. Stopping; send the log.")
print("hooks verified")
PY
stamp "1/3 fit dynamics inside trained depth"; python -m kl.fit $A --batch_size 16 --n_train 2000
stamp "2/3 evaluate depths 1..12 x policies";  python -m kl.evaluate $A --batch_size 8 --n_gsm 256 --n_arc 400
stamp "3/3 GSM8K generation";                  python -m kl.generate $A --batch_size 16 --n 250
stamp "DONE -> $OUT/REPORT.md"

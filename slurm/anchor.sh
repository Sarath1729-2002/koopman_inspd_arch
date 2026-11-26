#!/bin/bash
# Anchor arms for content drift (sf_cc/run_sf.py: hgNN, adpN, adtK), Self-Forcing + CausVid at 120 s (ID-Forcing's
# headline length), prompts 0-15, then the benchmark table with AN1-AN3 (sf_cc/bench.py). Submit from the repo root:
#   bash slurm/anchor.sh
cd "$(dirname "$0")/.." || exit 1
ARMS="sink3 sink3_hg25 sink3_hg50 sink3_adp2 sink3_adt2 sink3_reenc2_cache causvid_sink3 causvid_sink3_hg50 causvid_sink3_adp2"
j=$(OUT=results_sf_anchor ARMS="$ARMS" PROMPTS=0-15 LAT=480 LPG=1 SMOKE_ARM=sink3_hg50_adp2 \
    sbatch --parsable --job-name=anchor --time=14:00:00 slurm/sf.sh) && echo "anchor -> $j"
b=$(sbatch --parsable --job-name=anchorB --dependency=afterany:$j --time=1:30:00 --wrap \
    "cd $PWD; unset VIRTUAL_ENV PYTHONHOME; module load gcc arrow python/3.11; source \$HOME/venvs/sf/bin/activate; python sf_cc/bench.py" \
    --account=aip-oberman --gpus-per-node=h100:4 --cpus-per-task=32 --mem=0 --output=logs/%x-%j.out) && echo "bench  -> $b (runs after)"

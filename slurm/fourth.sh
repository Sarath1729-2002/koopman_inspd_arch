#!/bin/bash
# Deep Forcing rerun (prompt-path fix) + delayed anchor refresh for prompt switches (CC6). Submit: bash slurm/fourth.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
d=$(OUT=results_sf_anchor3 PROMPTS=32-47 LAT=480 sbatch --parsable slurm/deepforcing.sh) && echo "deepF    -> $d"
s=$(OUT=results_sf_switch PROMPTS=8-15 LAT=480 LPG=1 SMOKE_ARM=sink3_adp3_cache_sw_rf8 \
    ARMS="sink3_adp3_cache_sw_rf8 sink3_adp3_cache_sw_rf16 sink3_adp2_cache_sw_rf8" \
    sbatch --parsable --job-name=anchorS3 --time=8:00:00 slurm/sf.sh) && echo "anchorS3 -> $s"

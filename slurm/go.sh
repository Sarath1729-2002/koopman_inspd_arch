#!/bin/bash
# One command for the Nov 25 afternoon batch (run on the login node from the repo root, any venv state):
#   ouro2gen  GSM8k answer accuracy with steering (U2), Ouro-1.4B + 2.6B                       ~3 h
#   sfH       CausVid 4 min on fresh prompts 24-39 (V3), then the "what degrades" readout (M1)  ~12-16 h
# Results afterwards:  python -m vid.checks_report ; cat results_sf_long/MECH.md
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
j1=$(sbatch --parsable slurm/ouro2gen.sh) && echo "ouro2gen -> $j1"
j2=$(OUT=results_sf_long ARMS="causvid_sink3 causvid_sink3_reenc2_cache" PROMPTS=24-39 LAT=960 LPG=1 \
     sbatch --parsable --job-name=sfH --time=20:00:00 slurm/sf.sh) && echo "sfH      -> $j2"

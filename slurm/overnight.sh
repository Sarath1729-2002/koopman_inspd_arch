#!/bin/bash
# Overnight confirmation of anchor re-denoising (Nov 27). Three jobs, criteria in sf_cc/bench.py (FP1-3, L4, SEAM, SW1-2).
#   anchorF  fresh prompts 16-31, 120 s: Self-Forcing sink3 / adp1 / adp2 / adp3 / adt2 / adp2+re-encode,
#            CausVid sink3 / adp2 / adp2+re-encode                                       -> results_sf_anchor2
#   anchorL  4 min, prompts 0-7: Self-Forcing adp2, CausVid adp2 (references already there) -> results_sf_long
#   anchorS  prompt switch halfway, 120 s, prompts 0-7: sink3 / adp2 / adp2 + anchor refresh   -> results_sf_switch
# Submit from the repo root:  bash slurm/overnight.sh
# Afterwards, on the login node:
#   module load gcc arrow python/3.11 && source ~/venvs/sf/bin/activate && python sf_cc/bench.py | tail -60
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
a=$(OUT=results_sf_anchor2 PROMPTS=16-31 LAT=480 LPG=1 SMOKE_ARM=sink3_adp2_reenc2_cache \
    ARMS="sink3 sink3_adp1 sink3_adp2 sink3_adp3 sink3_adt2 sink3_adp2_reenc2_cache causvid_sink3 causvid_sink3_adp2 causvid_sink3_adp2_reenc2_cache" \
    sbatch --parsable --job-name=anchorF --time=16:00:00 slurm/sf.sh) && echo "anchorF -> $a"
b=$(OUT=results_sf_long PROMPTS=0-7 LAT=960 LPG=1 SMOKE_ARM=sink3_adp2 ARMS="sink3_adp2 causvid_sink3_adp2" \
    sbatch --parsable --job-name=anchorL --time=12:00:00 slurm/sf.sh) && echo "anchorL -> $b"
c=$(OUT=results_sf_switch PROMPTS=0-7 LAT=480 LPG=1 SMOKE_ARM=sink3_adp2_sw_rf ARMS="sink3_sw sink3_adp2_sw sink3_adp2_sw_rf" \
    sbatch --parsable --job-name=anchorS --time=8:00:00 slurm/sf.sh) && echo "anchorS -> $c"

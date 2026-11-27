#!/bin/bash
# Third confirmation (Nov 27): cache-only anchor re-denoise (corrects what the model conditions on, not what is shown),
# on a third fresh prompt set, at 4 min, with a prompt switch, and against Deep Forcing. Criteria CC1-CC5, CMP in sf_cc/bench.py.
# Login-node prep (once):
#   git clone https://github.com/cvlab-kaist/DeepForcing.git ext/DeepForcing
#   ln -s $PWD/ext/Self-Forcing/checkpoints ext/DeepForcing/checkpoints; ln -s $PWD/ext/Self-Forcing/wan_models ext/DeepForcing/wan_models
# Submit:  bash slurm/third.sh      Results (login node):  python sf_cc/bench.py | tail -70
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
a=$(OUT=results_sf_anchor3 PROMPTS=32-47 LAT=480 LPG=1 SMOKE_ARM=sink3_adp3_cache \
    ARMS="sink3 sink3_adp2_cache sink3_adp3_cache causvid_sink3 causvid_sink3_adp2_cache causvid_sink3_adp3_cache" \
    sbatch --parsable --job-name=anchorC --time=14:00:00 slurm/sf.sh) && echo "anchorC -> $a"
b=$(OUT=results_sf_long PROMPTS=8-15 LAT=960 LPG=1 SMOKE_ARM=sink3_adp3_cache ARMS="sink3_adp3_cache causvid_sink3_adp3_cache" \
    sbatch --parsable --job-name=anchorL2 --time=12:00:00 slurm/sf.sh) && echo "anchorL2 -> $b"
c=$(OUT=results_sf_switch PROMPTS=8-15 LAT=480 LPG=1 SMOKE_ARM=sink3_adp3_cache_sw_rf ARMS="sink3_sw sink3_adp3_cache_sw_rf" \
    sbatch --parsable --job-name=anchorS2 --time=8:00:00 slurm/sf.sh) && echo "anchorS2 -> $c"
d=$(OUT=results_sf_anchor3 PROMPTS=32-47 LAT=480 sbatch --parsable slurm/deepforcing.sh) && echo "deepF    -> $d"

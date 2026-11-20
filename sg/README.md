# seriality — does course-correcting Koopman close the seriality gap? (Ross's direction, state-space stage)

The Seriality Gap in Video Diffusion Models (2026): parallel generation fails on chains of collisions; AR works.
Hypothesis: serial compute is needed per EVENT, not per frame. Koopman rolls out free flight in closed form (parallel)
and re-encodes from its own decoded frames at events (trigger = latent leaves the encoder manifold).
Arms: parallel (1 pass), ar (48 passes), koop one-shot, koop + periodic re-encoding, koop + event re-encoding.
Verdict: sg/report.py.   sbatch slurm/seriality.sh  -> results_sg/REPORT.md

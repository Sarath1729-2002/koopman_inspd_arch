# sg2 — overnight proof of concept: Koopman course correction vs the seriality gap

Testbed: the hard-sphere dynamics of "The Seriality Gap in Video Diffusion Models" (2026), in state space.
Generative models (rectified-flow diffusion over the 48 future frames), 3 seeds each:
  block48 / block8 / block1   parallel (bidirectional) / block-causal / autoregressive diffusion (attention masks only)
  koopdiff                    parallel diffusion conditioned on the closed-form Koopman rollout Lambda^t phi(window),
                              re-encoded from its own generated frames: 1 pass, periodic, or event-triggered
                              (latent leaves the encoder manifold; thresholds calibrated to 2/4/8/16 passes)
  featdiff                    capacity-matched control: same conditioning interface, no linear-dynamics structure
Pre-registered claims C1-C4: see sg2/report.py.   sbatch slurm/poc_overnight.sh -> results_sg2/REPORT.md, poc.png

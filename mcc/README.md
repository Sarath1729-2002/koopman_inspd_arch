# mamba_cc — course-correcting pretrained Mamba past its training length (iteration 1 of 3)

Re-encoding from observations: staggered "shadow" states give a fresh, in-distribution encoding of the last <= 2k
tokens at every step; every 64 tokens past 2k, only the DRIFTED parts of the long state are replaced by the shadow.
Baselines: none, window (sliding-window forget), clamp. Verdict and iteration rule: mcc/report.py.

    cd ~/koopman_inspd_arch && tar xzf mamba_cc.tgz && mkdir -p logs
    module load gcc arrow python/3.11 && source ~/venvs/ouro/bin/activate && export HF_HOME=$SCRATCH/hf
    python -m mcc.prep                          # login node: Mamba 130m..2.8b (HF format)
    OMP_NUM_THREADS=4 bash tests/smoke_mcc.sh   # ~1 min
    sbatch slurm/mamba_cc.sh                    # -> results_mcc/REPORT.md

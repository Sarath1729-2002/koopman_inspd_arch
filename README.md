# lm_drift — does periodic re-encoding fix long-context drift in a real (byte-level) linear-RNN LM?

drift_kill showed re-encoding = error correction works when the state has a codebook. Here the state is continuous.
Byte LMs on wikitext-103 with Griffin RG-LRU and Mamba-style cores, trained at context 512, tested at 8192 (16x).
Arms: base · reset (forget every 512) · clamp (clip state to in-dist range) · ae (post-hoc denoising AE re-encoder,
training-free) · re64 (trained-in decode->normalise->re-encode every 64 tokens). Verdict: lmd/report.py.

    # LOGIN NODE (internet), once:
    cd ~/koopman_inspd_arch && tar xzf lm_drift.tgz && cd lm_drift && mkdir -p logs
    module load gcc arrow python/3.11 && source ~/venvs/ktrm/bin/activate
    export HF_HOME=$SCRATCH/hf
    python -c "import huggingface_hub, pyarrow" || pip install huggingface_hub
    python scripts/prep_data.py          # -> data/train.bin (~540 MB), data/eval.bin
    OMP_NUM_THREADS=4 bash tests/smoke.sh
    sbatch slurm/run.sh                  # ~30-45 min -> results/REPORT.md

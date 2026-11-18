# depth_koop — course-correcting Koopman across depth (last Koopman kill test)

Replace m of a pretrained LLM's layers with least-squares linear maps (no training). Fathi et al. predict linear
surrogate error compounds over consecutive steps and real nonlinear steps in between reset it. So at equal m:
interleaved (lin_inter) should beat one contiguous linear block (lin_contig, ReplaceMe-style).
Pre-registered verdict at the top of dkoop/report.py. Uses data/ from lm_drift + data/gsm8k_test.jsonl.

    cd ~/koopman_inspd_arch && tar xzf depth_koop.tgz        # unpacks dkoop/, slurm/depth_koop.sh, tests/
    ls $SCRATCH/hf/hub | grep -i qwen2.5-7b                   # model must be cached (offline nodes)
    source ~/venvs/ouro/bin/activate && OMP_NUM_THREADS=4 bash tests/smoke_depth.sh   # ~1 min on login node
    sbatch slurm/depth_koop.sh                                # ~30-45 min -> results_depth/REPORT.md

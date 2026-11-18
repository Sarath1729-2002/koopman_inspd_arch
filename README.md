# drift_kill — periodic re-encoding as error correction (not computation)

The S5 test showed boundary maps can't do composition (needs |G|^k cases). Cleanup only needs |G| cases.
Tasks the linear core solves exactly (parity, mod-5 counting); train <= 64, test to 4096; any long-length
failure is drift. Re-encoding must beat a unit-circle core AND norm-only renormalisation. Verdict: dk/report.py.

    cd ~/koopman_inspd_arch && tar xzf drift_kill.tgz && cd drift_kill && mkdir -p logs
    OMP_NUM_THREADS=4 bash tests/smoke.sh
    sbatch slurm/run.sh        # -> runs_drift/REPORT.md, runs_drift/drift.png

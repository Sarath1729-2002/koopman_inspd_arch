# boundary_kill — does periodic re-encoding let a linear RNN track S5?

One stress test, pre-registered verdict (top of bk/report.py).
S5 word problem, per-token accuracy, train length <= 64, test 64/128/256/512.

Arms: lstm (re-encode every token) · deltaprod (n_h=4 Householder, the bar) · lrnn (diagonal, eigs in (-1,1)) ·
bnd{2,4,8,16} (lrnn inside chunks of k, LN(MLP(h)) at every boundary).

SURVIVES only if a bnd arm with k>=8 gets >=0.90 at 4x length, within 0.05 of deltaprod, and >=0.10 above lrnn.
Valid only if lstm or deltaprod reaches >=0.95 at train length.

    tar xzf boundary_kill.tgz && cd boundary_kill
    bash tests/smoke.sh                 # ~1 min on login node, all arms + report
    sbatch slurm/run.sh                 # 21 runs, one H100 node, then report -> runs_s5/REPORT.md
    GROUP=s3 sbatch slurm/run.sh        # fallback only if S5 comes back INCONCLUSIVE -> runs_s3/

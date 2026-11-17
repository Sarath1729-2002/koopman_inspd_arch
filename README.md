# koopman_loopgen — Phase 1: does a Koopman-stable loop extrapolate better?

Testbed: the looped-transformer length-generalisation tasks of Fan et al. (ICLR 2025), with their
setup (GPT-2 blocks, NoPE, input injection, loss after exactly T(n) loops, curriculum, online data).
Harder (longer) instances need MORE loops, so stable extrapolation is directly worth accuracy.

| task | train lengths | eval lengths | loops T(n) | block |
|---|---|---|---|---|
| parity | 1–20 | 5–50 | n | 1 layer |
| copy (digits) | 1–19 | 5–40 | n | 2 layers |
| binary addition | 1–19 | 5–35 | n+1 | 3 layers |

Arms (same width 256, same block; `klg/model.py`):
- `plain` — h ← Block(h + e)  (Fan et al.)
- `gainctl` — LayerNorm on the loop output (normalisation control)
- `koopman` — h ← Λh + e + f(h + e), Λ = block-diagonal rotation-scaling, radius ≤ 1 (spectrally bounded backbone)
- `koopman_re` — `koopman` + every 4 loops decode to the output vocabulary and re-embed (course correction, trained in)

Measured per length: exact match with T(n) loops (oracle), with 2·T(n) loops (over-looping),
with loops chosen by confidence in 1..3T (no oracle), the full accuracy-vs-loops curve, and state-norm drift.

**Kill rule (pre-registered, `klg/report.py`)**: SURVIVES if the best Koopman arm beats the best control by
≥ 0.10 OOD exact match averaged over tasks, OR plain loses > 0.20 when over-looped while Koopman loses ≤ 0.05.
Otherwise CLOSED.

## Run (TamIA)
```bash
mkdir -p ~/koopman_loopgen && tar xzf koopman_loopgen.tar.gz --strip-components=1 -C ~/koopman_loopgen
cd ~/koopman_loopgen
module load gcc arrow python/3.11 && source ~/venvs/ktrm/bin/activate   # existing env: torch + matplotlib
bash tests/smoke.sh                                                      # ~1-2 min, ends with SMOKE OK
mkdir -p logs && sbatch slurm/phase1.sh                                  # 36 runs, 8 in parallel on 4 H100s
```
Progress: `grep -h "done:" logs/phase1/*.log`. Result: `runs/REPORT.md`, `runs/phase1.png`.
Resumable: rerunning `sbatch slurm/phase1.sh` skips finished runs.

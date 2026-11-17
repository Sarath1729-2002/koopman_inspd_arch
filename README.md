# koopman_loop — Koopman course correction for looped LLMs (Ouro-1.4B)

**Question.** Ouro runs its whole 24-layer stack 4 times and gets worse when looped more
(paper Tab. 10: MMLU 67.5 → 64.5, ARC-C 60.9 → 58.2 at 8 loops; AIME-24 65 → 39 for the Thinking model).
Can dynamics learned *inside* the trained depth keep the extra loops on track, and turn them into
gains, better than simple non-dynamical fixes?

Since you can always just stop at 4 loops, matching 4 loops is worth nothing. The bar is **beating STOP**.

## Stages (one job: `slurm/ouro_poc.sh`)
| stage | script | what |
|---|---|---|
| 0 | `kl.check` | hooks fire once per loop; depth-d readout from a 12-loop forward equals a real d-loop model; drift stats |
| 1 | `kl.fit` | on GSM8K train, depths 1–4 only: PCA(256) of the residual stream, DMD, Koopman autoencoder |
| 2 | `kl.evaluate` | depths 1–12 × 11 policies: GSM8K gold-solution NLL (primary), ARC-C accuracy (secondary), paired bootstrap verdict |
| 3 | `kl.generate` | GSM8K 4-shot greedy accuracy at depths 4/8/12 for none, best Koopman, best control |

Policies act on the state after every loop past 4:
controls `norm_match`, `pca_project@a`; Koopman `dmd_pull@a`, `kae_pull@a/k` (pull toward the
Koopman k-step rollout every k loops: periodic course correction).

**Verdict (pre-registered):** SURVIVES iff a Koopman policy at some depth > 4 has lower GSM8K NLL than
STOP *and* than the best control at that depth (95% bootstrap CIs exclude 0), with ARC-C no more than 1 pt below STOP.

## Run on TamIA
```bash
cd ~/koopman_loop
export HF_HOME=$SCRATCH/hf
bash scripts/setup_ouro.sh          # login node: venv ~/venvs/ouro (transformers 4.54.1), weights, data
source ~/venvs/ouro/bin/activate && bash tests/smoke.sh   # CPU check with a mock looped model, ~1 min
mkdir -p logs && sbatch slurm/ouro_poc.sh
grep -E "=====|\[check\]|verified|Error|Traceback" logs/koopman_ouro-*.out
cat runs/ouro14/REPORT.md
```
The job stops right after stage 0 if the hooks don't match the real model; send me the log in that case.
Other model: `MODEL=ByteDance/Ouro-2.6B OUT=runs/ouro26 sbatch slurm/ouro_poc.sh` (after downloading it).

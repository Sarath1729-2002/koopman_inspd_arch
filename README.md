# koopman_trm — is TRM's recursion a Koopman-friendly dynamical system? (ARC-AGI-1)

Viability study before building Koopman methods for tiny recursive reasoners.
Model: TRM (Jolicoeur-Martineau 2025, 7M params). Data: ARC-AGI-1 (TRM's own build, 1000 augmentations).

## Three stages
| stage | script | question | output |
|---|---|---|---|
| 1 | `ktrm.rollout` | What does the recursion trajectory look like, including **past the trained depth** (32 vs 16 ACT steps)? | `traj.npz`, accuracy per step |
| 2 | `ktrm.diagnose` | Is the H-cycle map s_t→s_{t+1} predictable, and linear? DMDc vs MLP vs Koopman-AE (pure latent / periodic re-encoding). Spectrum. Do early dynamics predict correctness? | `REPORT.md`, `diagnose.png` + a GO/CAUTION gate |
| 3 | `ktrm.explore` | PTRM-style exploration where only the **noise direction** changes: isotropic (=PTRM) vs PCA vs **slow Koopman modes** vs random subspace of equal rank | `EXPLORE.md` |

State per token = PCA(z_H) ⊕ PCA(z_L) after every H-cycle (3 per ACT step); control u = input embedding.
Splits for stage 2 are by original puzzle, so no augmentation of a test puzzle is seen in fitting.

## TamIA
```bash
git clone <this repo> ~/koopman_trm && cd ~/koopman_trm
bash scripts/setup_tamia.sh                      # login node: venv ~/venvs/ktrm, TRM code, ARC-1 data build
source ~/venvs/ktrm/bin/activate
bash tests/smoke.sh                              # CPU end-to-end check, ~2 min

# A) pretrained checkpoint, if one exists
export HF_HOME=$SCRATCH/hf
python scripts/find_checkpoints.py               # lists candidates
python scripts/find_checkpoints.py --download <repo_id>
CKPT=$SCRATCH/trm_ckpts/<repo>/<step_file> sbatch slurm/analyze.sh

# B) otherwise train one (paper: ~3 days, 4x H100)
EPOCHS=100000 sbatch slurm/train_trm_arc1.sh     # or EPOCHS=20000 EVAL_INTERVAL=5000 for a faster, weaker model
CKPT=checkpoints/trm_arc1_e100000/step_<N> sbatch slurm/analyze.sh
```
Results: `runs/arc1/REPORT.md`, `runs/arc1/diagnose.png`, `runs/arc1/EXPLORE.md`.

## Reading the gate
- **Relative z_H update after ACT step 2 < 0.01** → the recursion has frozen; there is little dynamics for Koopman to model (consistent with "shallow recursion" findings on ARC).
- **DMDc one-step skill > 0.3** → linear dynamics already capture the map → spectral methods (stage 3) are well-posed.
- **MLP ≫ DMDc and Koopman-AE closes the gap** → a learned lift earns its keep.
- **Re-encoding (every 3 steps) > pure latent at k=6** → the course-correction mechanism matters for reasoning trajectories.
- Stage 3 survives only if `koop_slow` beats both `iso_full` (PTRM) and `rand_sub` at matched sigma.

## Caveats
- A checkpoint is tied to its dataset build (TRM learns an embedding per augmented puzzle id). `rollout` warns if accuracy ≈ 0.
- Accuracy here is per augmented example with no voting (TRM's reported 45% uses 1000-way voting; per-example pass@1 ≈ 29–40%).
- Noise is injected after every H-cycle (PTRM injects per recursion step); selection uses the Q-head, as in PTRM.
- If `torch.compile` breaks training on the cluster, run with `DISABLE_COMPILE=1`.

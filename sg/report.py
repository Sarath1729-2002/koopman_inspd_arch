"""results_sg/eval.json -> REPORT.md, pre-registered verdict.

PRE-REGISTERED (before the GPU run). local = the paper's local error Delta x^(5); top bucket = most collisions.
  Two physics metrics, both required: local error (consistency with the true simulator over 5 frames) and energy
  error (relative kinetic-energy drift; catches predictions that slow down / blur, which local error alone rewards).
  PRECONDITION  the seriality gap appears in state space: on local OR energy error, the parallel/AR ratio in the top
                bucket is >= 1.2 AND >= 0.1 larger than in the bottom bucket. Otherwise MOOT.
  SURVIVES      some event-triggered Koopman setting is within 1.1 x AR on BOTH local and energy error (overall AND in
                the top bucket) using <= 16 mean serial passes (one third of AR's 48).
  STRONG        additionally, at a matched budget, event-triggered beats periodic re-encoding by >= 10% local error.
  CLOSED        otherwise."""
import argparse, json, os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_sg"); a = ap.parse_args()
r = json.load(open(os.path.join(a.root, "eval.json")))
A, Bk = r["arms"], r["buckets"]
lab = [f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in Bk]
out = ["# Seriality gap vs course-correcting Koopman (hard spheres, state space)", "",
       "Collision buckets = # ball-ball collisions in the true 48-frame horizon; counts: " +
       ", ".join(f"{l}: {c}" for l, c in zip(lab, r["bucket_counts"])), "",
       f"Metric floor on the true trajectories: local {r['floor']['local']:.5f}, energy {r['floor']['energy']:.3f}", "",
       "| arm | serial passes | local err | energy err | global err | " + " | ".join(f"local/energy [{l}]" for l in lab) + " |",
       "|---|---|---|---|---|" + "---|" * len(lab)]
for n, e in A.items():
    out.append(f"| {n} | {e['passes']:.1f} | {e['local']:.5f} | {e['energy']:.3f} | {e['global']:.4f} | " +
               " | ".join(f"{b['local']:.4f} / {b['energy']:.3f}" for b in e["by_bucket"]) + " |")
v = "INCOMPLETE"
if "parallel" in A and "ar" in A:
    ar, par = A["ar"], A["parallel"]
    gaps = {}
    for mtr in ("local", "energy"):
        rt = par["by_bucket"][-1][mtr] / max(1e-9, ar["by_bucket"][-1][mtr])
        rb = par["by_bucket"][0][mtr] / max(1e-9, ar["by_bucket"][0][mtr])
        gaps[mtr] = (rt, rb, rt >= 1.2 and rt >= rb + 0.1)
    pre = any(g[2] for g in gaps.values())
    near = lambda e, mtr: e[mtr] <= 1.1 * ar[mtr] and e["by_bucket"][-1][mtr] <= 1.1 * ar["by_bucket"][-1][mtr]
    ev = {n: e for n, e in A.items() if n.startswith("koop_event")}
    ok = {n: e for n, e in ev.items() if e["passes"] <= 16 and near(e, "local") and near(e, "energy")}
    per = sorted([(e["passes"], e["local"]) for n, e in A.items() if n.startswith("koop_every") or n == "koop_none"])
    strong = False
    for n, e in ok.items():
        xs, ys = zip(*per)
        ref = float(np.interp(e["passes"], xs, ys))
        strong = strong or e["local"] <= 0.9 * ref
    out += ["", "- seriality gap (parallel/AR ratio, top vs bottom bucket): " +
            "; ".join(f"{m} {g[0]:.2f} vs {g[1]:.2f}" for m, g in gaps.items()) + f" -> {'present' if pre else 'absent'}",
            f"- event-triggered settings meeting the AR-match @ <=16 passes: {sorted(ok) or 'none'}",
            f"- strong (event beats periodic at matched budget by >=10%): {strong}"]
    v = "MOOT (no seriality gap in state space)" if not pre else ("SURVIVES" + (" (STRONG)" if strong else "") if ok else "CLOSED")
out += ["", "## Pre-registered verdict", "", f"**VERDICT: {v}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

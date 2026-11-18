"""results/*.json -> results/REPORT.md, pre-registered verdict.

PRE-REGISTERED (before the GPU run). in-dist = positions 256-512 (inside the training context), far = last bucket.
  PRECONDITION (per core)  the base model drifts: far bpb >= in-dist bpb + 0.05. If no core drifts: MOOT.
  SURVIVES  on some drifting core, the best course-correction arm (ae = post-hoc, re64 = trained-in)
            has far bpb <= the best heuristic (reset / clamp) - 0.02, and (for re64) costs <= 0.02 bpb in-dist.
  CLOSED    otherwise.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); a = ap.parse_args()
R = {}
for p in glob.glob(os.path.join(a.root, "*.json")):
    r = json.load(open(p)); R.setdefault(r["core"], {})[r["arm"]] = r
out, verdicts = ["# Course correction for long-context drift in a byte LM (bits per byte by position)", ""], []
for core in sorted(R):
    rs = R[core]; ks = list(next(iter(rs.values()))["buckets"]); far, ind = ks[-1], "256-512"
    out += [f"## {core} (trained at context {next(iter(rs.values()))['ctx']})", "",
            "| arm | " + " | ".join(ks) + " | state rms in-dist → far |", "|---|" + "---|" * (len(ks) + 1)]
    for arm in [x for x in ["base", "reset", "clamp", "ae"] if x in rs] + sorted(x for x in rs if x.startswith("re") and x != "reset"):
        r = rs[arm]
        out.append(f"| {arm} | " + " | ".join(f"{r['buckets'][k]:.3f}" for k in ks) +
                   f" | {r['state_rms'][ind]:.3f} → {r['state_rms'][far]:.3f} |")
    if "base" not in rs:
        continue
    b = rs["base"]["buckets"]; drift = b[far] - b[ind]
    heur = min(rs[x]["buckets"][far] for x in ("reset", "clamp") if x in rs)
    cands = []
    if "ae" in rs:
        cands.append(("ae", rs["ae"]["buckets"][far]))
    for x in rs:
        if x.startswith("re") and x != "reset" and rs[x]["buckets"][ind] <= b[ind] + 0.02:
            cands.append((x, rs[x]["buckets"][far]))
    best = min(cands, key=lambda c: c[1]) if cands else ("none", 99)
    ok = drift >= 0.05 and best[1] <= heur - 0.02
    verdicts.append((drift >= 0.05, ok))
    out += ["", f"- drift (base far − in-dist): {drift:+.3f} (precondition needs ≥ +0.05) → {'drifts' if drift >= 0.05 else 'no drift'}",
            f"- best heuristic far: {heur:.3f}; best course correction: `{best[0]}` {best[1]:.3f} (needs ≤ {heur - 0.02:.3f})", ""]
v = "MOOT (no core drifts)" if not any(d for d, _ in verdicts) else ("SURVIVES" if any(o for _, o in verdicts) else "CLOSED")
out += ["## Pre-registered verdict", "", f"**VERDICT: {v}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

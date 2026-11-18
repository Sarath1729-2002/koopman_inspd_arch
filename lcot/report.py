"""results_lcot/*.json -> REPORT.md, pre-registered verdict.

PRE-REGISTERED (before the GPU run). acc = GSM8K test exact match (1319 items; ~1.3 pt standard error).
  PRECONDITION  Coconut shows the drift deficit: its final-stage accuracy is >= 3 pts below its best stage
                (collapse with more thoughts, as in SIM-CoT) OR >= 3 pts below CoT. Otherwise MOOT.
  SURVIVES      the best re-encoding arm's final-stage accuracy >= Coconut's final-stage accuracy + 3 pts.
                (Reported alongside: gap to CoT, thought homogeneity, vocab proximity, accuracy with extra thoughts.)
  CLOSED        otherwise.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_lcot"); a = ap.parse_args()
R = {json.load(open(p))["arm"]: json.load(open(p)) for p in glob.glob(os.path.join(a.root, "*.json"))}
cot = R.pop("cot", None)
cot_acc = cot["evals"][0]["acc"] if cot else None
f = lambda v: "—" if v is None else f"{v:.3f}"
stages = sorted({e["stage"] for r in R.values() for e in r["evals"] if e["stage"] != "extra"})
out = ["# Latent CoT with periodic re-encoding through the vocabulary (GPT-2, GSM8k-Aug)", "",
       f"CoT (stage 0) accuracy: {f(cot_acc)}", "",
       "| arm | " + " | ".join(f"stage {s}" for s in stages) + " | extra thoughts | thought cos (final) | vocab cos (final) |",
       "|---|" + "---|" * (len(stages) + 3)]
final = {}
for arm in ["coconut"] + sorted(x for x in R if x != "coconut"):
    if arm not in R:
        continue
    ev = {e["stage"]: e for e in R[arm]["evals"]}
    S = max(s for s in ev if s != "extra")
    final[arm] = (ev[S]["acc"], max(ev[s]["acc"] for s in ev if s != "extra"))
    out.append(f"| {arm} | " + " | ".join(f(ev[s]["acc"]) if s in ev else "—" for s in stages) +
               f" | {f(ev.get('extra', {}).get('acc'))} | {f(ev[S]['thought_cos'])} | {f(ev[S]['vocab_cos'])} |")
v = "INCOMPLETE"
if "coconut" in final:
    cf, cb = final["coconut"]
    pre = (cb - cf >= 0.03) or (cot_acc is not None and cot_acc - cf >= 0.03)
    snaps = {k: x for k, x in final.items() if k != "coconut"}
    best = max(snaps, key=lambda k: snaps[k][0]) if snaps else None
    gain = snaps[best][0] - cf if best else 0
    out += ["", f"- Coconut final {cf:.3f}, best stage {cb:.3f}, CoT {f(cot_acc)} -> drift deficit: {'yes' if pre else 'no'}",
            f"- best re-encoding arm `{best}` final {snaps[best][0]:.3f} -> gain over Coconut {gain:+.3f} (needs ≥ +0.030)" if best else "- no re-encoding arms",
            f"- gap of `{best}` to CoT: {snaps[best][0] - cot_acc:+.3f}" if best and cot_acc is not None else ""]
    v = "MOOT (Coconut shows no drift deficit)" if not pre else ("SURVIVES" if gain >= 0.03 else "CLOSED")
out += ["", "## Pre-registered verdict", "", f"**VERDICT: {v}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

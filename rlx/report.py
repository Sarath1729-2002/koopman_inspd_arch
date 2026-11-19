"""results_rlx/*.json -> REPORT.md.

PRE-REGISTERED.  acc = GSM8K test (1319). werr = ||pred - true|| / ||true - delta(T_cut)|| (weights, all tensors).
  PRECONDITION  RL improves the model: acc(150) - acc(0) >= 0.03, and extrapolation is meaningful: the best linear
                one-shot method (relex / line3) has werr@150 < 1.0 (better than staying at T_cut). Else MOOT.
  SURVIVES      (a) one-shot: koop3 werr@150 <= 0.8 x min(line3, linerec3, relex) werr@150 AND koop3 acc@150 >= best
                    linear acc@150 - 0.01;   or
                (b) course-corrected: cc_koop3 acc >= cc_line3 acc + 0.02 AND >= acc(150) - 0.01 (with ~43% real steps).
  CLOSED        otherwise."""
import argparse, json, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_rlx"); a = ap.parse_args()
L = lambda f: json.load(open(os.path.join(a.root, f))) if os.path.exists(os.path.join(a.root, f)) else None
ref, one, cck, ccl = L("ref.json"), L("oneshot.json"), L("cc_koop3.json"), L("cc_line3.json")
out = ["# Course-correcting Koopman extrapolation of GRPO training", ""]
if ref:
    out += ["Reference GRPO run (real steps): " + ", ".join(f"step {k}: {v:.3f}" for k, v in sorted(ref["acc"].items(), key=lambda x: int(x[0]))), ""]
M = ["relex", "line3", "linerec3", "koop3"]
if one:
    Ts = sorted({int(k.split("@")[1]) for k in one["acc"]})
    out += ["| method (from T_cut=%d real steps) | " % one["t_cut"] + " | ".join(f"acc@{T} | werr@{T}" for T in Ts) + " |",
            "|---|" + "---|---|" * len(Ts)]
    for m in M:
        out.append(f"| {m} | " + " | ".join(f"{one['acc'].get(f'{m}@{T}', float('nan')):.3f} | {one.get('werr', {}).get(f'{m}@{T}', float('nan')):.3f}" for T in Ts) + " |")
for c in (ccl, cck):
    if c:
        out.append(f"\n- course-corrected `{c['method']}`: acc {c['acc']:.3f} at equivalent step {c['t_final']} using {c['real_steps']} real steps")
v = "INCOMPLETE"
if ref and one and "150" in ref["acc"] and one.get("werr"):
    gain = ref["acc"]["150"] - ref["acc"]["0"]
    lin_err = min(one["werr"][f"{m}@150"] for m in ("relex", "line3"))
    pre = gain >= 0.03 and lin_err < 1.0
    best_lin_err = min(one["werr"][f"{m}@150"] for m in ("relex", "line3", "linerec3"))
    best_lin_acc = max(one["acc"][f"{m}@150"] for m in ("relex", "line3", "linerec3"))
    sa = one["werr"]["koop3@150"] <= 0.8 * best_lin_err and one["acc"]["koop3@150"] >= best_lin_acc - 0.01
    sb = bool(cck and ccl and cck["acc"] >= ccl["acc"] + 0.02 and cck["acc"] >= ref["acc"]["150"] - 0.01)
    out += ["", f"- RL gain 0->150: {gain:+.3f} (needs ≥ +0.03); best linear werr@150: {lin_err:.3f} (needs < 1.0)",
            f"- (a) koop3 werr@150 {one['werr']['koop3@150']:.3f} vs best linear {best_lin_err:.3f} (needs ≤ {0.8 * best_lin_err:.3f}); "
            f"acc {one['acc']['koop3@150']:.3f} vs {best_lin_acc:.3f} -> {sa}",
            f"- (b) course-corrected koop3 vs line3: {cck and cck['acc']} vs {ccl and ccl['acc']} -> {sb}"]
    v = "MOOT" if not pre else ("SURVIVES" if (sa or sb) else "CLOSED")
out += ["", "## Pre-registered verdict", "", f"**VERDICT: {v}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

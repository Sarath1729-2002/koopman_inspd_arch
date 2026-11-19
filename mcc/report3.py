"""results_mcc3/*.json -> REPORT.md. FINAL DECISION for the Mamba course-correction direction (iteration 3 of 3).

PRE-REGISTERED.  far = last PPL bucket (8192-12288); recall gains in nats/token.
  WORKS  on >= 2 models, the learned gate either
           (a) has far PPL <= 0.97 x window's far PPL, or
           (b) has recall gain at G=3072 >= window's + 0.10,
         AND keeps in-window recall: gain at G=512 >= 0.9 x window's.
  DEAD   otherwise (this form of course correction does not beat clean forgetting)."""
import argparse, glob, json, math, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_mcc3"); a = ap.parse_args()
out = ["# Course-correcting pretrained Mamba — iteration 3 (learned gate): final decision", ""]
wins = 0
for p in sorted(glob.glob(os.path.join(a.root, "*.json"))):
    r = json.load(open(p)); W = r["W"]; L = r["ppl"]["L"]; lc = r["ppl"]["loss_curve"]
    ppl = lambda arm, lo, hi: math.exp(sum(lc[arm][lo // 64: max(lo // 64 + 1, hi // 64)]) / max(1, (hi - lo) // 64))
    bk = [(W // 2, W), (W, 2 * W), (2 * W, 4 * W), (4 * W, L)]
    gs = list(r["recall"])
    out += [f"## {r['model']}  (gate keep-fraction after training: bias {r['gate_bias_mean']:.3f}, "
            f"last train g_mean {r['train_log'][-1]['g_mean']:.3f})" if r["train_log"] else f"## {r['model']}", "",
            "| arm | " + " | ".join(f"PPL {lo}-{hi}" for lo, hi in bk) + " | " + " | ".join(f"recall G={g}" for g in gs) + " |",
            "|---|" + "---|" * (len(bk) + len(gs))]
    for arm in r["arms"]:
        out.append(f"| {arm} | " + " | ".join(f"{ppl(arm, lo, hi):.2f}" for lo, hi in bk) + " | " +
                   " | ".join(f"{r['recall'][g][arm]['gain']:+.3f}" for g in gs) + " |")
    far = bk[-1]; gfar, gnear = max(gs, key=int), min(gs, key=int)
    pr = ppl("learned", *far) / ppl("window", *far)
    dr = r["recall"][gfar]["learned"]["gain"] - r["recall"][gfar]["window"]["gain"]
    keep = r["recall"][gnear]["learned"]["gain"] >= 0.9 * r["recall"][gnear]["window"]["gain"]
    ok = (pr <= 0.97 or dr >= 0.10) and keep
    wins += ok
    out += ["", f"- learned/window far PPL {pr:.3f} (≤ 0.97), far recall over window {dr:+.3f} (≥ +0.10), "
                f"in-window recall kept: {keep} -> {'WIN' if ok else 'no'}", ""]
out += ["## Pre-registered final decision", "", f"**{'WORKS' if wins >= 2 else 'DEAD'}** ({wins} model(s) pass; needs ≥ 2)"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

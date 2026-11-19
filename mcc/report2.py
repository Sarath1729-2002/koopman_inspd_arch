"""results_mcc2/*.json -> REPORT.md (iteration 2 of 3).

PRE-REGISTERED.
  PROMISING  on some model, a correction arm (obs3/obs5/blend50/blend25) has recall gain at the far gap (G=3072)
             >= max(window, clamp) gain + 0.10 nats/token, AND its far natural PPL is within 1% of the best of
             window/clamp.  -> iteration 3 (scale up, PG-19-style long text, LongMamba baseline).
  NOT YET    otherwise; DEAD if iteration 3 is still not promising."""
import argparse, glob, json, math, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_mcc2"); a = ap.parse_args()
out = ["# Course-correcting pretrained Mamba — iteration 2: correct without forgetting", ""]
CORR = ["obs3", "obs5", "blend50", "blend25"]
prom = []
for p in sorted(glob.glob(os.path.join(a.root, "*.json"))):
    r = json.load(open(p)); W = r["W"]; L = r["ppl"]["L"]; lc = r["ppl"]["loss_curve"]
    ppl = lambda arm, lo, hi: math.exp(sum(lc[arm][lo // 64: max(lo // 64 + 1, hi // 64)]) / max(1, (hi - lo) // 64))
    bk = [(W // 2, W), (W, 2 * W), (2 * W, 4 * W), (4 * W, L)]
    out += [f"## {r['model']}", "", "| arm | " + " | ".join(f"PPL {lo}-{hi}" for lo, hi in bk) + " | " +
            " | ".join(f"recall gain G={g}" for g in r["recall"]) + " |", "|---|" + "---|" * (len(bk) + len(r["recall"]))]
    for arm in r["arms"]:
        out.append(f"| {arm} | " + " | ".join(f"{ppl(arm, lo, hi):.2f}" for lo, hi in bk) + " | " +
                   " | ".join(f"{r['recall'][g][arm]['gain']:+.3f}" for g in r["recall"]) + " |")
    far = bk[-1]; gfar = max(r["recall"], key=int)
    heur_ppl = min(ppl("window", *far), ppl("clamp", *far))
    heur_gain = max(r["recall"][gfar]["window"]["gain"], r["recall"][gfar]["clamp"]["gain"])
    ok = [(c, r["recall"][gfar][c]["gain"] - heur_gain, ppl(c, *far) / heur_ppl) for c in CORR]
    best = max(ok, key=lambda x: x[1])
    prom.append(any(d >= 0.10 and q <= 1.01 for _, d, q in ok))
    out += ["", f"- far-gap recall: none {r['recall'][gfar]['none']['gain']:+.3f}, best heuristic {heur_gain:+.3f}; "
                f"best correction `{best[0]}` {best[1]:+.3f} over heuristic (needs ≥ +0.10), far PPL ratio {best[2]:.3f} (needs ≤ 1.01)", ""]
out += ["## Pre-registered verdict (iteration 2)", "", f"**{'PROMISING -> iteration 3' if any(prom) else 'NOT YET (one iteration left)'}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

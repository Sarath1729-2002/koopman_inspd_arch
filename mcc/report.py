"""results_mcc/*.json -> REPORT.md. Iteration 1 of a 3-iteration budget.

PRE-REGISTERED.  PPL buckets by position; in-dist = [W/2, W), far = last bucket [L/2, L).
  PRECONDITION (per model)  'none' drifts: far PPL >= 1.10 x in-dist PPL.  No model drifts -> MOOT.
  PROMISING                 on some drifting model, best re-encoding arm (obs_elem / obs_chan) has far PPL
                            <= 0.97 x min(window, clamp) far PPL, and PPL on [W, 2W) no worse than 'none' + 2%.
                            -> iterate (iteration 2: tune W-shadow age, every, tau; add LongMamba-style baseline).
  NOT YET                   otherwise; the direction is called DEAD only if iteration 3 is still not promising."""
import argparse, glob, json, math, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_mcc"); a = ap.parse_args()
out = ["# Course-correcting pretrained Mamba past its training length (iteration 1)", ""]
verdicts = []
for p in sorted(glob.glob(os.path.join(a.root, "*.json"))):
    r = json.load(open(p)); W, L = r["W"], r["L"]
    ppl = lambda arm, lo, hi: math.exp(sum(r["loss_curve"][arm][lo // 64: hi // 64]) / max(1, (hi - lo) // 64))
    edges = [0, W // 2, W, 2 * W] + [2 ** k * W for k in range(2, 10) if 2 ** k * W <= L]
    if edges[-1] < L:
        edges.append(L)
    bk = list(zip(edges[:-1], edges[1:]))
    out += [f"## {r['model']} (W={W}, {r['n_seq']} sequences of {L} tokens)", "",
            "| arm | " + " | ".join(f"{lo}-{hi}" for lo, hi in bk) + " |", "|---|" + "---|" * len(bk)]
    for arm in r["arms"]:
        out.append(f"| {arm} | " + " | ".join(f"{ppl(arm, lo, hi):.2f}" for lo, hi in bk) + " |")
    far, ind = bk[-1], (W // 2, W)
    drift = ppl("none", *far) / ppl("none", *ind)
    heur = min(ppl("window", *far), ppl("clamp", *far))
    best = min(("obs_elem", "obs_chan"), key=lambda x: ppl(x, *far))
    ok_near = ppl(best, W, 2 * W) <= 1.02 * ppl("none", W, 2 * W)
    prom = drift >= 1.10 and ppl(best, *far) <= 0.97 * heur and ok_near
    verdicts.append((drift >= 1.10, prom))
    df = r["drift_frac_by_block"]
    out += ["", f"- drift ratio far/in-dist PPL ('none'): {drift:.2f} (precondition ≥ 1.10)",
            f"- best heuristic far {heur:.2f}; best re-encoding `{best}` far {ppl(best, *far):.2f} "
            f"(needs ≤ {0.97 * heur:.2f}); near-range cost ok: {ok_near}",
            "- fraction of state elements outside in-dist ±tau·sd ('none', mid layer) by W-block: " +
            " ".join(f"{k}:{v:.3f}" for k, v in df.items() if v is not None), ""]
v = "MOOT (no model drifts)" if not any(d for d, _ in verdicts) else ("PROMISING -> iteration 2" if any(p for _, p in verdicts) else "NOT YET (iterate; DEAD only after iteration 3)")
out += ["## Pre-registered verdict (iteration 1)", "", f"**{v}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

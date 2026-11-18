"""Aggregate <root>/*_s*.json -> REPORT.md + s5.png with the pre-registered verdict.

PRE-REGISTERED (written before any run):
  VALID       the test means something only if lstm or deltaprod reaches >= 0.95 per-token accuracy at the
              training length (mean over seeds). Otherwise INCONCLUSIVE (training budget too small).
  SURVIVES    some boundary arm with k >= 8 reaches per-token accuracy >= 0.90 at 4x the training length
              AND is no more than 0.05 below deltaprod there
              AND beats the plain linear RNN (lrnn) there by >= 0.10 (else the task was too easy to discriminate).
              (k <= 4 does not count: that is a nonlinear RNN at stride <= 4, with no throughput story.)
  CLOSED      otherwise.
"""
import argparse
import glob
import json
import os

import numpy as np

ORDER = ["lstm", "deltaprod", "lrnn", "bnd2", "bnd4", "bnd8", "bnd16", "bnd32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    R = {}
    for p in glob.glob(os.path.join(a.root, "*_s*.json")):
        r = json.load(open(p))
        if "arm" in r and "eval" in r:
            R.setdefault(r["arm"], []).append(r)
    arms = [x for x in ORDER if x in R] + sorted(set(R) - set(ORDER))
    tl = next(iter(R.values()))[0]["train_len"]
    Ls = sorted(int(k) for k in next(iter(R.values()))[0]["eval"])
    m = lambda arm, L, key="acc": np.array([r["eval"][str(L)][key] for r in R[arm]])
    g = next(iter(R.values()))[0].get("group", "s5")
    out = [f"# {g.upper()} per-token state tracking — periodic re-encoding kill test", "",
           f"Trained at length <= {tl}. Per-token accuracy, mean ± std over seeds.", "",
           "| arm | seeds | params | " + " | ".join(f"L={L}" for L in Ls) + " | boundary / mid-chunk @4x |",
           "|---|---|---|" + "---|" * len(Ls) + "---|"]
    L4 = 4 * tl if 4 * tl in Ls else max(Ls)
    for arm in arms:
        cells = [f"{m(arm, L).mean():.3f} ± {m(arm, L).std():.3f}" for L in Ls]
        bm = (f"{m(arm, L4, 'acc_at_boundary').mean():.3f} / {m(arm, L4, 'acc_mid_chunk').mean():.3f}"
              if arm.startswith("bnd") else "")
        out.append(f"| {arm} | {len(R[arm])} | {R[arm][0]['params'] / 1e6:.2f}M | " + " | ".join(cells) + f" | {bm} |")
    ref = [x for x in ("lstm", "deltaprod") if x in R]
    valid = any(m(x, tl).mean() >= 0.95 for x in ref)
    dp = m("deltaprod", L4).mean() if "deltaprod" in R else 0.0
    cand = [(x, m(x, L4).mean()) for x in arms if x.startswith("bnd") and int(x[3:]) >= 8]
    lr = m("lrnn", L4).mean() if "lrnn" in R else 0.0
    surv = [(x, v) for x, v in cand if v >= 0.90 and v >= dp - 0.05 and v >= lr + 0.10]
    out += ["", "## Pre-registered verdict", "",
            f"- validity: best reference at train length = "
            + ", ".join(f"{x} {m(x, tl).mean():.3f}" for x in ref) + f" (needs >= 0.95) -> {'VALID' if valid else 'INVALID'}",
            f"- deltaprod at L={L4}: {dp:.3f}   lrnn at L={L4}: {lr:.3f}",
            "- boundary arms with k >= 8 at L=" + str(L4) + ": " + (", ".join(f"{x} {v:.3f}" for x, v in cand) or "none run"),
            "", f"**VERDICT: {'INCONCLUSIVE (references did not learn S5; raise --steps)' if not valid else ('SURVIVES: ' + ', '.join(x for x, _ in surv)) if surv else 'CLOSED'}**"]
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.root, "REPORT.md"), "w").write(txt)
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in arms:
            mu = [m(arm, L).mean() for L in Ls]
            ax.plot(Ls, mu, marker="o", label=arm, ls="--" if arm in ("lstm", "deltaprod", "lrnn") else "-")
        ax.axvline(tl, c="k", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel="sequence length", ylabel="per-token accuracy", ylim=(-0.02, 1.02),
               title=f"{g.upper()} state tracking (dotted = train length)")
        ax.set_xticks(Ls); ax.set_xticklabels(Ls)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(a.root, f"{g}.png"), dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()

"""Aggregate <root>/<task>/<arm>_s<seed>.json -> REPORT.md + drift.png.

PRE-REGISTERED (written before the GPU run):
  PRECONDITION  drift is real: on every task, the best non-re-encoding linear arm (rot / unit / norm16)
                is below 0.90 per-token accuracy at the longest length. Otherwise MOOT (nothing to correct).
  SURVIVES      on every task, re16 or re32 reaches >= 0.98 at the longest length AND beats the best
                non-re-encoding linear arm there by >= 0.20.
  CLOSED        otherwise.
"""
import argparse, glob, json, os

import numpy as np

ORDER = ["lstm", "rot", "unit", "norm16", "re4", "re16", "re32", "ure16"]
BASE = ["rot", "unit", "norm16"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); a = ap.parse_args()
    R = {}
    for p in glob.glob(os.path.join(a.root, "*", "*_s*.json")):
        r = json.load(open(p))
        if "task" in r and "eval" in r:
            R.setdefault(r["task"], {}).setdefault(r["arm"], []).append(r)
    out = ["# Re-encoding as error correction — drift kill test", ""]
    pre, surv = [], []
    for t in sorted(R):
        arms = [x for x in ORDER if x in R[t]] + sorted(set(R[t]) - set(ORDER))
        any_ = next(iter(R[t].values()))[0]
        Ls = sorted(int(k) for k in any_["eval"]); Lmax = max(Ls)
        m = lambda arm, L: np.array([r["eval"][str(L)]["acc"] for r in R[t][arm]])
        out += [f"## {t} (trained at length <= {any_['train_len']}; per-token accuracy, mean ± std over seeds)", "",
                "| arm | seeds | " + " | ".join(f"L={L}" for L in Ls) + " |", "|---|---|" + "---|" * len(Ls)]
        for arm in arms:
            out.append(f"| {arm} | {len(R[t][arm])} | " + " | ".join(f"{m(arm, L).mean():.3f} ± {m(arm, L).std():.3f}" for L in Ls) + " |")
        bb = max([(m(x, Lmax).mean(), x) for x in BASE if x in R[t]], default=(0.0, "none"))
        br = max([(m(x, Lmax).mean(), x) for x in ("re16", "re32") if x in R[t]], default=(0.0, "none"))
        pre.append(bb[0] < 0.90); surv.append(br[0] >= 0.98 and br[0] >= bb[0] + 0.20)
        out += ["", f"- best baseline at L={Lmax}: `{bb[1]}` {bb[0]:.3f}; best re-encoder (k=16/32): `{br[1]}` {br[0]:.3f}", ""]
    v = "MOOT (no drift to correct)" if not all(pre) else ("SURVIVES" if all(surv) else "CLOSED")
    out += ["## Pre-registered verdict", "", f"precondition (drift on every task): {all(pre)}; "
            f"re-encoding fixes it on every task: {all(surv)}", "", f"**VERDICT: {v}**"]
    txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(R), figsize=(6 * len(R), 4), squeeze=False)
        for j, t in enumerate(sorted(R)):
            for arm in [x for x in ORDER if x in R[t]]:
                Ls = sorted(int(k) for k in R[t][arm][0]["eval"])
                ax[0, j].plot(Ls, [np.mean([r["eval"][str(L)]["acc"] for r in R[t][arm]]) for L in Ls], marker="o",
                              label=arm, ls="-" if arm.startswith(("re", "ure")) else "--")
            ax[0, j].set(xscale="log", title=t, xlabel="length", ylabel="per-token acc", ylim=(0, 1.02)); ax[0, j].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.root, "drift.png"), dpi=130)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()

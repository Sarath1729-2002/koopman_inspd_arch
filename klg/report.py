"""Aggregate runs/<task>/<arm>_s<seed>.json -> runs/REPORT.md + runs/phase1.png, with the pre-registered verdict.

Pre-registered kill rule (Phase 1):
  SURVIVES (extrapolation)  if, averaged over tasks, the best Koopman arm (koopman / koopman_re) beats the best
                            control arm (plain / gainctl) by >= 0.10 mean exact-match on OOD lengths (oracle loops);
  SURVIVES (stability)      or if the plain arm loses > 0.20 accuracy when over-looped (2T) while the best Koopman
                            arm loses <= 0.05, averaged over tasks;
  otherwise CLOSED.  Per-task means are over seeds; the std over seeds is reported next to every number.
"""
import argparse
import glob
import json
import os

import numpy as np

ARMS = ["plain", "gainctl", "koopman", "koopman_re"]
CONTROLS, KOOP = ["plain", "gainctl"], ["koopman", "koopman_re"]


def load(root):
    R = {}
    for p in glob.glob(os.path.join(root, "*", "*_s*.json")):
        r = json.load(open(p))
        R.setdefault(r["task"], {}).setdefault(r["arm"], []).append(r)
    return R


def summ(runs):
    """per-seed scalars -> (mean, std) dict"""
    rows = []
    for r in runs:
        tm = r["train_max"]
        ev = {int(k): v for k, v in r["eval"].items()}
        ood = [v for L, v in ev.items() if L > tm]
        rows.append({
            "in_dist": ev[tm]["oracle"],
            "ood_oracle": np.mean([v["oracle"] for v in ood]),
            "ood_confstop": np.mean([v["confstop"] for v in ood]),
            "all_oracle": np.mean([v["oracle"] for v in ev.values()]),
            "all_overloop": np.mean([v["overloop"] for v in ev.values()]),
            "norm_ratio": np.mean([v["norm_ratio"] for v in ev.values()]),
        })
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()
    R = load(a.root)
    tasks = sorted(R)
    S = {t: {arm: summ(R[t][arm]) for arm in ARMS if arm in R[t]} for t in tasks}
    L = ["# Phase 1 — Koopman-stable looped core vs looped transformer (length generalisation)", "",
         "Exact-match accuracy. OOD = lengths beyond the training max, loops = T(n) (oracle). "
         "Overloop = 2·T(n) loops, all lengths. Confstop = loops picked by max confidence in 1..3T (no oracle). mean ± std over seeds.", "",
         "| task | arm | seeds | in-dist (train max) | OOD oracle | OOD confstop | oracle (all) | overloop (all) | ‖h_2T‖/‖h_T‖ |",
         "|---|---|---|---|---|---|---|---|---|"]
    f = lambda m: f"{m[0]:.3f} ± {m[1]:.3f}"
    for t in tasks:
        for arm, (s, n) in S[t].items():
            L.append(f"| {t} | {arm} | {n} | {f(s['in_dist'])} | {f(s['ood_oracle'])} | {f(s['ood_confstop'])} | "
                     f"{f(s['all_oracle'])} | {f(s['all_overloop'])} | {s['norm_ratio'][0]:.2f} |")
    # verdict
    gains, stab = [], []
    for t in tasks:
        st = S[t]
        if not all(x in st for x in ARMS):
            continue
        bk = max(KOOP, key=lambda k: st[k][0]["ood_oracle"][0])
        bc = max(CONTROLS, key=lambda k: st[k][0]["ood_oracle"][0])
        gains.append((t, bk, bc, st[bk][0]["ood_oracle"][0] - st[bc][0]["ood_oracle"][0]))
        drop = lambda k: st[k][0]["all_oracle"][0] - st[k][0]["all_overloop"][0]
        bks = min(KOOP, key=drop)
        stab.append((t, drop("plain"), bks, drop(bks)))
    L += ["", "## Pre-registered verdict", ""]
    if gains:
        mg = float(np.mean([g[3] for g in gains]))
        mp = float(np.mean([s[1] for s in stab]))
        mk = float(np.mean([s[3] for s in stab]))
        for t, bk, bc, g in gains:
            L.append(f"- {t}: best Koopman `{bk}` vs best control `{bc}` on OOD: **{g:+.3f}**")
        for t, dp, bks, dk in stab:
            L.append(f"- {t}: accuracy lost when over-looping — plain {dp:+.3f}, `{bks}` {dk:+.3f}")
        s1, s2 = mg >= 0.10, (mp > 0.20 and mk <= 0.05)
        L += ["", f"Mean OOD gain {mg:+.3f} (needs ≥ +0.10) → {'PASS' if s1 else 'fail'}. "
              f"Over-loop loss plain {mp:.3f} vs Koopman {mk:.3f} (needs >0.20 vs ≤0.05) → {'PASS' if s2 else 'fail'}.", "",
              f"**VERDICT: {'SURVIVES' + (' (extrapolation)' if s1 else '') + (' (stability)' if s2 else '') if (s1 or s2) else 'CLOSED'}**"]
    open(os.path.join(a.root, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plot(R, tasks, a.root)


def plot(R, tasks, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"plain": "#7f7f7f", "gainctl": "#1f77b4", "koopman": "#d62728", "koopman_re": "#ff7f0e"}
    fig, ax = plt.subplots(2, len(tasks), figsize=(5.2 * len(tasks), 8), squeeze=False)
    for j, t in enumerate(tasks):
        for arm in ARMS:
            if arm not in R[t]:
                continue
            runs = R[t][arm]
            Ls = sorted(int(k) for k in runs[0]["eval"])
            for key, ls in (("oracle", "-"), ("overloop", ":")):
                M = np.array([[r["eval"][str(L)][key] for L in Ls] for r in runs])
                ax[0, j].plot(Ls, M.mean(0), ls, c=col[arm], marker="o", ms=3, label=f"{arm} {key}")
            tm = runs[0]["train_max"]
            cur = [r["eval"][str(max(Ls))]["curve"] for r in runs if r["eval"][str(max(Ls))]["curve"]]
            if cur:
                C = np.array(cur).mean(0)
                ax[1, j].plot(np.arange(1, len(C) + 1), C, c=col[arm], label=arm)
        ax[0, j].axvline(tm, c="k", lw=0.8, ls="--")
        ax[0, j].set(title=f"{t}: exact match vs length (dashed = train max)", xlabel="length n", ylim=(-0.02, 1.02))
        Tn = runs[0]["eval"][str(max(Ls))]["T"]
        ax[1, j].axvline(Tn, c="k", lw=0.8, ls="--")
        ax[1, j].set(title=f"{t}: accuracy vs loops at n={max(Ls)} (dashed = T(n))", xlabel="loops", ylim=(-0.02, 1.02))
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "phase1.png"), dpi=130)


if __name__ == "__main__":
    main()

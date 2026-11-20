"""results_sg2/eval_s*.json -> REPORT.md (+ figure). Means +- std over seeds. Pre-registered claims:

  C1 seriality gap reproduced: parallel diffusion (block48, N=50) has local-error ratio vs AR (block1) >= 1.2 in the
     most-collision bucket AND that ratio is >= 0.1 higher than in the fewest-collision bucket; and 4x more denoising
     steps (N=200) does not cut parallel local error by more than 10%.
  C2 Koopman closes it cheaply: the best event-triggered koopdiff with <= 8 mean passes has local error <= 1.15 x AR
     (overall and in the most-collision bucket) and <= 0.7 x parallel.
  C3 it is the linear dynamics, not capacity: koopdiff beats the capacity-matched featdiff control by >= 10% local
     error at matched passes (1 pass and 8 passes; best koopdiff schedule vs featdiff periodic).
  C4 events beat schedules: event-triggered koopdiff beats periodic koopdiff by >= 5% at matched passes (4 and 8).
Each claim PASS/FAIL on seed means; the headline needs C1 and C2."""
import argparse, glob, json, os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_sg2"); a = ap.parse_args()
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.root, "eval_s*.json")))]
if not runs:
    raise SystemExit("no eval files")
arms = sorted({k for r in runs for k in r["arms"]}, key=lambda k: (not k.startswith("block"), k))
B = runs[0]["buckets"]; lab = [f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in B]


def stat(arm, f):
    v = [f(r["arms"][arm]) for r in runs if arm in r["arms"]]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), float("nan"), 0)


fm = lambda m: f"{m[0]:.4f} ± {m[1]:.4f}"
out = ["# Koopman course correction vs the seriality gap — generative proof of concept", "",
       f"Seeds: {len(runs)}. Test trajectories per seed: {sum(runs[0]['bucket_counts'])}; collision buckets "
       + ", ".join(f"{l}: {c}" for l, c in zip(lab, runs[0]["bucket_counts"])) + ".",
       f"Metric floor (true trajectories): local {runs[0]['floor']['local']:.4f}, energy {runs[0]['floor']['energy']:.3f}, "
       f"overlap {runs[0]['floor']['overlap']:.3f}. Denoising steps per pass N = {runs[0]['N']}.", "",
       "| arm | serial passes | local error | energy error | overlap rate | " + " | ".join(f"local [{l}]" for l in lab) + " |",
       "|---|---|---|---|---|" + "---|" * len(lab)]
for arm in arms:
    p = stat(arm, lambda e: e["passes"])
    out.append(f"| {arm} | {p[0]:.1f} | {fm(stat(arm, lambda e: e['local']))} | {stat(arm, lambda e: e['energy'])[0]:.3f} | "
               f"{stat(arm, lambda e: e['overlap'])[0]:.3f} | " +
               " | ".join(f"{stat(arm, lambda e, i=i: e['by_bucket'][i]['local'])[0]:.4f}" for i in range(len(B))) + " |")
out += ["", "Local error by horizon quarter (frames 6-17 / 18-29 / 30-41 / 42-47):", ""]
for arm in arms:
    out.append(f"- {arm}: " + " / ".join(f"{stat(arm, lambda e, i=i: e['by_quarter'][i])[0]:.4f}" for i in range(4)))
M = lambda arm, f=lambda e: e["local"]: stat(arm, f)[0]
top = lambda arm: M(arm, lambda e: e["by_bucket"][-1]["local"])
bot = lambda arm: M(arm, lambda e: e["by_bucket"][0]["local"])
res = {}
if all(x in arms for x in ("block48", "block1")):
    rt, rb = top("block48") / top("block1"), bot("block48") / bot("block1")
    steps_ok = ("block48_N200" not in arms) or (M("block48_N200") >= 0.9 * M("block48"))
    res["C1"] = (rt >= 1.2 and rt >= rb + 0.1 and steps_ok,
                 f"parallel/AR local ratio: most-collision bucket {rt:.2f}, fewest {rb:.2f}; N=200 vs N=50 parallel: "
                 f"{M('block48_N200') if 'block48_N200' in arms else float('nan'):.4f} vs {M('block48'):.4f}")
    evts = [x for x in arms if x.startswith("koopdiff_evt") and M(x, lambda e: e["passes"]) <= 8.5]
    good = [x for x in evts if M(x) <= 1.15 * M("block1") and top(x) <= 1.15 * top("block1") and M(x) <= 0.7 * M("block48")]
    best = min(evts, key=M) if evts else None
    res["C2"] = (bool(good), f"event-triggered arms <= 8 passes: " + (", ".join(f"{x} {M(x):.4f} (passes {M(x, lambda e: e['passes']):.1f}, "
                 f"top bucket {top(x):.4f})" for x in evts) or "none") + f"; AR {M('block1'):.4f} (top {top('block1'):.4f}), parallel {M('block48'):.4f}")
    c3 = []
    for P, karms, farm in ((1, ["koopdiff_1"], "featdiff_1"), (8, ["koopdiff_per8", "koopdiff_evt8"], "featdiff_per8")):
        ks = [x for x in karms if x in arms]
        if ks and farm in arms:
            kb = min(M(x) for x in ks)
            c3.append((kb <= 0.9 * M(farm), f"{P} pass(es): koopdiff {kb:.4f} vs featdiff {M(farm):.4f}"))
    res["C3"] = (bool(c3) and all(c[0] for c in c3), "; ".join(c[1] for c in c3) or "missing arms")
    c4 = []
    for P in (4, 8):
        if f"koopdiff_evt{P}" in arms and f"koopdiff_per{P}" in arms:
            c4.append((M(f"koopdiff_evt{P}") <= 0.95 * M(f"koopdiff_per{P}"), f"{P}: event {M(f'koopdiff_evt{P}'):.4f} vs periodic {M(f'koopdiff_per{P}'):.4f}"))
    res["C4"] = (bool(c4) and all(c[0] for c in c4), "; ".join(c[1] for c in c4) or "missing arms")
out += ["", "## Pre-registered claims", ""]
for k in ("C1", "C2", "C3", "C4"):
    if k in res:
        out.append(f"- **{k}: {'PASS' if res[k][0] else 'FAIL'}** — {res[k][1]}")
head = res.get("C1", (False,))[0] and res.get("C2", (False,))[0]
out += ["", f"**HEADLINE: {'SUPPORTED' if head else 'NOT SUPPORTED'}** (needs C1 and C2; C3/C4 strengthen it)"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm, st in (("block48", "s-"), ("block8", "^-"), ("block1", "o-"), ("koopdiff_1", "D--"), ("featdiff_1", "x:")):
        if arm in arms:
            ax[0].plot(lab, [stat(arm, lambda e, i=i: e["by_bucket"][i]["local"])[0] for i in range(len(B))], st, label=arm)
    ax[0].set(xlabel="ball-ball collisions in horizon", ylabel="local error", title="Physics error vs chain length")
    ax[0].legend(fontsize=8)
    for fam, st in (("koopdiff_evt", "o-"), ("koopdiff_per", "s--"), ("featdiff_per", "x:")):
        pts = sorted((M(x, lambda e: e["passes"]), M(x)) for x in arms if x.startswith(fam))
        if pts:
            ax[1].plot(*zip(*pts), st, label=fam)
    for arm, mk in (("block1", "*"), ("block48", "P"), ("block8", "^")):
        if arm in arms:
            ax[1].scatter([M(arm, lambda e: e["passes"])], [M(arm)], marker=mk, s=120, label=arm, zorder=5)
    ax[1].set(xscale="log", xlabel="serial passes (log)", ylabel="local error", title="Physics error vs serial compute")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.root, "poc.png"), dpi=140)
    print("figure ->", os.path.join(a.root, "poc.png"))
except Exception as e:
    print("figure skipped:", e)

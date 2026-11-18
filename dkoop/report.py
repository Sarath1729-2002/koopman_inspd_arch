"""results_depth/*.json -> REPORT.md with the pre-registered verdict.

PRE-REGISTERED. gap(arm) = ppl(arm) - ppl(full).
  PRECONDITION  error compounds over contiguous linear depth: gap(lin_contig, largest m) >= 2.5 x gap(lin_contig, smallest m)
                when the largest m is 2x the smallest (scaled proportionally otherwise). Else MOOT.
  SURVIVES      at >= 2 budgets: gap(lin_inter) <= 0.9 x gap(lin_contig)  AND  gsm8k(lin_inter) >= gsm8k(lin_contig) - 0.02.
  CLOSED        otherwise.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_depth"); a = ap.parse_args()
full = json.load(open(os.path.join(a.root, "full.json")))["arms"][0]
R = sorted((json.load(open(p)) for p in glob.glob(os.path.join(a.root, "m*.json"))), key=lambda r: r["m"])
f = lambda v: "—" if v is None else f"{v:.3f}"
out = ["# Course-correcting Koopman across depth — interleaved vs contiguous linear replacement", "",
       f"full model: ppl {full['ppl']:.3f}, gsm8k {f(full['gsm8k'])}", "",
       "| m (of L) | arm | layers | ppl | gap to full | gsm8k |", "|---|---|---|---|---|---|"]
G = {}
for r in R:
    for x in r["arms"]:
        g = x["ppl"] - full["ppl"]; G[(r["m"], x["arm"])] = (g, x["gsm8k"])
        out.append(f"| {r['m']} of {r['L']} | {x['arm']} | {x['layers']} | {x['ppl']:.3f} | {g:+.3f} | {f(x['gsm8k'])} |")
ms = [r["m"] for r in R]
lo, hi = min(ms), max(ms)
need = 2.5 * (hi / lo) / 2
ratio = G[(hi, "lin_contig")][0] / max(1e-9, G[(lo, "lin_contig")][0])
pre = ratio >= need
wins = []
for m in ms:
    if (m, "lin_inter") not in G:
        continue
    gi, ai = G[(m, "lin_inter")]; gc, ac = G[(m, "lin_contig")]
    ok = gi <= 0.9 * gc and (ai is None or ac is None or ai >= ac - 0.02)
    wins.append(ok)
    out.append("") if m == ms[0] else None
    out.append(f"- m={m}: lin_inter gap {gi:+.3f} vs lin_contig {gc:+.3f} ({(1 - gi / gc) * 100 if gc else 0:+.0f}% reduction), "
               f"gsm8k {f(ai)} vs {f(ac)} -> {'win' if ok else 'no'}")
out += ["", f"- compounding: gap ratio m={hi}/m={lo} = {ratio:.2f} (needs ≥ {need:.2f}) -> {'yes' if pre else 'no'}", "",
        "## Pre-registered verdict", "",
        f"**VERDICT: {'MOOT (no compounding over contiguous depth)' if not pre else ('SURVIVES' if sum(wins) >= 2 else 'CLOSED')}**"]
txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)

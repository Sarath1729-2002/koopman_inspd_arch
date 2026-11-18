"""results_rif/*.jsonl -> REPORT.md: per-window trace compliance, accuracy, decay timescale, memory-vs-control probe.

PRE-REGISTERED (before the GPU run). Windows = 64 generated thinking tokens (injected reminders excluded).
  PRECONDITION  base arm, averaged over instructions: window-0 compliance >= 0.5 AND mean compliance over windows
                >= 8 (512+ tokens) is >= 0.20 below window 0 (the instruction decays). Otherwise MOOT.
  SURVIVES      some re-encoding arm with k >= 128 raises mean trace compliance by >= 0.15 over base, with GSM8K
                accuracy dropping by <= 0.03.
  CLOSED        otherwise.
Reported, not gating: decay half-life from the base curve vs the best k; probe for the instruction identity on
VIOLATING windows only (surface text can't reveal it there): >= 0.60 = instruction still held (control failure),
<= 0.45 = instruction lost (memory failure); chance = 1/3."""
import argparse, glob, json, os

import numpy as np
import torch

from .checks import answer_of, correct, window_ok

W = 64


def analyse(rows, tok, end_think):
    per = []
    for r in rows:
        ids = r["gen_ids"]
        cut = ids.index(end_think) if end_think in ids else len(ids)
        think, ans = ids[:cut], ids[cut + 1:]
        wins = [window_ok(r["kind"], tok.decode(think[i:i + W])) for i in range(0, len(think), W)]
        atext = tok.decode(ans, skip_special_tokens=True)
        per.append({"kind": r["kind"], "item": r["item"], "wins": wins,
                    "ans_ok": window_ok(r["kind"], atext) if atext.strip() else None,
                    "acc": correct(answer_of(atext if ans else tok.decode(think)), r["gold"]), "finished": end_think in ids})
    return per


def curve(per, n=20):
    c = []
    for w in range(n):
        v = [p["wins"][w] for p in per if len(p["wins"]) > w and p["wins"][w] is not None]
        c.append(float(np.mean(v)) if len(v) >= 10 else None)
    return c


def mean_comp(per):
    v = [x for p in per for x in p["wins"] if x is not None]
    return float(np.mean(v)) if v else float("nan")


def probe(path, base_rows, base_per, kinds, seed=0):
    S = torch.load(path)
    idx, pos, H = S["idx"].numpy(), S["pos"].numpy(), S["h"].float()
    lab_kind = np.array([kinds.index(base_rows[i]["kind"]) for i in idx])
    viol = np.array([(lambda w: w is False)(base_per[i]["wins"][p // W] if p // W < len(base_per[i]["wins"]) else None) for i, p in zip(idx, pos)])
    items = np.array([base_rows[i]["item"] for i in idx])
    rng = np.random.default_rng(seed)
    uit = np.unique(items); rng.shuffle(uit); test_items = set(uit[: len(uit) * 3 // 10])
    te = np.array([it in test_items for it in items])
    res = {}
    for name, sel in (("violating", viol), ("all", np.ones_like(viol, dtype=bool))):
        tr_m, te_m = sel & ~te, sel & te
        if tr_m.sum() < 50 or te_m.sum() < 20 or len(np.unique(lab_kind[tr_m])) < 2:
            res[name] = None; continue
        X, y = H[tr_m], torch.tensor(lab_kind[tr_m])
        mu, sd = X.mean(0), X.std(0) + 1e-4
        clf = torch.nn.Linear(X.shape[1], len(kinds))
        opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-3)
        Xn = (X - mu) / sd
        for _ in range(300):
            opt.zero_grad(); torch.nn.functional.cross_entropy(clf(Xn), y).backward(); opt.step()
        pred = clf((H[te_m] - mu) / sd).argmax(-1).numpy()
        res[name] = {"acc": float((pred == lab_kind[te_m]).mean()), "n_test": int(te_m.sum()),
                     "class_counts": np.bincount(lab_kind[te_m], minlength=len(kinds)).tolist()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_rif")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    et = tok.convert_tokens_to_ids("</think>")
    end_think = et if isinstance(et, int) and et != tok.unk_token_id else -1
    arms = {}
    for p in sorted(glob.glob(os.path.join(a.root, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        rows = [json.loads(l) for l in open(p)]
        arms[name] = (rows, analyse(rows, tok, end_think))
    order = ["base"] + sorted((x for x in arms if x != "base"), key=lambda s: int(s[2:]))
    kinds = sorted({p["kind"] for p in arms["base"][1]})
    out = ["# Periodic re-encoding of instructions inside reasoning traces", "",
           "| arm | " + " | ".join(f"{kd} trace" for kd in kinds) + " | mean trace compliance | answer compliance | GSM8K acc | finished |",
           "|---|" + "---|" * (len(kinds) + 4)]
    S = {}
    for arm in order:
        per = arms[arm][1]
        comp = mean_comp(per)
        acc = float(np.mean([p["acc"] for p in per]))
        ansc = [p["ans_ok"] for p in per if p["ans_ok"] is not None]
        S[arm] = (comp, acc)
        out.append(f"| {arm} | " + " | ".join(f"{mean_comp([p for p in per if p['kind'] == kd]):.3f}" for kd in kinds) +
                   f" | {comp:.3f} | {np.mean(ansc) if ansc else float('nan'):.3f} | {acc:.3f} | {np.mean([p['finished'] for p in per]):.2f} |")
    base = arms["base"][1]
    cv = curve(base)
    out += ["", "Base compliance by 64-token window (all instructions): " + " ".join("—" if c is None else f"{c:.2f}" for c in cv)]
    for kd in kinds:
        out.append(f"  {kd}: " + " ".join("—" if c is None else f"{c:.2f}" for c in curve([p for p in base if p["kind"] == kd])))
    vals = [c for c in cv if c is not None]
    w0 = cv[0] if cv[0] is not None else float("nan")
    late = [c for c in cv[8:] if c is not None]
    late_m = float(np.mean(late)) if late else float("nan")
    pre = (w0 >= 0.5) and late and (late_m <= w0 - 0.20)
    half = next((i * W for i, c in enumerate(cv) if c is not None and c <= w0 - (w0 - late_m) / 2), None) if late else None
    cands = [(arm, S[arm][0] - S["base"][0], S["base"][1] - S[arm][1]) for arm in order if arm != "base" and int(arm[2:]) >= 128]
    wins = [c for c in cands if c[1] >= 0.15 and c[2] <= 0.03]
    best = max((x for x in order if x != "base"), key=lambda x: S[x][0] - 2 * max(0, S["base"][1] - S[x][1]), default=None)
    out += ["", f"- decay: window-0 {w0:.3f}, windows 8+ {late_m:.3f} -> {'decays' if pre else 'no qualifying decay'}; "
                f"half-life ≈ {half if half is not None else '—'} tokens; best arm (compliance minus 2x accuracy cost): {best}"]
    for arm, dc, dacc in cands:
        out.append(f"- {arm}: compliance {dc:+.3f}, accuracy cost {dacc:+.3f}")
    sp = os.path.join(a.root, "states_base.pt")
    if os.path.exists(sp):
        pr = probe(sp, arms["base"][0], base, kinds)
        v = pr.get("violating")
        diag = "—" if v is None else ("instruction still held while violating -> CONTROL failure" if v["acc"] >= 0.60 else
                                     "instruction lost -> MEMORY failure" if v["acc"] <= 0.45 else "ambiguous")
        out += ["", f"- probe (instruction identity, chance 0.333): violating windows {v if v is None else round(v['acc'], 3)}"
                    f" (n={v and v['n_test']}), all windows {pr.get('all') and round(pr['all']['acc'], 3)} -> {diag}"]
    v = "MOOT (instruction does not decay in base)" if not pre else ("SURVIVES: " + ", ".join(w[0] for w in wins) if wins else "CLOSED")
    out += ["", "## Pre-registered verdict", "", f"**VERDICT: {v}**"]
    txt = "\n".join(out) + "\n"; open(os.path.join(a.root, "REPORT.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()

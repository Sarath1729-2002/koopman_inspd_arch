"""Stage 2: does course correction stop drift past the trained depth, or turn extra loops into gains?

For every policy, ONE forward with T_max loops scores all depths 1..T_max:
  GSM8K  mean NLL of the gold step-by-step solution given the question (teacher-forced)  [primary]
  ARC-C  zero-shot multiple-choice accuracy (length-normalised log-likelihood)            [secondary]
Policies only act past T_train, so depths <= T_train are identical across policies.

Pre-registered verdict (primary metric, paired bootstrap over GSM8K items):
  SURVIVES iff some Koopman policy at some depth d > T_train has lower NLL than
    (a) STOP  = no extra loops (depth T_train, no correction), CI excluding 0, and
    (b) the best control policy at that same depth, CI excluding 0,
  and its ARC-C accuracy at d is not below STOP by more than 1 point.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .common import load, parser, save_json
from .corrections import CONTROLS, DEFAULT, KOOPMAN, Policies
from .data import arc, arc_items, gsm8k, gsm_nll_items


@torch.no_grad()
def score_sequences(a, tok, model, ctl, prompts, targets):
    """Per-sequence (sum logprob over target tokens, n target tokens) at every depth -> [N, T] each."""
    N, T = len(prompts), a.t_max
    lp, nt = np.zeros((N, T)), np.zeros(N)
    for i in range(0, N, a.batch_size):
        P, Tg = prompts[i: i + a.batch_size], targets[i: i + a.batch_size]
        full = [p + t for p, t in zip(P, Tg)]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=768).to(a.device)
        plen = [len(tok(p).input_ids) for p in P]
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for b, n in enumerate(plen):
            labels[b, :n] = -100
        model(**enc, use_cache=False)
        tgt = labels[:, 1:]
        mask = tgt != -100
        nt[i: i + len(P)] = mask.sum(1).cpu().numpy()
        for d in range(1, T + 1):
            logits = ctl.readout(d)[:, :-1].float()
            ll = -F.cross_entropy(logits.transpose(1, 2), tgt.clamp_min(0), reduction="none")
            lp[i: i + len(P), d - 1] = (ll * mask).sum(1).cpu().numpy()
    return lp, nt


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = parser(__doc__)
    ap.add_argument("--n_gsm", type=int, default=256)
    ap.add_argument("--n_arc", type=int, default=400)
    ap.add_argument("--policies", default=",".join(DEFAULT))
    a = ap.parse_args()
    tok, model, ctl = load(a)
    pol = Policies(torch.load(os.path.join(a.out, "fit.pt"), map_location="cpu"), a.device)
    gsm = gsm_nll_items(gsm8k(a.data_dir, "test"), a.n_gsm, a.seed)
    arcs = arc_items(arc(a.data_dir, "test"), a.n_arc, a.seed)
    ap_, at_ = [], []
    owner = []
    for j, it in enumerate(arcs):
        for c in it["choices"]:
            ap_.append(it["prompt"]); at_.append(c); owner.append(j)
    owner = np.array(owner)
    T, Tt = a.t_max, a.t_train
    res, nll, acc = {}, {}, {}
    for name in a.policies.split(","):
        ctl.correct = pol.make(name)
        lp, nt = score_sequences(a, tok, model, ctl, [p for p, _ in gsm], [t for _, t in gsm])
        nll[name] = -lp / nt[:, None]                                   # [N_gsm, T]
        clp, _ = score_sequences(a, tok, model, ctl, ap_, at_)
        norm = clp / np.array([max(1, len(t)) for t in at_])[:, None]   # length-normalised (per char)
        corr = np.zeros((len(arcs), T))
        for j, it in enumerate(arcs):
            corr[j] = (norm[owner == j].argmax(0) == it["gold"])
        acc[name] = corr
        res[name] = {"gsm_nll_by_depth": nll[name].mean(0).tolist(), "arc_acc_by_depth": corr.mean(0).tolist()}
        print(f"[{name:16s}] GSM NLL d{Tt}={nll[name][:, Tt - 1].mean():.4f} " +
              " ".join(f"d{d}={nll[name][:, d - 1].mean():.4f}" for d in (Tt + 2, Tt + 4, T)) +
              f" | ARC d{Tt}={corr[:, Tt - 1].mean():.3f} " + " ".join(f"d{d}={corr[:, d - 1].mean():.3f}" for d in (Tt + 2, Tt + 4, T)), flush=True)
    ctl.correct = None
    np.savez(os.path.join(a.out, "eval_per_example.npz"), **{f"nll__{k}": v for k, v in nll.items()},
             **{f"arc__{k}": v for k, v in acc.items()})

    # ---- verdict ----
    stop_nll = nll["none"][:, Tt - 1]
    stop_arc = acc["none"][:, Tt - 1].mean()
    controls = [k for k in nll if k.split("@")[0] in CONTROLS or k == "none"]
    koops = [k for k in nll if k.split("@")[0] in KOOPMAN]
    rows, survive, best = [], False, None
    for d in range(Tt + 1, T + 1):
        bc = min(controls, key=lambda k: nll[k][:, d - 1].mean())
        for k in koops:
            v_stop = boot_ci(stop_nll - nll[k][:, d - 1])          # >0 = Koopman better than stopping
            v_ctl = boot_ci(nll[bc][:, d - 1] - nll[k][:, d - 1])  # >0 = Koopman better than best control
            arc_drop = stop_arc - acc[k][:, d - 1].mean()
            ok = v_stop[1] > 0 and v_ctl[1] > 0 and arc_drop <= 0.01
            survive |= ok
            if best is None or v_stop[0] > best[2]:
                best = (k, d, v_stop[0])
            rows.append((d, k, bc, v_stop, v_ctl, arc_drop, ok))
    drift = {k: float(nll[k][:, -1].mean() - stop_nll.mean()) for k in nll}
    out = {"t_train": Tt, "t_max": T, "n_gsm": len(gsm), "n_arc": len(arcs), "policies": res,
           "stop": {"gsm_nll": float(stop_nll.mean()), "arc_acc": float(stop_arc)},
           "drift_at_tmax_vs_stop": drift, "survives": survive, "best_koopman_vs_stop": best}
    save_json(out, a, "eval.json")

    L = [f"# Course correction past the trained depth — {a.model}", "",
         f"Trained depth {Tt}, evaluated to {T}. GSM8K gold-solution NLL (n={len(gsm)}), ARC-C acc (n={len(arcs)}).", "",
         "## GSM8K NLL by depth (lower is better)", "",
         "| policy | " + " | ".join(f"d{d}" for d in range(1, T + 1)) + " |", "|---|" + "---|" * T]
    for k in nll:
        L.append(f"| {k} | " + " | ".join(f"{v:.3f}" for v in nll[k].mean(0)) + " |")
    L += ["", "## ARC-C accuracy by depth", "", "| policy | " + " | ".join(f"d{d}" for d in range(1, T + 1)) + " |", "|---|" + "---|" * T]
    for k in acc:
        L.append(f"| {k} | " + " | ".join(f"{v:.3f}" for v in acc[k].mean(0)) + " |")
    L += ["", f"Drift (NLL at d{T} minus STOP at d{Tt}; >0 = worse): " + ", ".join(f"{k} {v:+.4f}" for k, v in drift.items()), "",
          "## Koopman vs STOP and vs best control (paired bootstrap, NLL improvement, 95% CI)", "",
          "| depth | policy | best control | vs STOP | vs best control | ARC drop vs STOP | passes |", "|---|---|---|---|---|---|---|"]
    for d, k, bc, vs, vc, ad, ok in rows:
        L.append(f"| {d} | {k} | {bc} | {vs[0]:+.4f} [{vs[1]:+.4f},{vs[2]:+.4f}] | {vc[0]:+.4f} [{vc[1]:+.4f},{vc[2]:+.4f}] | {ad:+.3f} | {'yes' if ok else ''} |")
    L += ["", f"**Pre-registered verdict: {'SURVIVES' if survive else 'CLOSED'}**  "
          f"(best Koopman vs STOP: {best[0]} at depth {best[1]}, mean NLL gain {best[2]:+.4f})"]
    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

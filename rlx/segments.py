"""Is the GRPO trajectory straight WITHIN phases? Cosine between displacement segments (energy-weighted over tensors).
Segments from the saved checkpoints: 0->10, 10->20, 20->30, 30->40, 40->60, 60->100, 100->150.
High cosine between consecutive LATE segments = a new, stable direction after the early (format) phase,
i.e. extrapolable if the operator is re-fitted from recent steps (course correction that re-encodes the subspace).
-> <out>/SEGMENTS.md"""
import argparse, os

import torch

STEPS = [0, 10, 20, 30, 40, 60, 100, 150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expandvars("$SCRATCH/ckpt_rlx"))
    ap.add_argument("--out", default="results_rlx")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    D = {t: torch.load(os.path.join(a.ckpt, "deltas", f"step{t}.pt")) for t in STEPS if t > 0}
    names = list(D[STEPS[1]])
    segs = [(STEPS[i], STEPS[i + 1]) for i in range(len(STEPS) - 1)]
    k = len(segs)
    dots = torch.zeros(k, k, dtype=torch.float64)
    CH = 16_000_000
    for n in names:
        P = D[STEPS[1]][n].numel()
        for c0 in range(0, P, CH):
            c1 = min(P, c0 + CH)
            get = lambda t: torch.zeros(c1 - c0, device=dev) if t == 0 else D[t][n].reshape(-1)[c0:c1].to(dev, torch.float32)
            S = torch.stack([get(b) - get(a_) for a_, b in segs])           # [k, chunk]
            dots += (S @ S.T).double().cpu()
    nrm = dots.diagonal().sqrt()
    C = dots / (nrm[:, None] * nrm[None, :])
    lab = [f"{a_}->{b}" for a_, b in segs]
    L = ["# Direction of GRPO weight-trajectory segments (cosine, all tensors pooled)", "",
         "| | " + " | ".join(lab) + " |", "|---|" + "---|" * k]
    for i in range(k):
        L.append(f"| {lab[i]} | " + " | ".join(f"{C[i, j]:.2f}" for j in range(k)) + " |")
    L += ["", "Segment lengths (relative): " + ", ".join(f"{lab[i]} {float(nrm[i] / nrm.max()):.2f}" for i in range(k)), "",
          f"- consecutive cos: " + ", ".join(f"{lab[i]}·{lab[i + 1]} {C[i, i + 1]:.2f}" for i in range(k - 1)),
          f"- early vs late: cos(0->10, 100->150) = {C[0, -1]:.2f}"]
    late = float(C[-2, -1])
    L += ["", "## Reading", "",
          ("LATE PHASE IS STRAIGHT (cos(60->100, 100->150) >= 0.6): the trajectory turns after the early phase, then holds a "
           "new direction -> extrapolation should work from a late window, and course correction must re-fit the subspace."
           if late >= 0.6 else
           "NO STABLE LATE DIRECTION (cos(60->100, 100->150) < 0.6): the trajectory keeps turning at this scale; "
           "weight-space extrapolation (RELEX or Koopman) is not viable here.")]
    txt = "\n".join(L) + "\n"
    os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, "SEGMENTS.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()

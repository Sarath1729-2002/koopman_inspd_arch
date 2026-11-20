"""Why did extrapolation fail? Diagnose the saved GRPO weight trajectory (no training).

Reads <ckpt>/deltas/step{1..T}.pt (prefix) and step{60,100,150}.pt. Per tensor, then energy-weighted overall:
  rank1 / rank3      fraction of prefix delta energy in the top-1 / top-3 trajectory modes (RELEX's premise: high)
  drift ratio rho    ||D_T||^2 / sum_t ||d_t||^2 over prefix increments d_t.  ~1 = random walk (noise-dominated),
                     ~T = straight-line drift. Reported as rho/T in [~1/T, 1].
  noise share        fraction of increment energy that is noise around the mean increment (1 - SNR share)
  future capture@r   fraction of the FUTURE move (D_T* - D_T) lying in the prefix top-r subspace, r = 1, 3, 10:
                     the best any subspace extrapolator could do; low = the run moves into new directions
  direction cos      cos(D_T - D_{T/2}, D_T* - D_T): does the recent trend continue?
  oracle werr@r      ||P_perp (D_T* - D_T)|| / ||D_T* - D_T||: error floor for a perfect in-subspace extrapolator
Batch-size estimate: under drift + noise with noise variance ~ 1/batch, the batch multiplier b needed so that
noise is <= 25% of the prefix displacement energy.   -> <out>/DIAG.md + diag.json"""
import argparse, glob, json, math, os, re, time

import torch


def group(name):
    for k in ("embed", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "norm", "lm_head"):
        if k in name:
            return k
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expandvars("$SCRATCH/ckpt_rlx"))
    ap.add_argument("--t_cut", type=int, default=40)
    ap.add_argument("--targets", default="60,100,150")
    ap.add_argument("--out", default="results_rlx")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, T = a.device, a.t_cut
    t0 = time.time()
    prefix = [torch.load(os.path.join(a.ckpt, "deltas", f"step{t}.pt")) for t in range(1, T + 1)]
    targets = [int(x) for x in a.targets.split(",") if os.path.exists(os.path.join(a.ckpt, "deltas", f"step{int(x)}.pt"))]
    fut = {Ts: torch.load(os.path.join(a.ckpt, "deltas", f"step{Ts}.pt")) for Ts in targets}
    print(f"loaded {T} prefix steps + targets {targets} ({time.time() - t0:.0f}s)", flush=True)
    names = list(prefix[0])
    acc = {}                                    # energy-weighted accumulators: key -> [sum(w*x), sum(w)]
    per_group = {}

    def add(key, x, w, g):
        for d in (acc, per_group.setdefault(g, {})):
            s = d.setdefault(key, [0.0, 0.0]); s[0] += w * x; s[1] += w

    for n in names:
        D = torch.stack([p[n].to(dev, torch.float32).reshape(-1) for p in prefix])          # [T, P]
        if not torch.isfinite(D).all():
            raise SystemExit(f"non-finite deltas in {n}")
        E = float((D[-1] ** 2).sum())
        if E == 0:
            continue
        g = group(n)
        G = (D @ D.T).double()
        ev = torch.linalg.eigvalsh(G + 1e-12 * G.diagonal().mean() * torch.eye(T, dtype=G.dtype, device=G.device)).flip(0).clamp(min=0)
        tot = float(ev.sum())
        add("rank1", float(ev[0]) / tot, E, g); add("rank3", float(ev[:3].sum()) / tot, E, g)
        inc = torch.diff(torch.cat([torch.zeros_like(D[:1]), D]), dim=0)                     # [T, P] increments
        sum_inc = float((inc ** 2).sum())
        rho = E / sum_inc
        add("rho_over_T", rho / T, E, g)
        mu = inc.mean(0)
        s_hat = float(((inc - mu) ** 2).sum()) / (T - 1)                 # per-step noise energy (unbiased)
        m_hat = max(0.0, float((mu ** 2).sum()) - s_hat / T)              # drift energy, noise bias removed
        add("noise_share", s_hat / max(1e-30, s_hat + m_hat), E, g)
        # subspace of the prefix (right singular vectors) for future capture
        evc, U = torch.linalg.eigh(G)
        U = U.flip(1).to(D.dtype); s = evc.flip(0).clamp(min=1e-30).sqrt().to(D.dtype)
        V = (D.T @ U[:, :10]) / s[:10]                                                         # [P, 10]
        V, _ = torch.linalg.qr(V)
        half = D[-1] - D[T // 2 - 1]
        for Ts in targets:
            F = fut[Ts][n].to(dev, torch.float32).reshape(-1) - D[-1]
            fe = float((F ** 2).sum())
            if fe == 0:
                continue
            for r in (1, 3, 10):
                cap = float(((V[:, :r].T @ F) ** 2).sum()) / fe
                add(f"capture{r}@{Ts}", cap, fe, g)
                add(f"oracle_werr{r}@{Ts}", math.sqrt(max(0.0, 1 - cap)), fe, g)
            add(f"dir_cos@{Ts}", float(torch.nn.functional.cosine_similarity(half, F, dim=0)), fe, g)
        del D, inc, G
    mean = lambda d, k: d[k][0] / d[k][1] if k in d and d[k][1] > 0 else float("nan")
    res = {k: mean(acc, k) for k in acc}
    res["per_group"] = {g: {k: mean(d, k) for k in d} for g, d in per_group.items()}
    # batch multiplier: rho ~ (T^2 m + T s) / (T m + T s) with m = |mu|^2, s = noise var per step (s ~ 1/batch).
    # noise share n = s/(m+s)  ->  s/m = n/(1-n). Noise fraction of the T-step displacement: s/(T m + s).
    # Want s'/(T m + s') <= 0.25 with s' = s/b  ->  b >= 3 s / (T m).
    nshare = res["noise_share"]
    s_over_m = nshare / max(1e-6, 1 - nshare)
    disp_noise = s_over_m / (T + s_over_m)
    b_need = max(1.0, 3 * s_over_m / T)
    res.update({"noise_frac_of_displacement": disp_noise, "batch_multiplier_needed": b_need, "T": T})
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "diag.json"), "w"), indent=1)
    f = lambda k: f"{res.get(k, float('nan')):.3f}"
    L = ["# GRPO weight-trajectory diagnosis", "",
         f"Prefix: steps 1..{T}. Energy-weighted over all tensors.", "",
         "| quantity | value | reading |", "|---|---|---|",
         f"| rank-1 energy of prefix | {f('rank1')} | RELEX premise: high (most motion along one direction) |",
         f"| rank-3 energy of prefix | {f('rank3')} | |",
         f"| drift ratio rho/T | {f('rho_over_T')} | 1 = straight line, ~{1 / T:.3f} = pure random walk |",
         f"| noise share of per-step increments | {f('noise_share')} | per-step SGD noise vs consistent drift |",
         f"| noise share of the {T}-step displacement | {res['noise_frac_of_displacement']:.3f} | |",
         f"| batch multiplier for noise <= 25% of displacement | {res['batch_multiplier_needed']:.1f}x | (32 prompts x 8 now) |"]
    for Ts in targets:
        L += [f"| future capture @{Ts}: top-1 / top-3 / top-10 | {f(f'capture1@{Ts}')} / {f(f'capture3@{Ts}')} / {f(f'capture10@{Ts}')} | share of the real future move inside the prefix subspace |",
              f"| oracle in-subspace werr @{Ts} (r=3) | {f(f'oracle_werr3@{Ts}')} | floor for ANY rank-3 extrapolator (ours got 1.2-2.4) |",
              f"| direction cos(recent trend, future move) @{Ts} | {f(f'dir_cos@{Ts}')} | |"]
    L += ["", "## Per tensor type (rank-1, rho/T, noise share, capture3 @ last target)", "",
          "| type | rank1 | rho/T | noise | capture3 |", "|---|---|---|---|---|"]
    lt = targets[-1] if targets else None
    for g, d in sorted(res["per_group"].items()):
        L.append(f"| {g} | {d.get('rank1', float('nan')):.3f} | {d.get('rho_over_T', float('nan')):.3f} | "
                 f"{d.get('noise_share', float('nan')):.3f} | {d.get(f'capture3@{lt}', float('nan')):.3f} |")
    # automatic reading
    cap = res.get(f"capture3@{lt}", float("nan"))
    if res["noise_frac_of_displacement"] > 0.5:
        why = (f"NOISE-DOMINATED: per-step gradient noise is most of the motion; a {res['batch_multiplier_needed']:.0f}x larger "
               f"batch (or averaging) is needed before any extrapolator can see the trend.")
    elif cap < 0.3:
        why = "NEW DIRECTIONS: the future move mostly lies outside the prefix subspace; subspace extrapolation cannot work at this horizon."
    else:
        why = "EXTRAPOLABLE IN PRINCIPLE: the future move lies in the prefix subspace and the trend is not noise-dominated; the failure is in the dynamics model (fixable)."
    L += ["", "## Reading", "", why]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "DIAG.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()

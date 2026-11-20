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

    CH = 8_000_000                                   # columns per chunk (40 x 8M fp32 = 1.3 GB on GPU)
    for n in names:
        P = prefix[0][n].numel()
        chunks = [(i, min(P, i + CH)) for i in range(0, P, CH)]
        col = lambda t_list, a_, b_: torch.stack([x[n].reshape(-1)[a_:b_].to(dev, torch.float32) for x in t_list])
        # pass 1: Gram matrix and increment statistics, accumulated over column chunks
        G = torch.zeros(T, T, dtype=torch.float64, device=dev)
        E = sum_inc = mu2 = dev2 = 0.0
        for a_, b_ in chunks:
            D = col(prefix, a_, b_)
            if not torch.isfinite(D).all():
                raise SystemExit(f"non-finite deltas in {n}")
            G += (D @ D.T).double()
            E += float((D[-1] ** 2).sum())
            inc = torch.diff(torch.cat([torch.zeros_like(D[:1]), D]), dim=0)
            mu = inc.mean(0)
            sum_inc += float((inc ** 2).sum()); mu2 += float((mu ** 2).sum()); dev2 += float(((inc - mu) ** 2).sum())
            del D, inc, mu
        if E == 0:
            continue
        g = group(n)
        Gj = G + 1e-12 * G.diagonal().mean() * torch.eye(T, dtype=G.dtype, device=dev)
        evc, U = torch.linalg.eigh(Gj)
        ev = evc.flip(0).clamp(min=0); U = U.flip(1)
        tot = float(ev.sum())
        add("rank1", float(ev[0]) / tot, E, g); add("rank3", float(ev[:3].sum()) / tot, E, g)
        add("rho_over_T", (E / sum_inc) / T, E, g)
        s_hat = dev2 / (T - 1); m_hat = max(0.0, mu2 - s_hat / T)
        add("noise_share", s_hat / max(1e-30, s_hat + m_hat), E, g)
        r_eff = int(min(10, (ev > ev[0] * 1e-10).sum()))
        Us = (U[:, :r_eff] / ev[:r_eff].sqrt()).float()          # V = D^T Us is orthonormal [P, r_eff]
        # pass 2: project the future moves on the prefix subspace, chunk by chunk
        for Ts in targets:
            proj = torch.zeros(r_eff, dtype=torch.float64, device=dev)
            fe = dot = hn = 0.0
            for a_, b_ in chunks:
                D = col(prefix, a_, b_)
                F = fut[Ts][n].reshape(-1)[a_:b_].to(dev, torch.float32) - D[-1]
                half = D[-1] - D[T // 2 - 1]
                proj += ((D.T @ Us).T @ F).double()
                fe += float((F ** 2).sum()); dot += float((half * F).sum()); hn += float((half ** 2).sum())
                del D, F, half
            if fe == 0:
                continue
            for r in (1, 3, 10):
                cap = min(1.0, float((proj[:min(r, r_eff)] ** 2).sum()) / fe)
                add(f"capture{r}@{Ts}", cap, fe, g)
                add(f"oracle_werr{r}@{Ts}", math.sqrt(max(0.0, 1 - cap)), fe, g)
            add(f"dir_cos@{Ts}", dot / max(1e-30, math.sqrt(hn * fe)), fe, g)
        del G
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

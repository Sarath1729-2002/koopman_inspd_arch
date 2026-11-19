"""Reference GRPO run. Saves, relative to the base weights:
   deltas/step{t}.pt for t = 1..T_cut (bf16)   -> the observed prefix the extrapolators fit on
   deltas/step{t}.pt for t in --keep (e.g. 60,100,150) -> ground truth for weight-space error
   opt_Tcut.pt (optimizer state at T_cut) and PREFIX_DONE flag (other GPUs start then)
and evaluates GSM8K test accuracy at 0, T_cut, and every --keep step.  -> <out>/ref.json"""
import argparse, json, os, random, time

import torch

from .common import evaluate, grpo_step, load_gsm, load_model, params, snapshot_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--t_cut", type=int, default=40)
    ap.add_argument("--keep", default="60,100,150")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--train", default="data/gsm8k_train.jsonl")
    ap.add_argument("--test", default="data/gsm8k_test.jsonl")
    ap.add_argument("--n_test", type=int, default=1319)
    ap.add_argument("--ckpt", default="ckpt_rlx")
    ap.add_argument("--out", default="results_rlx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = a.device
    tok, model = load_model(a.model, dev)
    base = {n: p.detach().clone() for n, p in params(model).items()}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    train, test = load_gsm(a.train), load_gsm(a.test, a.n_test)
    keep = sorted(int(x) for x in a.keep.split(","))
    os.makedirs(os.path.join(a.ckpt, "deltas"), exist_ok=True); os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"model": a.model, "t_cut": a.t_cut, "steps": a.steps, "acc": {}, "reward": []}
    res["acc"]["0"] = evaluate(model, tok, test, dev)
    print(f"[ref] step 0 acc {res['acc']['0']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    for t in range(1, a.steps + 1):
        info = grpo_step(model, tok, opt, rng.sample(train, a.prompts), a.samples, a.max_new, a.temp, dev)
        res["reward"].append(info["reward"]); res.setdefault("steps_info", []).append(info)
        skips = sum(not x["updated"] for x in res["steps_info"])
        if not info["updated"]:
            print(f"[ref] step {t}: NON-FINITE loss/grad, update skipped ({skips} so far)", flush=True)
        if skips > 5:
            raise RuntimeError("more than 5 non-finite updates: training is numerically broken")
        if not all(torch.isfinite(p).all() for p in params(model).values()):
            raise RuntimeError(f"non-finite weights at step {t}")
        if t <= a.t_cut or t in keep:
            torch.save(snapshot_delta(model, base), os.path.join(a.ckpt, "deltas", f"step{t}.pt"))
        if t == a.t_cut:
            r0 = sum(res["reward"][:5]) / 5; rT = sum(res["reward"][-10:]) / 10
            if rT < 0.5 * r0:
                raise RuntimeError(f"reward collapsed by step {t} ({r0:.3f} -> {rT:.3f}); aborting before extrapolation")
            torch.save(opt.state_dict(), os.path.join(a.ckpt, "opt_Tcut.pt"))
            res["acc"][str(t)] = evaluate(model, tok, test, dev)
            open(os.path.join(a.ckpt, "PREFIX_DONE"), "w").write("ok")
            print(f"[ref] step {t} acc {res['acc'][str(t)]:.3f} -- prefix saved", flush=True)
        if t in keep:
            res["acc"][str(t)] = evaluate(model, tok, test, dev)
            print(f"[ref] step {t} acc {res['acc'][str(t)]:.3f}", flush=True)
            json.dump(res, open(os.path.join(a.out, "ref.json"), "w"), indent=1)
        if t % 5 == 0 or t <= 3:
            last = res["steps_info"][-5:]
            print(f"[ref] step {t} reward(last5) {sum(x['reward'] for x in last) / len(last):.3f} "
                  f"grad_norm {info['grad_norm']:.3f} loss {info['loss']:+.4f} len {info['mean_len']:.0f} "
                  f"trunc {info['trunc_frac']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(a.out, "ref.json"), "w"), indent=1)
    open(os.path.join(a.ckpt, "REF_DONE"), "w").write("ok")


if __name__ == "__main__":
    import sys, traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        ck = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else "ckpt_rlx"
        os.makedirs(ck, exist_ok=True)
        open(os.path.join(ck, "REF_FAILED"), "w").write("failed")      # other GPUs stop waiting
        raise

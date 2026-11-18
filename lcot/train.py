"""Train + evaluate one arm on GSM8k-Aug (GPT-2).
  stage 0   : CoT SFT (question \n steps \n ### answer)  -> --do_stage0 trains it once and saves; arms load it
  stage s>=1: first s steps replaced by s*c continuous thoughts; final stage S removes all language steps (Coconut)
Each stage: fresh AdamW (Coconut), then test accuracy + latent diagnostics. Writes <out>/<arm>.json"""
import argparse, json, math, os, random, re, time

import torch
import torch.nn.functional as F

from .model import greedy, latent_rollout, setup_tokenizer_model, suffix_forward


def load(path, n=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:n] if n else rows


def encode(tok, rows):
    qs = tok([r["question"] + "\n" for r in rows])["input_ids"]
    an = tok(["### " + r["answer"] for r in rows])["input_ids"]
    flat = [s + "\n" for r in rows for s in r["steps"]]
    fs = tok(flat)["input_ids"]
    out, i = [], 0
    for r, q, a in zip(rows, qs, an):
        st = fs[i:i + len(r["steps"])]; i += len(r["steps"])
        out.append({"q": q[-160:], "steps": st, "a": a + [tok.eos_token_id], "gold": r["answer"]})
    return out


def pad(seqs, val, left=False):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), val, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        if left:
            ids[i, L - len(s):] = torch.tensor(s); m[i, L - len(s):] = 1
        else:
            ids[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return ids, m


def parse(txt):
    t = txt.split("###")[-1] if "###" in txt else ""
    m = re.search(r"-?\d[\d,]*\.?\d*", t)
    return m.group(0).replace(",", "").rstrip(".") if m else None


def correct(p, g):
    try:
        return p is not None and abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def batches(data, bs, rng):
    idx = list(range(len(data))); rng.shuffle(idx)
    for i in range(0, len(idx) - bs + 1, bs):
        yield [data[j] for j in idx[i:i + bs]]


def removed(ex, s, S):
    return len(ex["steps"]) if s >= S else min(s, len(ex["steps"]))


def run_stage(model, tok, data, s, a, dev, log):
    S, c = a.stages, a.c
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    rng = random.Random(a.seed * 100 + s)
    epochs = a.stage0_epochs if s == 0 else a.epochs_per_stage
    bot, eot, pad_id = (tok.convert_tokens_to_ids(x) for x in ("<|bot|>", "<|eot|>")), None, tok.pad_token_id
    bot, eot = tok.convert_tokens_to_ids("<|bot|>"), tok.convert_tokens_to_ids("<|eot|>")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    t0, step = time.time(), 0
    model.train()
    for ep in range(epochs):
        for B in batches(data, a.batch, rng):
            if s == 0:
                seqs = [ex["q"] + sum(ex["steps"], []) + ex["a"] for ex in B]
                ids, m = pad(seqs, pad_id)
                lab = ids.clone(); lab[m == 0] = -100
                for i, ex in enumerate(B):
                    lab[i, :len(ex["q"])] = -100
                ids, m, lab = ids.to(dev), m.to(dev), lab.to(dev)
                with amp:
                    lg = model(input_ids=ids, attention_mask=m).logits
                loss = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
            else:
                n = s * c
                pre, pm = pad([ex["q"] + [bot] for ex in B], pad_id, left=True)
                suf, sm = pad([[eot] + sum(ex["steps"][removed(ex, s, S):], []) + ex["a"] for ex in B], pad_id)
                pre, pm, suf, sm = pre.to(dev), pm.to(dev), suf.to(dev), sm.to(dev)
                with amp:
                    cache, mask, nxt, _ = latent_rollout(model, pre, pm, n, a.snap, a.k, a.tau)
                    lg, *_ = suffix_forward(model, cache, mask, nxt, suf, sm)
                lab = suf.clone(); lab[sm == 0] = -100
                loss = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            if step % 200 == 0 or step == 1:
                print(f"[{a.arm}] stage {s} ep {ep} step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(loss.item()):
                print("non-finite loss"); return
    log.append({"stage": s, "train_minutes": (time.time() - t0) / 60})


@torch.no_grad()
def evaluate(model, tok, test, s, a, dev, n_override=None):
    model.eval()
    bot, eot, eos = tok.convert_tokens_to_ids("<|bot|>"), tok.convert_tokens_to_ids("<|eot|>"), tok.eos_token_id
    n = n_override if n_override is not None else s * a.c
    ok, homog, vcos, cnt = 0, [], [], 0
    W = F.normalize(model.get_input_embeddings().weight.float(), dim=-1)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    for i in range(0, len(test), a.eval_bs):
        B = test[i:i + a.eval_bs]
        with amp:
            if s == 0:
                ids, m = pad([ex["q"] for ex in B], tok.pad_token_id, left=True)
                ids, m = ids.to(dev), m.to(dev)
                tr = model.transformer if hasattr(model, "transformer") else model.model
                pos = (m.cumsum(-1) - 1).clamp(min=0)
                o = tr(input_ids=ids[:, :-1], attention_mask=m[:, :-1], position_ids=pos[:, :-1], use_cache=True)
                gen = greedy(model, o.past_key_values, m[:, :-1], pos[:, -1:], ids[:, -1:], 200, eos)
            else:
                pre, pm = pad([ex["q"] + [bot] for ex in B], tok.pad_token_id, left=True)
                cache, mask, nxt, th = latent_rollout(model, pre.to(dev), pm.to(dev), n, a.snap, a.k, a.tau)
                first = torch.full((len(B), 1), eot, dtype=torch.long, device=dev)
                gen = greedy(model, cache, mask, nxt, first, 160 if s < a.stages else 24, eos)
                if n >= 2 and i < 512:
                    T = F.normalize(torch.stack(th, 1).float(), dim=-1)            # [B, n, d]
                    G = T @ T.transpose(1, 2)
                    homog.append(((G.sum((1, 2)) - n) / (n * (n - 1))).mean().item())
                    vcos.append((T @ W.T).max(-1).values.mean().item())
        for ex, g in zip(B, gen):
            txt = tok.decode(g, skip_special_tokens=False).split(tok.eos_token)[0]
            ok += correct(parse(txt), ex["gold"]); cnt += 1
    model.train()
    return {"acc": ok / cnt, "n_lat": n, "thought_cos": sum(homog) / len(homog) if homog else None,
            "vocab_cos": sum(vcos) / len(vcos) if vcos else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--snap", default="none", choices=["none", "soft", "hard"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--data", default="data_lcot")
    ap.add_argument("--do_stage0", action="store_true", help="train the CoT stage and save it (run once)")
    ap.add_argument("--stage0_ckpt", default="ckpt_lcot/stage0.pt")
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--c", type=int, default=2)
    ap.add_argument("--stage0_epochs", type=int, default=2)
    ap.add_argument("--epochs_per_stage", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_lcot")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = a.device
    tok, model = setup_tokenizer_model(a.model, dev)
    t0 = time.time()
    train = encode(tok, load(os.path.join(a.data, "train.jsonl"), a.n_train))
    test = encode(tok, load(os.path.join(a.data, "test.jsonl"), a.n_test))
    print(f"[{a.arm}] train {len(train)} test {len(test)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    res = {"arm": a.arm, "snap": a.snap, "k": a.k, "c": a.c, "stages": a.stages, "evals": [], "log": []}
    if a.do_stage0:
        run_stage(model, tok, train, 0, a, dev, res["log"])
        os.makedirs(os.path.dirname(a.stage0_ckpt) or ".", exist_ok=True)
        torch.save(model.state_dict(), a.stage0_ckpt)
        e = evaluate(model, tok, test, 0, a, dev); e["stage"] = 0
        res["evals"].append(e)
        print(f"[{a.arm}] CoT (stage 0) acc {e['acc']:.3f}", flush=True)
        json.dump(res, open(os.path.join(a.out, "cot.json"), "w"), indent=1)
        return
    model.load_state_dict(torch.load(a.stage0_ckpt, map_location=dev))
    for s in range(1, a.stages + 1):
        run_stage(model, tok, train, s, a, dev, res["log"])
        e = evaluate(model, tok, test, s, a, dev); e["stage"] = s
        res["evals"].append(e)
        print(f"[{a.arm}] stage {s} ({e['n_lat']} thoughts) acc {e['acc']:.3f} thought_cos {e['thought_cos']} vocab_cos {e['vocab_cos']}", flush=True)
        json.dump(res, open(os.path.join(a.out, f"{a.arm}.json"), "w"), indent=1)
    e = evaluate(model, tok, test, a.stages, a, dev, n_override=(a.stages + 2) * a.c); e["stage"] = "extra"
    res["evals"].append(e)
    print(f"[{a.arm}] extra thoughts ({e['n_lat']}) acc {e['acc']:.3f}", flush=True)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(a.out, f"{a.arm}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

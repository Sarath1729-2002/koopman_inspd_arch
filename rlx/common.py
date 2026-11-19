"""Shared: model/data loading, GSM8K reward + eval, one on-policy GRPO step, delta helpers."""
import json, math, os, random, re

import torch
import torch.nn.functional as F

SYS = "Solve the problem step by step. Put the final answer in \\boxed{}."


def load_model(name, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev)
    return tok, model


def load_gsm(path, n=None):
    rows = []
    for l in open(path):
        r = json.loads(l)
        q, a = r.get("question"), str(r.get("answer", ""))
        if q and a:
            rows.append((q, a.split("####")[-1].strip().replace(",", "")))
    return rows[:n] if n else rows


def prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    if tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return SYS + "\nQ: " + q + "\nA:"


def extract(txt):
    m = re.findall(r"\\boxed\{([^{}]*)\}", txt)
    s = m[-1] if m else txt
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def reward(txt, gold):
    if os.environ.get("RLX_FAKE_REWARD"):            # smoke test only
        return float(random.random() < 0.4)
    p = extract(txt)
    try:
        return float(p is not None and abs(float(p) - float(gold)) < 1e-6)
    except ValueError:
        return 0.0


@torch.no_grad()
def generate(model, tok, prompts, max_new, temp, dev, bs=256):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            g = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new, do_sample=temp > 0, temperature=temp if temp > 0 else None,
                               top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        P = enc["input_ids"].shape[1]
        outs += [(enc["input_ids"][j], g[j, P:]) for j in range(len(g))]
    return outs


@torch.no_grad()
def evaluate(model, tok, items, dev, max_new=400):
    model.eval()
    res = generate(model, tok, [prompt(tok, q) for q, _ in items], max_new, 0.0, dev)
    acc = sum(reward(tok.decode(o, skip_special_tokens=True), g) for (_, o), (_, g) in zip(res, items)) / len(items)
    model.train()
    return acc


def grpo_step(model, tok, opt, items, n_samples, max_new, temp, dev, micro=32):
    """On-policy policy gradient with group-normalised advantages (one update per batch, no KL; Dr.GRPO/DAPO style)."""
    prompts = [prompt(tok, q) for q, _ in items for _ in range(n_samples)]
    golds = [g for _, g in items for _ in range(n_samples)]
    model.eval()
    gens = generate(model, tok, prompts, max_new, temp, dev)
    model.train()
    R = torch.tensor([reward(tok.decode(o, skip_special_tokens=True), g) for (_, o), g in zip(gens, golds)])
    Rg = R.view(-1, n_samples)
    A = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)).view(-1)
    keep = [i for i in range(len(gens)) if abs(A[i]) > 1e-6]
    opt.zero_grad(set_to_none=True)
    ntok = sum(int((gens[i][1] != tok.pad_token_id).sum()) for i in keep) or 1
    for s in range(0, len(keep), micro):
        idx = keep[s:s + micro]
        seqs = [torch.cat([gens[i][0][gens[i][0] != tok.pad_token_id], gens[i][1][gens[i][1] != tok.pad_token_id]]) for i in idx]
        plen = [int((gens[i][0] != tok.pad_token_id).sum()) for i in idx]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long, device=dev)
        msk = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = x; msk[j, :len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            lg = model(input_ids=ids, attention_mask=msk).logits[:, :-1].float()
        lp = -F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").view(len(seqs), -1)
        comp = torch.zeros_like(lp)
        for j, (x, p) in enumerate(zip(seqs, plen)):
            comp[j, p - 1:len(x) - 1] = 1
        adv = A[idx].to(dev)[:, None]
        loss = -(adv * lp * comp).sum() / ntok
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(R.mean())


def params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def snapshot_delta(model, base):
    return {n: (p.detach() - base[n]).to(torch.bfloat16).cpu() for n, p in params(model).items()}


def set_weights(model, base, delta):
    with torch.no_grad():
        for n, p in params(model).items():
            p.copy_(base[n] + delta[n].to(p.device, p.dtype))

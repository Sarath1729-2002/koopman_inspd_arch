"""GSM8K and ARC-Challenge from local JSONL (exported on the login node; compute nodes are offline)."""
import json
import os
import random
import re


def _jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def gsm8k(data_dir, split):
    return _jsonl(os.path.join(data_dir, f"gsm8k_{split}.jsonl"))


def arc(data_dir, split="test"):
    return _jsonl(os.path.join(data_dir, f"arc_challenge_{split}.jsonl"))


def gsm_answer(text):
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm_fewshot(train, k, seed=0):
    rng = random.Random(seed)
    shots = rng.sample(train, k)
    return "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)


def gsm_nll_items(test, n, seed=0):
    """(prompt, target) pairs: target = gold step-by-step solution (teacher forced)."""
    rng = random.Random(seed)
    items = rng.sample(test, min(n, len(test)))
    return [(f"Question: {x['question']}\nAnswer:", " " + x["answer"]) for x in items]


def arc_items(rows, n, seed=0):
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in rows:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        out.append({"prompt": f"Question: {r['question']}\nAnswer:", "choices": [" " + t for t in texts],
                    "gold": labels.index(r["answerKey"])})
    return out

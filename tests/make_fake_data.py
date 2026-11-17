"""Tiny GSM8K / ARC-C look-alikes in the exported JSONL format (smoke tests only)."""
import json, os, random, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True); r = random.Random(0)
def gsm(n):
    rows = []
    for _ in range(n):
        x, y = r.randint(1, 50), r.randint(1, 50)
        rows.append({"question": f"Tom has {x} apples and gets {y} more. How many?", "answer": f"{x}+{y}={x+y}\n#### {x+y}"})
    return rows
for s, n in (("train", 60), ("test", 40)):
    with open(f"{out}/gsm8k_{s}.jsonl", "w") as f: [f.write(json.dumps(x) + "\n") for x in gsm(n)]
with open(f"{out}/arc_challenge_test.jsonl", "w") as f:
    for i in range(30):
        f.write(json.dumps({"question": f"Which is largest? {i}", "choices": {"label": ["A","B","C","D"], "text": ["one","two","three","four"]}, "answerKey": "D"}) + "\n")
print("ok", out)

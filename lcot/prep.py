"""LOGIN NODE (internet): download GSM8k-Aug + GPT-2 into $HF_HOME and write data_lcot/{train,val,test}.jsonl
(fields: question, steps [list of '<<a*b=c>>'], answer).   python -m lcot.prep"""
import glob, json, os

from huggingface_hub import snapshot_download

OUT = "data_lcot"
os.makedirs(OUT, exist_ok=True)
snapshot_download("openai-community/gpt2", allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
print("gpt2 cached")


def rows_from(root):
    import pyarrow.parquet as pq
    splits = {}
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        name = os.path.basename(f).lower()
        sp = "train" if "train" in name else "val" if ("valid" in name or "val" in name or "dev" in name) else "test" if "test" in name else None
        if sp is None:
            continue
        if f.endswith(".parquet"):
            rows = pq.read_table(f).to_pylist()
        elif f.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(f) if l.strip()]
        elif f.endswith(".json"):
            d = json.load(open(f)); rows = d if isinstance(d, list) else []
        else:
            continue
        splits.setdefault(sp, []).extend(rows)
    return splits


def norm(r):
    q, a = r.get("question"), str(r.get("answer", "")).replace(",", "").strip()
    st = r.get("steps")
    if st is None and r.get("cot") is not None:
        st = [s if s.endswith(">>") else s + ">>" for s in str(r["cot"]).replace(">> <<", ">>\n<<").split("\n") if s.strip()]
    if not q or not a or not st:
        return None
    return {"question": q.strip(), "steps": [s.strip() for s in st], "answer": a}


for repo in ["whynlp/gsm8k-aug", "cs-giung/clean-gsm8k-aug", "zen-E/GSM8k-Aug"]:
    try:
        root = snapshot_download(repo, repo_type="dataset")
        S = rows_from(root)
        if "train" in S and "test" in S:
            break
        print(f"{repo}: splits found {list(S)}; trying next")
    except Exception as e:
        print(f"{repo}: {e}")
else:
    raise SystemExit("no GSM8k-Aug source worked")
if "val" not in S:
    S["val"] = S["train"][-500:]; S["train"] = S["train"][:-500]
for sp, rows in S.items():
    rows = [x for x in map(norm, rows) if x]
    with open(os.path.join(OUT, f"{sp}.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"{repo} {sp}: {len(rows)} rows")
print("example:", json.dumps(rows[0]))

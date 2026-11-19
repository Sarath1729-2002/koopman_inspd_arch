"""LOGIN NODE: cache Qwen2.5-0.5B-Instruct and make sure data/gsm8k_{train,test}.jsonl exist.   python -m rlx.prep"""
import glob, json, os
from huggingface_hub import snapshot_download

print("model:", snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"]))
need = [s for s in ("train", "test") if not os.path.exists(f"data/gsm8k_{s}.jsonl")]
if need:
    import pyarrow.parquet as pq
    r = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/*"])
    os.makedirs("data", exist_ok=True)
    for s in need:
        rows = pq.read_table(glob.glob(os.path.join(r, "main", f"{s}-*"))[0]).to_pylist()
        with open(f"data/gsm8k_{s}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps({"question": x["question"], "answer": x["answer"]}) + "\n")
for s in ("train", "test"):
    r0 = json.loads(open(f"data/gsm8k_{s}.jsonl").readline())
    print(s, sum(1 for _ in open(f"data/gsm8k_{s}.jsonl")), "rows; keys", list(r0), "; answer has ####:", "####" in str(r0.get("answer")))

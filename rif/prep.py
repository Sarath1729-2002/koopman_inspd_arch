"""LOGIN NODE: make sure the reasoning model and GSM8K test are available.   python -m rif.prep [--model ...]"""
import argparse, json, os
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"); a = ap.parse_args()
p = snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print("model ready:", p)
if not os.path.exists("data/gsm8k_test.jsonl"):
    import pyarrow.parquet as pq, glob
    r = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/test-*"])
    rows = pq.read_table(glob.glob(os.path.join(r, "main", "test-*"))[0]).to_pylist()
    os.makedirs("data", exist_ok=True)
    with open("data/gsm8k_test.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps({"question": x["question"], "answer": x["answer"]}) + "\n")
print("gsm8k:", sum(1 for _ in open("data/gsm8k_test.jsonl")), "rows")

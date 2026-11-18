"""LOGIN NODE (needs internet). wikitext-103-raw -> data/{train,eval}.bin (raw bytes, uint8).
   python scripts/prep_data.py            (HF_HOME should point at $SCRATCH/hf)
   python scripts/prep_data.py --text some.txt   (fallback: any big text file, last 2 MB used for eval)"""
import argparse, glob, os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--text", default=None); ap.add_argument("--out", default="data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.text:
    b = open(a.text, "rb").read()
    tr, ev = b[:-2_000_000], b[-2_000_000:]
else:
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns="wikitext-103-raw-v1/*")
    read = lambda pat: "".join(t for f in sorted(glob.glob(os.path.join(root, "wikitext-103-raw-v1", pat)))
                               for t in pq.read_table(f).column("text").to_pylist()).encode("utf-8")
    tr, ev = read("train-*.parquet"), read("validation-*.parquet") + read("test-*.parquet")
np.frombuffer(tr, dtype=np.uint8).tofile(os.path.join(a.out, "train.bin"))
np.frombuffer(ev, dtype=np.uint8).tofile(os.path.join(a.out, "eval.bin"))
print(f"train {len(tr) / 1e6:.1f} MB, eval {len(ev) / 1e6:.2f} MB -> {a.out}/")

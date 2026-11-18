"""wikitext-103-raw -> data/{train,eval}.bin (raw bytes, uint8), streamed row-group by row-group (low memory).
Download needs internet (login node); once the files are in $HF_HOME the conversion also runs offline
(the slurm job calls this with HF_HUB_OFFLINE=1 if data/train.bin is missing).
   python scripts/prep_data.py
   python scripts/prep_data.py --text some.txt     (fallback: any big text file; last 2 MB used for eval)"""
import argparse, glob, os

ap = argparse.ArgumentParser(); ap.add_argument("--text", default=None); ap.add_argument("--out", default="data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tr_p, ev_p = os.path.join(a.out, "train.bin"), os.path.join(a.out, "eval.bin")


def stream(files, dst):
    import pyarrow.parquet as pq
    n = 0
    with open(dst + ".tmp", "wb") as f:
        for fn in files:
            for batch in pq.ParquetFile(fn).iter_batches(batch_size=20000, columns=["text"]):
                b = "".join(batch.column(0).to_pylist()).encode("utf-8")
                f.write(b); n += len(b)
    os.replace(dst + ".tmp", dst)
    return n


if a.text:
    size = os.path.getsize(a.text)
    with open(a.text, "rb") as src, open(tr_p, "wb") as t, open(ev_p, "wb") as e:
        left = size - 2_000_000
        while left > 0:
            c = src.read(min(1 << 24, left)); t.write(c); left -= len(c)
        e.write(src.read())
    print(f"train {(size - 2e6) / 1e6:.1f} MB, eval 2.0 MB -> {a.out}/")
else:
    from huggingface_hub import snapshot_download
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns="wikitext-103-raw-v1/*")
    d = os.path.join(root, "wikitext-103-raw-v1")
    files = lambda pat: sorted(glob.glob(os.path.join(d, pat)))
    assert files("train-*.parquet"), f"no parquet files under {d}"
    nt = stream(files("train-*.parquet"), tr_p)
    ne = stream(files("validation-*.parquet") + files("test-*.parquet"), ev_p)
    print(f"train {nt / 1e6:.1f} MB, eval {ne / 1e6:.2f} MB -> {a.out}/")

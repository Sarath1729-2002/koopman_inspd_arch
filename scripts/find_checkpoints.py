"""Search the Hugging Face Hub for pretrained TRM ARC-AGI-1 checkpoints (run on a LOGIN node).

  python scripts/find_checkpoints.py                  # list candidates
  python scripts/find_checkpoints.py --download <repo_id> [--dest $SCRATCH/trm_ckpts]

A checkpoint only works with the exact dataset build it was trained on (TRM learns one embedding
per augmented puzzle id). The official build command (seed 42, num_aug 1000) is deterministic, so a
checkpoint trained with the README's ARC-AGI-1 command matches `scripts/setup_tamia.sh`'s build.
rollout.py warns if accuracy comes out near zero, which is the symptom of a mismatch.
"""
import argparse
import os

from huggingface_hub import HfApi, snapshot_download

QUERIES = ["TinyRecursiveModels", "tiny recursive", "trm arc", "trm-arc", "trm_arc", "TRM ARC-AGI", "tiny-recursive"]
AUTHORS = ["arcprize", "SamsungSAILMontreal", "alexiajm", "AlexiaJM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", default=None)
    ap.add_argument("--dest", default=os.path.join(os.environ.get("SCRATCH", "."), "trm_ckpts"))
    a = ap.parse_args()
    api = HfApi()
    if a.download:
        path = snapshot_download(a.download, local_dir=os.path.join(a.dest, a.download.replace("/", "__")))
        print("downloaded to", path)
        for root, _, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                print(f"  {os.path.getsize(p) / 1e6:9.1f} MB  {p}")
        print("\nPass the weights file (e.g. .../step_XXXXX) as --ckpt. If an all_config.yaml sits next to it, "
              "its arch settings are used automatically.")
        return
    seen = {}
    for q in QUERIES:
        for m in api.list_models(search=q, limit=50):
            seen[m.id] = m
    for au in AUTHORS:
        try:
            for m in api.list_models(author=au, limit=100):
                seen[m.id] = m
        except Exception:
            pass
    rows = []
    for mid, m in seen.items():
        low = mid.lower()
        if not any(s in low for s in ("trm", "recursive", "hrm")):
            continue
        try:
            files = [s.rfilename for s in api.model_info(mid, files_metadata=False).siblings]
        except Exception:
            files = []
        rows.append((mid, getattr(m, "downloads", 0) or 0, files))
    rows.sort(key=lambda r: -r[1])
    for mid, dl, files in rows:
        tag = " <-- ARC?" if any("arc" in f.lower() for f in files) or "arc" in mid.lower() else ""
        print(f"{mid:60s} downloads={dl}{tag}")
        for f in files[:12]:
            print("     ", f)
    if not rows:
        print("no candidates found — train one with slurm/train_trm_arc1.sh")


if __name__ == "__main__":
    main()

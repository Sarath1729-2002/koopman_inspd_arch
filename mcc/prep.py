"""LOGIN NODE: cache the pretrained Mamba checkpoints (HF format).   python -m mcc.prep"""
from huggingface_hub import snapshot_download
for s in ["130m", "370m", "790m", "1.4b", "2.8b"]:
    p = snapshot_download(f"state-spaces/mamba-{s}-hf", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print("ok", s, p)

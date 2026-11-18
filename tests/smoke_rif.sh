#!/bin/bash
# CPU smoke (~1-2 min): tiny random Qwen2 with a </think> token, base + re-encoding arms, report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1
T=$(mktemp -d)
python - <<PY
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
import json
text = [l for l in open("rif/checks.py")] * 20
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=600, special_tokens=["<eos>", "<think>", "</think>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<eos>"); t.save_pretrained("$T/m")
Qwen2ForCausalLM(Qwen2Config(vocab_size=len(t), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)).save_pretrained("$T/m")
with open("$T/g.jsonl", "w") as f:
    for i in range(8): f.write(json.dumps({"question": f"What is {i}+{i}?", "answer": f"#### {2*i}"}) + "\n")
PY
python -m rif.gen --model $T/m --gsm8k $T/g.jsonl --n 8 --arms base,re32 --max_new 160 --bs 12 --out $T/res --device cpu --layer 2 | tail -2
python -m rif.report --root $T/res --model $T/m | tail -8
echo "SMOKE OK ($T)"

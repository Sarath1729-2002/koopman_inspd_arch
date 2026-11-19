#!/bin/bash
# CPU smoke (~2-3 min): tiny random Qwen2, fake rewards; reference run, one-shot + both course-corrected arms, report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1 RLX_FAKE_REWARD=1
T=$(mktemp -d)
python - <<PY
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
text = [l for l in open("rlx/common.py")] * 20
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=400, special_tokens=["<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<eos>"); t.save_pretrained("$T/m")
Qwen2ForCausalLM(Qwen2Config(vocab_size=len(t), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)).save_pretrained("$T/m")
with open("$T/g.jsonl", "w") as f:
    for i in range(40): f.write(json.dumps({"question": f"What is {i}+1?", "answer": f"#### {i+1}"}) + "\n")
PY
X="--model $T/m --train $T/g.jsonl --test $T/g.jsonl --n_test 8 --prompts 4 --samples 4 --max_new 8 --t_cut 6 --ckpt $T/ck --out $T/res --device cpu"
python -m rlx.train_ref $X --steps 10 --keep 8,10 | tail -2
python -m rlx.extrap $X --mode oneshot --targets 8,10 | tail -2
python -m rlx.extrap $X --mode cc --method koop3 --k 2 --r_real 1 --t_final 10 | tail -1
python -m rlx.extrap $X --mode cc --method line3 --k 2 --r_real 1 --t_final 10 | tail -1
python -m rlx.report --root $T/res | tail -12
echo "SMOKE OK ($T)"

"""Build a tiny random Qwen2 model + BPE tokenizer for the CPU smoke test."""
import sys, os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
out = sys.argv[1]
text = open(__file__.replace("tests/make_tiny.py", "dkoop/run.py")).read().splitlines() * 5
tk = Tokenizer(models.BPE(unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.ByteLevel()
tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=500, special_tokens=["<unk>", "<eos>"]))
t = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")
t.save_pretrained(out)
cfg = Qwen2Config(vocab_size=len(t), hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024)
Qwen2ForCausalLM(cfg).save_pretrained(out)
print("tiny model ->", out)

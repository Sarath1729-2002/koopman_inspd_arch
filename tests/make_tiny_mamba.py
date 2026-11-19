import sys, numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, MambaConfig, MambaForCausalLM
out = sys.argv[1]
text = open("mcc/run.py").read().splitlines() * 30
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=500, special_tokens=["<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>"); t.save_pretrained(out + "/m")
MambaForCausalLM(MambaConfig(vocab_size=len(t), hidden_size=32, state_size=8, num_hidden_layers=3, expand=2)).save_pretrained(out + "/m")
b = ("\n".join(text) * 4).encode()
np.frombuffer(b, dtype=np.uint8).tofile(out + "/train.bin"); np.frombuffer(b[: len(b) // 3], dtype=np.uint8).tofile(out + "/eval.bin")
print("tiny ->", out)

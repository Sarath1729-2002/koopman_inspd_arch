import json, os, random, sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rnd = random.Random(0)
rows = []
for _ in range(400):
    a, b = rnd.randint(1, 20), rnd.randint(1, 20); c = a + b; d = c * 2
    rows.append({"question": f"Tom has {a} apples and gets {b} more, then doubles them. How many?",
                 "steps": [f"<<{a}+{b}={c}>>", f"<<{c}*2={d}>>"], "answer": str(d)})
os.makedirs(out + "/data", exist_ok=True)
for sp, rr in (("train", rows[:350]), ("test", rows[350:])):
    with open(f"{out}/data/{sp}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rr)
text = [r["question"] + " " + " ".join(r["steps"]) + " ### " + r["answer"] for r in rows]
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", bos_token="<|endoftext|>")
t.save_pretrained(out + "/m")
GPT2LMHeadModel(GPT2Config(vocab_size=len(t), n_positions=512, n_embd=64, n_layer=2, n_head=2)).save_pretrained(out + "/m")
print("tiny ->", out)

"""Generate reasoning traces under a trace instruction, with optional periodic re-encoding of the instruction.

arm 'base'   : no intervention
arm 're<k>'  : every k generated tokens, while the model is still thinking, inject the reminder text into the
               trace (prefilled into the cache: re-encoding the instruction through the observable channel)
For 'base', the residual stream at --layer is saved every 4th generated thinking token (for the memory/control probe).
Writes <out>/<arm>.jsonl (one row per trace) and <out>/states_base.pt."""
import argparse, json, os, re, time

import torch

from .checks import INSTR, REMIND


def load_gsm(path, n):
    rows = []
    for l in open(path):
        r = json.loads(l)
        q, a = r.get("question"), str(r.get("answer"))
        if q and a:
            rows.append((q, a.split("####")[-1].strip().replace(",", "")))
    return rows[:n]


def build_prompt(tok, q, kind):
    msg = q + "\n\n" + INSTR[kind] + " Put the final answer in \\boxed{}."
    if tok.chat_template:
        s = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    else:
        s = "User: " + msg + "\nAssistant:"
    if not s.rstrip().endswith("<think>"):
        s += "<think>\n"
    return s


def sample(logits, temp, top_p, gen):
    if temp <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits.float() / temp, -1)
    sp, si = p.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p
    sp = sp * keep
    idx = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)
    return si.gather(-1, idx)[:, 0]


@torch.no_grad()
def run_batch(model, tok, prompts, kinds, k, max_new, temp, top_p, gen, end_think, layer=None, every=4):
    dev = model.device
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    mask = enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    store = {}
    hook = None
    if layer is not None:
        def fh(mod, args, out):
            store["h"] = (out[0] if isinstance(out, tuple) else out)[:, -1].float().cpu()
        hook = model.model.layers[layer].register_forward_hook(fh)
    out = model(input_ids=enc["input_ids"], attention_mask=mask, position_ids=pos, use_cache=True)
    cache, logits = out.past_key_values, out.logits[:, -1]
    nxt = mask.sum(-1, keepdim=True)
    B = len(prompts)
    gen_ids = [[] for _ in range(B)]          # generated tokens only
    inj_at = [[] for _ in range(B)]           # indices into gen_ids where a reminder was injected (before that token)
    thinking = torch.ones(B, dtype=torch.bool, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    states = []                               # (row, gen_index, vector)
    rem = {kd: tok(REMIND.format(INSTR[kd]), add_special_tokens=False)["input_ids"] for kd in set(kinds)}
    for t in range(max_new):
        nt = sample(logits, temp, top_p, gen)
        for i in range(B):
            if not done[i]:
                gen_ids[i].append(int(nt[i]))
        thinking &= nt != end_think
        done |= nt == tok.eos_token_id
        if done.all():
            break
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        out = model(input_ids=nt[:, None], attention_mask=mask, position_ids=nxt, past_key_values=cache, use_cache=True)
        cache, logits, nxt = out.past_key_values, out.logits[:, -1], nxt + 1
        if layer is not None and t % every == 0:
            for i in range(B):
                if thinking[i] and not done[i]:
                    states.append((i, len(gen_ids[i]) - 1, store["h"][i].half()))
        if k and (t + 1) % k == 0 and (thinking & ~done).any():
            act = thinking & ~done
            L = max(len(rem[kd]) for kd in kinds)
            ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long, device=dev)
            m = torch.zeros((B, L), dtype=torch.long, device=dev)
            for i in range(B):
                if act[i]:
                    r = rem[kinds[i]]
                    ids[i, L - len(r):] = torch.tensor(r, device=dev); m[i, L - len(r):] = 1
                    inj_at[i].append(len(gen_ids[i]))
            mask = torch.cat([mask, m], 1)
            p = nxt + (m.cumsum(-1) - 1).clamp(min=0)
            out = model(input_ids=ids, attention_mask=mask, position_ids=p, past_key_values=cache, use_cache=True)
            cache, nxt = out.past_key_values, nxt + m.sum(-1, keepdim=True)
            logits = torch.where(act[:, None], out.logits[:, -1], logits)
    if hook:
        hook.remove()
    return gen_ids, inj_at, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--arms", default="base")
    ap.add_argument("--kinds", default="uppercase,no_commas,french")
    ap.add_argument("--gsm8k", default="data/gsm8k_test.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=1280)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--layer", type=int, default=None, help="layer for state capture (default: middle)")
    ap.add_argument("--out", default="results_rif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if a.device != "cpu" else torch.float32).to(a.device).eval()
    et = tok.convert_tokens_to_ids("</think>")
    end_think = et if isinstance(et, int) and et != tok.unk_token_id else -1
    layer = a.layer if a.layer is not None else len(model.model.layers) // 2
    items = load_gsm(a.gsm8k, a.n)
    jobs = [(i, q, g, kd) for kd in a.kinds.split(",") for i, (q, g) in enumerate(items)]
    os.makedirs(a.out, exist_ok=True)
    for arm in a.arms.split(","):
        k = 0 if arm == "base" else int(arm[2:])
        gen = torch.Generator(device=a.device).manual_seed(0)
        t0, rows, allstates = time.time(), [], []
        for s in range(0, len(jobs), a.bs):
            J = jobs[s:s + a.bs]
            prompts = [build_prompt(tok, q, kd) for _, q, _, kd in J]
            gids, inj, st = run_batch(model, tok, prompts, [kd for *_, kd in J], k, a.max_new, a.temp, a.top_p, gen,
                                      end_think, layer if arm == "base" else None)
            for (i, q, g, kd), ids, ij in zip(J, gids, inj):
                rows.append({"item": i, "kind": kd, "gold": g, "gen_ids": ids, "inj_at": ij})
            for r, gi, v in st:
                allstates.append((s + r, gi, v))
            print(f"[{arm}] {min(s + a.bs, len(jobs))}/{len(jobs)} traces ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(a.out, f"{arm}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if arm == "base" and allstates:
            torch.save({"idx": torch.tensor([x[0] for x in allstates]), "pos": torch.tensor([x[1] for x in allstates]),
                        "h": torch.stack([x[2] for x in allstates]), "layer": layer}, os.path.join(a.out, "states_base.pt"))
        print(f"[{arm}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

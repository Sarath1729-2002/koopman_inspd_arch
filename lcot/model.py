"""Coconut (Hao et al. 2024) on GPT-2, with periodic re-encoding of the continuous thoughts through the vocabulary.

Sequence:  question \n <bot> [n latent thoughts] <eot> remaining-steps \n ### answer <eos>
A latent thought = the last hidden state (post ln_f), fed back as the next input embedding (Coconut).
snap mode (re-encoding every k thoughts, i.e. the input after thought i is snap(h_i) when (i+1) % k == 0):
  none   Coconut
  soft   e = softmax(lm_head(h) / tau) @ W_emb                       (decode -> re-embed, differentiable)
  hard   e = W_emb[argmax lm_head(h)], straight-through gradient of the soft version
SIM-CoT (ICLR 2026) shows Coconut's thoughts drift away from the vocabulary embedding space and collapse as
the number of latent tokens grows; re-encoding through the vocabulary is the course correction for exactly that."""
import torch
import torch.nn.functional as F

SPECIAL = ["<|bot|>", "<|eot|>", "<|lat|>"]


def setup_tokenizer_model(name, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.resize_token_embeddings(len(tok))
    return tok, model.to(dev)


def snap(model, h, mode, tau=1.0):
    W = model.get_input_embeddings().weight
    logits = model.lm_head(h) / tau
    p = F.softmax(logits.float(), -1).to(W.dtype)
    soft = p @ W
    if mode == "soft":
        return soft
    hard = W[logits.argmax(-1)]
    return hard + (soft - soft.detach())


def latent_rollout(model, prefix_ids, prefix_mask, n_lat, snap_mode="none", k=1, tau=1.0, collect=False):
    """Runs the prefix, then n_lat continuous thoughts. Returns (cache, attn_mask, next_pos, thoughts list)."""
    tr = model.transformer if hasattr(model, "transformer") else model.model
    pos = (prefix_mask.cumsum(-1) - 1).clamp(min=0)
    out = tr(input_ids=prefix_ids, attention_mask=prefix_mask, position_ids=pos, use_cache=True)
    cache, h = out.past_key_values, out.last_hidden_state[:, -1:]
    mask = prefix_mask
    nxt = prefix_mask.sum(-1, keepdim=True)            # next position id per row
    thoughts = []
    for i in range(n_lat):
        thoughts.append(h[:, 0])
        inp = snap(model, h, snap_mode, tau) if (snap_mode != "none" and (i + 1) % k == 0) else h
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        out = tr(inputs_embeds=inp, attention_mask=mask, position_ids=nxt, past_key_values=cache, use_cache=True)
        cache, h, nxt = out.past_key_values, out.last_hidden_state, nxt + 1
    return cache, mask, nxt, thoughts


def suffix_forward(model, cache, mask, nxt, suf_ids, suf_mask):
    tr = model.transformer if hasattr(model, "transformer") else model.model
    m = torch.cat([mask, suf_mask], 1)
    pos = nxt + torch.arange(suf_ids.shape[1], device=suf_ids.device)[None]
    out = tr(input_ids=suf_ids, attention_mask=m, position_ids=pos, past_key_values=cache, use_cache=True)
    return model.lm_head(out.last_hidden_state), out.past_key_values, m, nxt + suf_ids.shape[1]


@torch.no_grad()
def greedy(model, cache, mask, nxt, first_ids, max_new, eos_id):
    """first_ids [B,1] are fed first; returns generated ids [B, <=max_new]."""
    tr = model.transformer if hasattr(model, "transformer") else model.model
    cur, outs = first_ids, []
    done = torch.zeros(first_ids.shape[0], dtype=torch.bool, device=first_ids.device)
    for _ in range(max_new):
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        o = tr(input_ids=cur, attention_mask=mask, position_ids=nxt, past_key_values=cache, use_cache=True)
        cache, nxt = o.past_key_values, nxt + 1
        cur = model.lm_head(o.last_hidden_state[:, -1:]).argmax(-1)
        outs.append(cur)
        done |= cur[:, 0] == eos_id
        if done.all():
            break
    return torch.cat(outs, 1)

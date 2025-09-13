#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Be safe with tokenizers threads when forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_messages(system: str, document: str, question: str, include_query: bool = True):
    user = f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except TypeError:
        s = tokenizer.apply_chat_template(messages, tokenize=False)
        if add_generation_prompt:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s


def tokens_from_messages(tokenizer, messages, device, add_generation_prompt=False):
    s = apply_chat_template(tokenizer, messages, add_generation_prompt)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def pkv_len(pkv) -> int:
    if hasattr(pkv, "layers"): return len(pkv.layers)
    if hasattr(pkv, "key_cache"): return len(pkv.key_cache)
    return len(pkv)


def pkv_get(pkv, idx: int):
    if hasattr(pkv, "layers"):
        layer = pkv.layers[idx]
        return layer.keys, layer.values
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[idx], pkv.value_cache[idx]
    return pkv[idx]


def dc_from_subset(pkv_src, idxs: List[int]) -> DynamicCache:
    dc = DynamicCache()
    for li in idxs:
        k, v = pkv_get(pkv_src, li)
        dc.update(k, v, li)
    return dc


@torch.inference_mode()
def lopa_generate(model, tokenizer, system: str, document: str, question: str, *, K: int, device: str,
                  max_new_tokens: int = 256, min_length: int = 16, temperature: float = 0.7,
                  top_p: float = 0.9, do_sample: bool = True) -> str:
    # Phase-1 ids
    msgs = build_messages(system, document, question, include_query=True)
    ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
    ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)
    sys_only   = tokens_from_messages(tokenizer, [{"role":"system","content":system}], device, add_generation_prompt=False)
    L_sys, L_all = sys_only.size(1), ids_phase1.size(1)
    L_doc = L_all - L_sys; assert L_doc > 0

    # 1) system-only prefill (base model)
    out_sys = model.model(input_ids=sys_only, attention_mask=torch.ones_like(sys_only), use_cache=True, return_dict=True)
    pkv_sys = out_sys.past_key_values
    n_layers = pkv_len(pkv_sys); K_eff = max(0, min(int(K), n_layers))

    # 2) lower-K doc pass (base model), upper continue 없음
    full_layers: nn.ModuleList = model.model.layers
    lower_layers = nn.ModuleList([full_layers[i] for i in range(K_eff)])
    model.model.layers = lower_layers
    dc_low_in = dc_from_subset(pkv_sys, list(range(K_eff))) if K_eff>0 else DynamicCache()
    attn_doc_full = torch.cat([torch.ones(1, L_sys, device=device, dtype=torch.long),
                               torch.ones(1, L_doc, device=device, dtype=torch.long)], dim=1)
    out_low = model.model(input_ids=ids_phase1[:, L_sys:], past_key_values=dc_low_in, attention_mask=attn_doc_full,
                          use_cache=True, return_dict=True)
    pkv_low = out_low.past_key_values
    model.model.layers = full_layers

    # 3) 결합: lower=sys+doc, upper=빈 past (length 0)
    combined = DynamicCache()
    for li in range(n_layers):
        k_sys, v_sys = pkv_get(pkv_sys, li)
        if li < K_eff:
            k_cat = torch.cat([k_sys[:, :, :L_sys, :], pkv_get(pkv_low, li)[0][:, :, -L_doc:, :]], dim=2)
            v_cat = torch.cat([v_sys[:, :, :L_sys, :], pkv_get(pkv_low, li)[1][:, :, -L_doc:, :]], dim=2)
        else:
            k_cat = k_sys[:, :, :0, :]
            v_cat = v_sys[:, :, :0, :]
        combined.update(k_cat.contiguous(), v_cat.contiguous(), li)

    # 4) header push step-by-step
    hdr_tail = ids_hdr[:, L_all:]
    if hdr_tail.numel() > 0:
        for j in range(hdr_tail.size(1)):
            past_len = pkv_get(combined, 0)[0].shape[2]
            attn_mask = torch.cat([torch.ones(1, past_len, device=device, dtype=torch.long),
                                   torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
            step_tok = hdr_tail[:, j:j+1]
            out_seed = model(input_ids=step_tok, past_key_values=combined, attention_mask=attn_mask,
                             use_cache=True, return_dict=True)
            combined = out_seed.past_key_values

    # 5) decoding
    from transformers.generation import LogitsProcessorList
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper
    eos_id = tokenizer.eos_token_id
    last = ids_hdr[:, -1:]
    generated = torch.empty((1,0), dtype=torch.long, device=device)
    procs = LogitsProcessorList()
    if temperature and temperature != 1.0: procs.append(TemperatureLogitsWarper(temperature=float(temperature)))
    if top_p and top_p < 1.0: procs.append(TopPLogitsWarper(top_p=float(top_p), min_tokens_to_keep=1))
    cur = 0
    while cur < max_new_tokens:
        past_len = pkv_get(combined, 0)[0].shape[2]
        attn_mask = torch.cat([torch.ones(1, past_len, device=device, dtype=torch.long),
                               torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
        out = model(input_ids=last, past_key_values=combined, attention_mask=attn_mask, use_cache=True, return_dict=True)
        combined = out.past_key_values
        logits = out.logits[:, -1, :].to(torch.float32)
        if eos_id is not None and cur < min_length: logits[:, eos_id] = -float("inf")
        inp = generated if generated.numel()>0 else last.new_zeros((1,0), dtype=torch.long, device=device)
        logits = procs(inp, logits)
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and int(next_tok.item()) == eos_id: break
        generated = torch.cat([generated, next_tok], dim=1)
        last = next_tok
        cur += 1
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser("LoPA-only inference helper")
    ap.add_argument("--best_dir", type=str, required=True, help="Path to best/ folder produced by training")
    ap.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prefill_layers", type=int, default=4)
    ap.add_argument("--system", type=str, default="You are a helpful assistant that answers questions based on the given document. ")
    ap.add_argument("--document", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_length", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device=="cuda" else torch.float32)

    best_dir = Path(args.best_dir)
    tok_src = str(best_dir) if (best_dir / "tokenizer.json").is_file() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=False, torch_dtype=dtype)
    from peft import PeftModel
    peft = PeftModel.from_pretrained(base, str(best_dir / "lora"))
    model = peft.merge_and_unload().to(device).eval()
    for k in ("attn_implementation", "_attn_implementation"):
        try:
            setattr(model.config, k, "sdpa"); setattr(model.model.config, k, "sdpa")
        except Exception:
            pass

    text = lopa_generate(
        model, tokenizer,
        system=args.system, document=args.document, question=args.question,
        K=int(args.prefill_layers), device=device,
        max_new_tokens=args.max_new_tokens, min_length=args.min_length,
        temperature=args.temperature, top_p=args.top_p, do_sample=True,
    )
    print(text)


if __name__ == "__main__":
    main()


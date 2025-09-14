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

# Special assistant-start token for Mistral-style templates
MISTRAL_ASSIST_START = "<Mistral_start>"

def _is_mistral_template(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    name = getattr(getattr(tokenizer, "init_kwargs", {}), "get", lambda k, d=None: d)("name_or_path", "")
    return ("[INST]" in tmpl) or ("mistral" in str(name).lower()) or ("mistral" in tmpl.lower())

def ensure_mistral_special_token(tokenizer, model=None):
    """Ensure the custom assistant-start token exists in tokenizer (and resize model embeddings if provided)."""
    if not _is_mistral_template(tokenizer):
        return False
    add_tok = []
    cur = set(tokenizer.get_vocab().keys())
    if MISTRAL_ASSIST_START not in cur:
        add_tok.append(MISTRAL_ASSIST_START)
    if add_tok:
        tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.special_tokens_map_extended.get("additional_special_tokens", []) + add_tok})
        if model is not None:
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
        return True
    return False


def build_messages(system: str, document: str, question: str, include_query: bool = True):
    user = f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    """Render chat with robust fallback across templates.

    - Prefer tokenizer.apply_chat_template(..., add_generation_prompt=...)
    - If that signature is unsupported, detect template style:
        * Llama-3 style → append assistant header tokens
        * Mistral/INST style → no explicit assistant header to append
        * Unknown → do not append anything
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except TypeError:
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        s = tokenizer.apply_chat_template(messages, tokenize=False)
        if add_generation_prompt:
            if "<|start_header_id|>" in tmpl:  # Llama 3 style
                s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            elif "[INST]" in tmpl or "</s>" in tmpl:  # Mistral style: no explicit header
                s += ""
            else:
                # Unknown template → safest is to append nothing
                s += ""
        return s


def tokens_from_messages(tokenizer, messages, device, add_generation_prompt=False):
    s = apply_chat_template(tokenizer, messages, add_generation_prompt)
    ids = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    # If Mistral template and generation prompt requested, append our assistant-start header token
    if add_generation_prompt and _is_mistral_template(tokenizer):
        try:
            tok_id = tokenizer.convert_tokens_to_ids(MISTRAL_ASSIST_START)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                extra = torch.tensor([[int(tok_id)]], device=ids.device, dtype=ids.dtype)
                ids = torch.cat([ids, extra], dim=1)
        except Exception:
            pass
    return ids


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


def _get_inner_model(m):
    """Return the base model object that holds .layers (e.g., LlamaModel/MistralModel)."""
    if hasattr(m, "module"):
        m = m.module
    # unwrap peft if any
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            try:
                base = m.get_base_model()
            except Exception:
                base = getattr(m, "base_model", m)
            if hasattr(base, "model"):
                return base.model
            if hasattr(base, "transformer"):
                return base.transformer
            m = base
    except Exception:
        pass
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model
    if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
        return m.transformer
    return m.model


def _kv_meta_from_model(model_like):
    """Return (num_kv_heads, head_dim, dtype)."""
    try:
        cfg = getattr(model_like, "config", None) or getattr(_get_inner_model(model_like), "config", None)
    except Exception:
        cfg = getattr(_get_inner_model(model_like), "config", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv = getattr(cfg, "num_key_value_heads", None) or num_heads
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = (hidden // num_heads) if (hidden and num_heads) else None
    try:
        dtype = next(_get_inner_model(model_like).parameters()).dtype
    except Exception:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return int(num_kv), int(head_dim), dtype


def _make_empty_kv(batch: int, num_kv: int, head_dim: int, device, dtype):
    shape = (batch, num_kv, 0, head_dim)
    k = torch.empty(shape, device=device, dtype=dtype)
    v = torch.empty(shape, device=device, dtype=dtype)
    return k.contiguous(), v.contiguous()

@torch.inference_mode()
def lopa_generate(model, tokenizer, system: str, document: str, question: str, *, K: int, device: str,
                  max_new_tokens: int = 256, min_length: int = 16, temperature: float = 0.7,
                  top_p: float = 0.9, top_k: int | None = None, do_sample: bool = True,
                  debug: bool = False) -> str:
    # Phase-1 ids
    msgs = build_messages(system, document, question, include_query=True)
    ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
    ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)
    sys_only   = tokens_from_messages(tokenizer, [{"role":"system","content":system}], device, add_generation_prompt=False)
    L_sys, L_all = sys_only.size(1), ids_phase1.size(1)
    L_doc = L_all - L_sys; assert L_doc > 0

    # 1) system-only prefill (lower-K only)
    inner = _get_inner_model(model)
    full_layers: nn.ModuleList = inner.layers
    n_layers = len(full_layers)
    K_eff = max(0, min(int(K), n_layers))
    lower_layers = nn.ModuleList([full_layers[i] for i in range(K_eff)])
    inner.layers = lower_layers
    out_sys_low = inner(input_ids=sys_only, attention_mask=torch.ones_like(sys_only), use_cache=True, return_dict=True)
    pkv_sys_low = out_sys_low.past_key_values

    # 2) lower-K doc pass
    dc_low_in = dc_from_subset(pkv_sys_low, list(range(K_eff))) if K_eff>0 else DynamicCache()
    out_low = inner(input_ids=ids_phase1[:, L_sys:], past_key_values=dc_low_in, attention_mask=None,
                    use_cache=True, return_dict=True)
    pkv_low = out_low.past_key_values
    inner.layers = full_layers

    # 3) 결합: lower=sys+doc, upper=빈 past (length 0)
    combined = DynamicCache()
    num_kv, head_dim, kv_dtype = _kv_meta_from_model(model)
    for li in range(n_layers):
        if li < K_eff:
            k_cat = torch.cat([pkv_get(pkv_sys_low, li)[0][:, :, :L_sys, :],
                               pkv_get(pkv_low, li)[0][:, :, -L_doc:, :]], dim=2)
            v_cat = torch.cat([pkv_get(pkv_sys_low, li)[1][:, :, :L_sys, :],
                               pkv_get(pkv_low, li)[1][:, :, -L_doc:, :]], dim=2)
        else:
            k_cat, v_cat = _make_empty_kv(1, num_kv, head_dim, device, kv_dtype)
        combined.update(k_cat.contiguous(), v_cat.contiguous(), li)

    if debug:
        lower_l = sorted({int(pkv_get(combined, li)[0].shape[2]) for li in range(0, K_eff)}) if K_eff>0 else []
        upper_l = sorted({int(pkv_get(combined, li)[0].shape[2]) for li in range(K_eff, n_layers)}) if K_eff<n_layers else []
        print(f"[debug] L_sys={L_sys}, L_doc={L_doc}, K={K_eff} | lower_lens={lower_l} | upper_lens={upper_l}")

    # 4) header push step-by-step (seed upper layers).
    #    If no header exists (common for Mistral templates), push the user's end-of-turn token
    #    i.e., the last token of the user prompt sequence (ids_phase1[:, -1:]).
    hdr_tail = ids_hdr[:, L_all:]
    pushed = 0
    last_pushed = None
    if hdr_tail.numel() > 0:
        for j in range(hdr_tail.size(1)):
            step_tok = hdr_tail[:, j:j+1]
            out_seed = model(input_ids=step_tok, past_key_values=combined, attention_mask=None,
                             use_cache=True, return_dict=True)
            combined = out_seed.past_key_values
            pushed += 1
            last_pushed = step_tok
    else:
        # Mistral-style: prefer user EOT (=eos) if present at the end of user turn; otherwise last user token
        try:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            last_tok = ids_phase1[:, -1:]
            if eos_id is not None and int(last_tok.item()) == int(eos_id):
                step_tok = last_tok  # use existing </s> without appending
            else:
                step_tok = last_tok
            out_seed = model(input_ids=step_tok, past_key_values=combined, attention_mask=None,
                             use_cache=True, return_dict=True)
            combined = out_seed.past_key_values
            pushed += 1
            last_pushed = step_tok
        except Exception:
            pass
    if debug:
        lower_l2 = sorted({int(pkv_get(combined, li)[0].shape[2]) for li in range(0, K_eff)}) if K_eff>0 else []
        upper_l2 = sorted({int(pkv_get(combined, li)[0].shape[2]) for li in range(K_eff, n_layers)}) if K_eff<n_layers else []
        print(f"[debug] after header/seed push: pushed={pushed} | lower_lens={lower_l2} | upper_lens={upper_l2}")

    # 5) decoding
    from transformers.generation import LogitsProcessorList
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper
    # Treat both eos and eot as stopping tokens
    eos_id = tokenizer.eos_token_id
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None
    stop_ids = set()
    if eos_id is not None: stop_ids.add(int(eos_id))
    if eot_id is not None and eot_id != tokenizer.unk_token_id: stop_ids.add(int(eot_id))
    # Start token for decoding: if we pushed header/seed, use the last token we just pushed; otherwise use last of hdr
    last = last_pushed if pushed > 0 else ids_hdr[:, -1:]
    generated = torch.empty((1,0), dtype=torch.long, device=device)
    procs = None
    if do_sample:
        procs = LogitsProcessorList()
        if temperature and temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature=float(temperature)))
        if top_p and top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p=float(top_p), min_tokens_to_keep=1))
        if top_k is not None and top_k > 0:
            procs.append(TopKLogitsWarper(top_k=int(top_k), filter_value=-float("inf")))
    cur = 0
    while cur < max_new_tokens:
        out = model(input_ids=last, past_key_values=combined, attention_mask=None, use_cache=True, return_dict=True)
        combined = out.past_key_values
        logits = out.logits[:, -1, :].to(torch.float32)
        if stop_ids and cur < min_length:
            for sid in stop_ids:
                logits[:, sid] = -float("inf")
        inp = generated if generated.numel()>0 else last.new_zeros((1,0), dtype=torch.long, device=device)
        if procs is not None:
            logits = procs(inp, logits)
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        if int(next_tok.item()) in stop_ids: break
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
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--do_sample", action="store_true")
    # numeric controls for reproducibility
    ap.add_argument("--dtype", type=str, choices=["auto","bf16","fp16","fp32"], default="auto")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--sdpa_math_only", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device=="cuda" else torch.float32)

    # global numeric toggles
    if args.no_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    if args.sdpa_math_only and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    best_dir = Path(args.best_dir)
    tok_src = str(best_dir) if (best_dir / "tokenizer.json").is_file() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prefer a saved base backbone under best_dir/base if present (captures resized embeddings)
    base_path = best_dir / "base"
    base_load_src = str(base_path) if base_path.exists() and any(base_path.iterdir()) else args.model_name
    base = AutoModelForCausalLM.from_pretrained(base_load_src, trust_remote_code=False, torch_dtype=dtype)
    # Ensure special token availability for Mistral
    ensure_mistral_special_token(tokenizer, base)
    # attach LoRA if exists
    lora_path = best_dir / "lora"
    if lora_path.exists() and any(lora_path.iterdir()):
        try:
            from peft import PeftModel
            peft = PeftModel.from_pretrained(base, str(lora_path))
            model = peft.merge_and_unload().to(device).eval()
        except Exception:
            model = base.to(device).eval()
    else:
        model = base.to(device).eval()
    # Force eager attention for all models (stability)
    impl = "eager"
    for k in ("attn_implementation", "_attn_implementation"):
        try:
            setattr(model.config, k, impl)
            setattr(model.model.config, k, impl)
        except Exception:
            pass
    print("[infer] Forcing attn_implementation='eager' for all models (stability mode).")

    text = lopa_generate(
        model, tokenizer,
        system=args.system, document=args.document, question=args.question,
        K=int(args.prefill_layers), device=device,
        max_new_tokens=args.max_new_tokens, min_length=args.min_length,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        do_sample=bool(args.do_sample), debug=bool(args.debug),
    )
    print(text)


if __name__ == "__main__":
    main()

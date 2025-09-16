#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoPA inference debugger (prefill K-layers only + header-start generation)

- Lower K layers: prefill with [system + document]
- Upper layers: start empty
- No remapping of positions (match training)
- Step-by-step: show top-k next tokens and per-layer past lengths

Works with checkpoints saved by the provided trainer:
  ckpt/
    base/  (clean backbone)
    lora/  (optional LoRA adapters)
  tokenizer files live under ckpt/ as saved by trainer
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ─────────────────────────────────────────────────────────────────────────────
# Constants & utils
# ─────────────────────────────────────────────────────────────────────────────
MISTRAL_ASSIST_START = "<Mistral_start>"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_inner_model(m):
    if hasattr(m, "module"):
        m = m.module
    try:
        from peft import PeftModel   # unwrap if needed
        if isinstance(m, PeftModel):
            try:
                m = m.get_base_model()
            except Exception:
                m = getattr(m, "base_model", m)
    except Exception:
        pass

    for attr in ("model","transformer","backbone","base_model","language_model"):
        if hasattr(m, attr):
            cand = getattr(m, attr)
            if hasattr(cand, "layers") and isinstance(getattr(cand, "layers", None), nn.ModuleList):
                return cand
            if hasattr(cand, "decoder") and hasattr(cand.decoder, "layers"):
                if isinstance(cand.decoder.layers, nn.ModuleList):
                    return cand.decoder
    if hasattr(m, "layers") and isinstance(getattr(m, "layers", None), nn.ModuleList):
        return m
    for child in m.modules():
        if child is m:
            continue
        if hasattr(child, "layers") and isinstance(getattr(child, "layers", None), nn.ModuleList):
            return child
    raise AttributeError("Could not locate inner base model with a .layers attribute")

def _is_mistral_template(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    name = getattr(getattr(tokenizer, "init_kwargs", {}), "get", lambda k, d=None: d)("name_or_path", "")
    return ("[INST]" in tmpl) or ("mistral" in str(name).lower()) or ("mistral" in tmpl.lower())

def ensure_mistral_special_token(tokenizer, model=None):
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
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except TypeError:
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        s = tokenizer.apply_chat_template(messages, tokenize=False)
        if add_generation_prompt and "<|start_header_id|>" in tmpl:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s

def tokens_from_messages(tokenizer, messages, device, add_generation_prompt=False):
    s = apply_chat_template(tokenizer, messages, add_generation_prompt)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

def pkv_len(pkv) -> int:
    if hasattr(pkv, "key_cache"): return len(pkv.key_cache)
    if hasattr(pkv, "layers"):    return len(pkv.layers)
    return len(pkv)

def pkv_get(pkv, idx: int):
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return pkv.key_cache[idx], pkv.value_cache[idx]
    if hasattr(pkv, "layers"):
        layer = pkv.layers[idx];  return layer.keys, layer.values
    return pkv[idx]

def dc_from_subset(pkv_src, layer_indices: List[int]) -> DynamicCache:
    dc = DynamicCache()
    for li in layer_indices:
        k, v = pkv_get(pkv_src, li)
        dc.update(k, v, li)
    return dc

def _get_input_embeddings(model_like) -> nn.Embedding:
    try:
        emb = model_like.get_input_embeddings()
        if isinstance(emb, nn.Embedding): return emb
    except Exception:
        pass
    inner = _get_inner_model(model_like)
    for name in ("embed_tokens","wte","word_embeddings"):
        if hasattr(inner, name):
            emb = getattr(inner, name)
            if isinstance(emb, nn.Embedding): return emb
    raise AttributeError("Could not locate input embeddings module")

def _kv_meta_from_model(model_like):
    try: cfg = getattr(model_like, "config", None) or getattr(_get_inner_model(model_like), "config", None)
    except Exception: cfg = getattr(_get_inner_model(model_like), "config", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv    = getattr(cfg, "num_key_value_heads", None) or num_heads
    hidden    = getattr(cfg, "hidden_size", None)
    head_dim  = (hidden // num_heads) if (hidden and num_heads) else None
    try: dtype = next(_get_inner_model(model_like).parameters()).dtype
    except Exception: dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return int(num_kv), int(head_dim), dtype

def _make_empty_kv(batch: int, num_kv: int, head_dim: int, device, dtype):
    shape = (batch, num_kv, 0, head_dim)
    k = torch.empty(shape, device=device, dtype=dtype)
    v = torch.empty(shape, device=device, dtype=dtype)
    return k.contiguous(), v.contiguous()

# ─────────────────────────────────────────────────────────────────────────────
# LoPA cache_position/position_ids 동적 오프셋 패치 (훈련 코드 동일)
# ─────────────────────────────────────────────────────────────────────────────
import contextlib
@contextlib.contextmanager
def lopa_cache_position_patch(model, past_key_values):
    inner = _get_inner_model(model)
    def _pkv_past_len(li: int) -> int:
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return int(past_key_values.key_cache[li].shape[2])
        if hasattr(past_key_values, "layers"):
            return int(past_key_values.layers[li].keys.shape[2])
        return int(past_key_values[li][0].shape[2])
    past_lens = [ _pkv_past_len(li) for li in range(len(inner.layers)) ]

    handles = []
    for li, layer in enumerate(inner.layers):
        layer._lopa_li = li
        layer._lopa_past = int(past_lens[li])

        def _pre_hook(module, args, kwargs):
            past_len = getattr(module, "_lopa_past", 0)
            cp = kwargs.get("cache_position", None)
            pi = kwargs.get("position_ids", None)
            start_val = None
            if isinstance(cp, torch.Tensor) and cp.numel() > 0:
                start_val = int(cp.view(-1)[0].item())
            elif isinstance(pi, torch.Tensor) and pi.numel() > 0:
                start_val = int(pi.view(-1)[0].item())
            if start_val is not None:
                off = start_val - past_len
                if off != 0:
                    if isinstance(cp, torch.Tensor): kwargs["cache_position"] = cp - off
                    if isinstance(pi, torch.Tensor): kwargs["position_ids"] = pi - off
            return args, kwargs

        h = layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        handles.append(h)

    try:
        yield
    finally:
        for h in handles: h.remove()
        for layer in inner.layers:
            for attr in ("_lopa_li","_lopa_past"):
                if hasattr(layer, attr): delattr(layer, attr)

# ─────────────────────────────────────────────────────────────────────────────
# Prefill & step-by-step generation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def build_combined_prefill_cache(model, tokenizer, system_prompt: str, document: str, question: str, K: int, device):
    """Return (combined_cache, lengths_meta) and some tokenized pieces for later steps."""
    msgs = build_messages(system_prompt, document, question, include_query=True)
    ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
    ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)
    sys_only   = tokens_from_messages(tokenizer, [{"role":"system","content":system_prompt}], device, add_generation_prompt=False)

    L_sys = sys_only.size(1)
    L_all = ids_phase1.size(1)
    L_doc = L_all - L_sys
    assert L_doc > 0, "Document tokens must be > 0"

    inner = _get_inner_model(model)
    full_layers: nn.ModuleList = inner.layers
    n_layers = len(full_layers)
    K_eff = max(0, min(K, n_layers))

    # 1) lower K layers: prefill [system] then [document]
    lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
    inner.layers = lower_layers
    out_sys_low = inner(input_ids=sys_only, attention_mask=torch.ones_like(sys_only), use_cache=True, return_dict=True)
    pkv_sys_low = out_sys_low.past_key_values

    dc_low_in = dc_from_subset(pkv_sys_low, list(range(K_eff))) if K_eff > 0 else DynamicCache()
    out_low = inner(input_ids=ids_phase1[:, L_sys:], past_key_values=dc_low_in,
                    attention_mask=None, use_cache=True, return_dict=True)
    pkv_low = out_low.past_key_values

    # restore all layers
    inner.layers = full_layers

    # 2) make combined cache (lower: sys+doc, upper: empty)
    combined = DynamicCache()
    num_kv, head_dim, kv_dtype = _kv_meta_from_model(model)
    for li in range(n_layers):
        if li < K_eff:
            k_sys, v_sys = pkv_get(pkv_sys_low, li)   # [B,Nkv,Lsys,H]
            k_low, v_low = pkv_get(pkv_low, li)       # [B,Nkv,Lsys+Ldoc,H]
            k_sys_slice = k_sys[:, :, :L_sys, :]
            v_sys_slice = v_sys[:, :, :L_sys, :]
            k_doc = k_low[:, :, -L_doc:, :]
            v_doc = v_low[:, :, -L_doc:, :]
            combined.update(torch.cat([k_sys_slice, k_doc], dim=2).contiguous(),
                            torch.cat([v_sys_slice, v_doc], dim=2).contiguous(), li)
        else:
            k_empty, v_empty = _make_empty_kv(1, num_kv, head_dim, device, kv_dtype)
            combined.update(k_empty, v_empty, li)

    # header tail used as seed for generation
    hdr_tail = ids_hdr[:, L_all:]              # assistant header piece only
    seed_default = hdr_tail if hdr_tail.numel() > 0 else ids_phase1[:, -1:]

    meta = dict(L_sys=int(L_sys), L_doc=int(L_doc), L_all=int(L_all), n_layers=int(n_layers), K_eff=int(K_eff))
    return combined, seed_default, ids_phase1, ids_hdr, meta

def print_cache_lengths(cache, tag: str):
    print(f"\n[{tag}] Combined cache per-layer past lengths:")
    n = pkv_len(cache)
    for li in range(n):
        k, _ = pkv_get(cache, li)
        print(f"  layer {li:02d}: past_seq = {int(k.shape[2])}")

def topk_from_logits(logits, tokenizer, k=10, temperature=0.0):
    last = logits[:, -1, :]  # [B,V]
    if temperature and temperature > 0.0:
        last = last / float(temperature)
    probs = last.softmax(dim=-1)
    top_p, top_i = torch.topk(probs, k, dim=-1)
    toks = [tokenizer.decode([int(t.item())], skip_special_tokens=False) for t in top_i[0]]
    return [(int(i.item()), float(p.item()), t) for i, p, t in zip(top_i[0], top_p[0], toks)]

@torch.no_grad()
def step_once(model, cache, input_ids, device, patch=True):
    # 한 스텝(forward). HF가 내부에서 cache_position/position_ids를 구성한다.
    if patch:
        with lopa_cache_position_patch(model, cache):
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    else:
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    return out.logits, out.past_key_values

def decode_piece(tokenizer, ids: torch.Tensor, limit=120):
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)[:limit].replace("\n","⏎")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("LoPA inference debugger")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to _best_ckpt")
    ap.add_argument("--prefill_layers", type=int, default=4)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--document", type=str, required=True)
    ap.add_argument("--system_prompt", type=str, default="You are a helpful assistant that answers questions based on the given document. ")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", type=str, choices=["auto","bf16","fp16","fp32"], default="auto")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer 우선 로드 (ckpt 루트에 저장됨)
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # dtype 결정
    if args.dtype == "fp32": dtype = torch.float32
    elif args.dtype == "bf16": dtype = torch.bfloat16
    elif args.dtype == "fp16": dtype = torch.float16
    else:
        dtype = (torch.bfloat16 if (device=="cuda" and torch.cuda.is_bf16_supported())
                 else (torch.float16 if device=="cuda" else torch.float32))

    # 모델 로드 (base + optional LoRA)
    base_dir = Path(args.ckpt) / "base"
    has_base = base_dir.is_dir()
    if not has_base:
        raise FileNotFoundError(f"Base backbone not found at {base_dir}")

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), trust_remote_code=False, torch_dtype=dtype).to(device)

    lora_dir = Path(args.ckpt) / "lora"
    if lora_dir.is_dir():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(lora_dir))
            print("[Info] Loaded LoRA adapters.")
        except Exception as e:
            print(f"[Warn] Failed to load LoRA: {e}")

    # Mistral 템플릿이면 start special token 보장
    _ = ensure_mistral_special_token(tok, model)

    # 안정성: eager 강제 (훈련과 동일)
    for k in ("attn_implementation","_attn_implementation"):
        try:
            setattr(model.config, k, "eager")
            setattr(_get_inner_model(model).config, k, "eager")
        except Exception:
            pass

    model.eval()

    # 1) Prefill (하위 K만 sys+doc, 상위는 빈 KV)
    combined, seed_default, ids_phase1, ids_hdr, meta = build_combined_prefill_cache(
        model, tok, args.system_prompt, args.document, args.question, args.prefill_layers, device
    )
    L_sys, L_doc, L_all, n_layers, K_eff = meta["L_sys"], meta["L_doc"], meta["L_all"], meta["n_layers"], meta["K_eff"]

    print("\n=== [Phase-1] Prefill result check ===")
    print(f"Layers total={n_layers} | K_eff={K_eff} | L_sys={L_sys} | L_doc={L_doc} | L_all={L_all}")
    print_cache_lengths(combined, tag="prefill")
    # 검증: 기대 past 길이
    ok = True
    for li in range(n_layers):
        k, _ = pkv_get(combined, li)
        expected = (L_sys + L_doc) if li < K_eff else 0
        if int(k.shape[2]) != expected:
            ok = False
    print(f"Prefill KV shape check: {'OK' if ok else 'MISMATCH'}")

    # 2) Seed 토큰 확정 (header tail 우선, 없으면 Mistral 전용 토큰, 그 외는 마지막 토큰 fallback)
    seed = seed_default
    if seed.numel() == 0:
        if _is_mistral_template(tok):
            tid = tok.convert_tokens_to_ids(MISTRAL_ASSIST_START)
            if tid is not None and tid >= 0:
                seed = torch.tensor([[int(tid)]], device=device, dtype=ids_hdr.dtype)
        if seed.numel() == 0:
            seed = ids_phase1[:, -1:]
    print("\n=== [Phase-2] Seed check ===")
    print(f"Seed length = {seed.size(1)} | seed tokens (decoded preview): {decode_piece(tok, seed)}")

    # 2-a) Seed만 입력해서 "다음 토큰 분포" 보기 (아직 생성하지 않고 logits만 본다)
    logits, combined = step_once(model, combined, input_ids=seed, device=device, patch=True)
    topk = topk_from_logits(logits, tok, k=args.topk, temperature=args.temperature)
    print("\n[Next-token distribution | after SEED only]")
    for rank, (tid, prob, txt) in enumerate(topk, 1):
        print(f"  {rank:2d}. id={tid:6d}  p={prob:8.5f}  tok={repr(txt)}")
    print_cache_lengths(combined, tag="after-seed")
    # 기대 past 길이: 하위 K는 L_sys+L_doc+seed_len, 상위는 seed_len
    seed_len = int(seed.size(1))
    ok2 = True
    for li in range(n_layers):
        k, _ = pkv_get(combined, li)
        expected = (L_sys + L_doc + seed_len) if li < K_eff else seed_len
        if int(k.shape[2]) != expected:
            ok2 = False
    print(f"After-seed KV shape check: {'OK' if ok2 else 'MISMATCH'}")

    # 3) Step-by-step 생성
    print("\n=== [Phase-3] Step-by-step generation ===")
    gen = []
    last_input = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) if args.temperature == 0.0 else None

    for step in range(args.max_new_tokens):
        # 선택 규칙: temperature>0이면 샘플링, 0이면 그리디
        if args.temperature and args.temperature > 0.0:
            # sampling from last logits (if first loop) else we run one-step with last token then sample
            if step == 0:
                dist = torch.distributions.Categorical(logits=logits[:, -1, :] / float(args.temperature))
                next_id = dist.sample().unsqueeze(0)
            else:
                # already computed in this iteration below
                pass
        else:
            next_id = last_input  # greedy from previous logits on step=0

        if step == 0 and next_id is not None:
            # 첫 스텝은 seed 뒤의 첫 토큰을 이미 선택했으므로 그걸 넣어 step_once
            stdin = next_id
        else:
            # 일반적으로는 방금 생성한 토큰 1개를 입력으로 사용
            stdin = last_input if last_input is not None else next_id

        logits, combined = step_once(model, combined, input_ids=stdin, device=device, patch=True)

        # top-k 리포트
        report = topk_from_logits(logits, tok, k=args.topk, temperature=args.temperature)
        chosen_id = (torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                     if args.temperature == 0.0
                     else torch.distributions.Categorical(logits=logits[:, -1, :]/float(args.temperature)).sample().unsqueeze(0))

        gen.append(int(stdin[0,0].item()))
        last_input = chosen_id

        dec_partial = tok.decode(gen, skip_special_tokens=False).replace("\n","⏎")
        print(f"\n[Step {step:02d}] input={stdin[0,0].item()} '{tok.decode([int(stdin[0,0])], skip_special_tokens=False)}'")
        for r,(tid,prob,txt) in enumerate(report, 1):
            print(f"  {r:2d}. id={tid:6d}  p={prob:8.5f}  tok={repr(txt)}")
        print(f"  -> chosen next id (for next step input) = {int(chosen_id[0,0])} '{tok.decode([int(chosen_id[0,0])], skip_special_tokens=False)}'")
        print_cache_lengths(combined, tag=f"after-step-{step:02d}")
        # 기대 길이 갱신 검증 (seed_len + (step+1) 만큼 증가)
        ok_step = True
        for li in range(n_layers):
            k, _ = pkv_get(combined, li)
            expected = (L_sys + L_doc + seed_len + (step+1)) if li < K_eff else (seed_len + (step+1))
            if int(k.shape[2]) != expected:
                ok_step = False
        print(f"  KV shape check: {'OK' if ok_step else 'MISMATCH'}")
        print(f"  partial decode: {dec_partial[:160]}")

    # 최종 디코드
    out_text = tok.decode(gen, skip_special_tokens=False)
    print("\n=== [Done] Generated text (raw) ===")
    print(out_text)

if __name__ == "__main__":
    main()

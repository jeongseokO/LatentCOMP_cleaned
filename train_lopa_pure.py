#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoPA-only trainer (Phase-1: lower-K prefill only; Phase-2: generation starting at assistant header)

Behavior:
- Prefill builds KV only for lower K layers from [system + document]; upper layers have no prefill KV.
- Assistant header is NOT included in prefill; it is pushed as the first token(s) of generation.
- Positions are unchanged (no remapping). Lower layers start with len=L_sys+L_doc; upper layers start with len=0 and grow.

This trainer is intentionally minimal and focused. It uses:
- HF AutoModelForCausalLM (trust_remote_code=False) + optional LoRA via peft
- A single-device loop (Accelerate not required here); integrate with Slurm outside if needed
- JSONL dataset with fields: question (str), document (str), optional response(s)

Outputs:
- best/ folder with base/ (clean backbone) and lora/ (adapter) + tokenizer & configs
Downstream LatentCOMP_cleaned/train.py repack step will inject remote-code wrappers if configured.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator
try:
    from accelerate import FullyShardedDataParallelPlugin, DeepSpeedPlugin
except Exception:
    FullyShardedDataParallelPlugin = None
    DeepSpeedPlugin = None
from huggingface_hub import create_repo, upload_folder
from transformers.cache_utils import DynamicCache
import contextlib

# Safety: avoid HF tokenizers parallel threads before forking (DataLoader workers)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Special assistant-start token for Mistral-style templates
MISTRAL_ASSIST_START = "<Mistral_start>"

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

def _get_input_embeddings(model_like) -> nn.Embedding:
    try:
        emb = model_like.get_input_embeddings()
        if isinstance(emb, nn.Embedding):
            return emb
    except Exception:
        pass
    try:
        inner = _get_inner_model(model_like)
        for name in ("embed_tokens", "wte", "word_embeddings"):
            if hasattr(inner, name):
                emb = getattr(inner, name)
                if isinstance(emb, nn.Embedding):
                    return emb
    except Exception:
        pass
    raise AttributeError("Could not locate input embeddings module")

def _strip_automap(cfg):
    try:
        if hasattr(cfg, "auto_map"):
            try: delattr(cfg, "auto_map")
            except Exception:
                try: setattr(cfg, "auto_map", {})
                except Exception: pass
        try: cfg.__dict__.pop("auto_map", None)
        except Exception: pass
    except Exception:
        pass

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_argparser():
    p = argparse.ArgumentParser("LoPA-only trainer (lower-K prefill, header-start generation)")
    # core
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefill_layers", type=int, default=4)
    # data options
    p.add_argument("--max_responses_per_sample", type=int, default=5,
                   help="Deprecated (kept for compat; ignored if explode=True)")
    # LoRA (optional)
    p.add_argument("--use_lora", type=str, default="True")
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # backend
    p.add_argument("--attn_impl", type=str, choices=["sdpa", "eager"], default="sdpa")
    # numeric controls
    p.add_argument("--dtype", type=str, choices=["auto","bf16","fp16","fp32"], default="auto",
                   help="Force compute dtype (auto=bf16 if supported else fp16; cpu=fp32)")
    p.add_argument("--no_tf32", action="store_true", help="Disable TF32 matmul/cuDNN")
    p.add_argument("--sdpa_math_only", action="store_true", help="Use SDPA math kernel only (disable flash/mem-e)")
    # schedule / clip
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # IO
    p.add_argument("--save_best_dir", type=str, default="./_best_ckpt")
    p.add_argument("--cache_dir_model", type=str, default=None)
    p.add_argument("--cache_dir_tokenizer", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    # data controls
    p.add_argument("--max_doc_tokens", type=int, default=2048)
    p.add_argument("--explode", action="store_true", default=True)
    p.add_argument("--no_explode", dest="explode", action="store_false")
    p.add_argument("--group_by_question", action="store_true", default=True)
    p.add_argument("--no_group_by_question", dest="group_by_question", action="store_false")
    # HF Hub
    p.add_argument("--hf_repo_id", type=str, default=None)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")
    # distributed / sharding
    p.add_argument("--dist_mode", type=str, choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    p.add_argument("--zero_stage", type=int, default=2)
    return p

class QADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_doc_tokens: int = 2048,
                 explode: bool = True, group_by_question: bool = True, seed: int = 42):
        self.recs = []
        rng = random.Random(int(seed))
        auto_idx = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = (rec.get("question") or "").strip()
                d = (rec.get("document") or "").strip()
                if not q or not d:
                    continue
                qid = rec.get("question_id", rec.get("id", None))
                if qid is None:
                    qid = f"auto_{auto_idx}"; auto_idx += 1
                try: qid = str(qid)
                except Exception: qid = f"auto_{auto_idx}"; auto_idx += 1
                try: n_tok = len(tokenizer(d, add_special_tokens=False).input_ids)
                except Exception: n_tok = 0
                if n_tok > max_doc_tokens:
                    continue
                cands: List[str] = []
                rs = rec.get("responses")
                if isinstance(rs, list):
                    for s in rs:
                        if isinstance(s, str) and s.strip():
                            cands.append(s.strip())
                r_single = rec.get("response")
                if (not cands) and isinstance(r_single, str) and r_single.strip():
                    cands = [r_single.strip()]
                if not cands:
                    continue
                if group_by_question:
                    self.recs.append((qid, q, d, cands))
                else:
                    if explode:
                        for a in cands:
                            self.recs.append((qid, q, d, a))
                    else:
                        self.recs.append((qid, q, d, cands[0]))

    def __len__(self): return len(self.recs)
    def __getitem__(self, idx): return self.recs[idx]

def collate_identity(batch): return batch

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
    if hasattr(pkv, "layers"): return len(pkv.layers)
    return len(pkv)

def pkv_get(pkv, idx: int):
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return pkv.key_cache[idx], pkv.value_cache[idx]
    if hasattr(pkv, "layers"):
        layer = pkv.layers[idx]
        return layer.keys, layer.values
    return pkv[idx]

def dc_from_subset(pkv_src, layer_indices: List[int]) -> DynamicCache:
    dc = DynamicCache()
    for li in layer_indices:
        k, v = pkv_get(pkv_src, li)
        dc.update(k, v, li)
    return dc

def _get_inner_model(m):
    """Return the decoder backbone that owns `.layers` (robust across wrappers).

    Unwraps DDP/Accelerate and PEFT, then tries common attributes; finally scans
    child modules to find the first module that exposes a ModuleList `layers`.
    """
    # unwrap DDP/Accelerate
    if hasattr(m, "module"):
        m = m.module
    # unwrap PEFT
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            try:
                m = m.get_base_model()
            except Exception:
                m = getattr(m, "base_model", m)
    except Exception:
        pass

    # quick paths on common attributes
    for attr in ("model", "transformer", "backbone", "base_model", "language_model"):
        if hasattr(m, attr):
            cand = getattr(m, attr)
            # llama/mistral style
            if hasattr(cand, "layers") and isinstance(getattr(cand, "layers", None), nn.ModuleList):
                return cand
            # some models nest under decoder
            if hasattr(cand, "decoder") and hasattr(cand.decoder, "layers") and isinstance(cand.decoder.layers, nn.ModuleList):
                return cand.decoder

    # direct (already the backbone)
    if hasattr(m, "layers") and isinstance(getattr(m, "layers", None), nn.ModuleList):
        return m

    # fallback: scan children to find first module exposing `.layers`
    for child in m.modules():
        if child is m:
            continue
        if hasattr(child, "layers") and isinstance(getattr(child, "layers", None), nn.ModuleList):
            return child

    raise AttributeError("Could not locate inner base model with a .layers attribute")

def _kv_meta_from_model(model_like):
    try: cfg = getattr(model_like, "config", None) or getattr(_get_inner_model(model_like), "config", None)
    except Exception: cfg = getattr(_get_inner_model(model_like), "config", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv = getattr(cfg, "num_key_value_heads", None) or num_heads
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = (hidden // num_heads) if (hidden and num_heads) else None
    try: dtype = next(_get_inner_model(model_like).parameters()).dtype
    except Exception: dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return int(num_kv), int(head_dim), dtype

def _make_empty_kv(batch: int, num_kv: int, head_dim: int, device, dtype):
    shape = (batch, num_kv, 0, head_dim)
    k = torch.empty(shape, device=device, dtype=dtype)
    v = torch.empty(shape, device=device, dtype=dtype)
    return k.contiguous(), v.contiguous()

# ─────────────────────────────────────────────────────────────────────────────
# LoPA용 레이어별 cache_position/position_ids 보정 패치 (동적 오프셋 버전)
#   * 각 레이어의 실제 past_len과 들어오는 cache_position/position_ids의 "시작값"을 비교
#   * off = start - past_len  만큼 빼서, 레이어별 위치를 정렬
#   * 상위 레이어는 0에서, 하위 레이어는 L_sys+L_doc에서 시작하게 됨
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def lopa_cache_position_patch(model, past_key_values):
    inner = _get_inner_model(model)

    # 레이어별 실제 past 길이
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
            li_local = getattr(module, "_lopa_li", 0)
            past_len = getattr(module, "_lopa_past", 0)

            # 우선 cache_position을 기준으로 동적 오프셋 계산
            cp = kwargs.get("cache_position", None)
            pi = kwargs.get("position_ids", None)

            # start 값 가져오기 (cache_position 우선, 없으면 position_ids 사용)
            start_val = None
            if isinstance(cp, torch.Tensor) and cp.numel() > 0:
                start_val = int(cp.view(-1)[0].item())
            elif isinstance(pi, torch.Tensor) and pi.numel() > 0:
                start_val = int(pi.view(-1)[0].item())

            if start_val is not None:
                off = start_val - past_len  # 동적 오프셋
                if off != 0:
                    if isinstance(cp, torch.Tensor):
                        kwargs["cache_position"] = cp - off
                    if isinstance(pi, torch.Tensor):
                        kwargs["position_ids"] = pi - off
            return args, kwargs

        h = layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()
        for layer in inner.layers:
            for attr in ("_lopa_li", "_lopa_past"):
                if hasattr(layer, attr):
                    delattr(layer, attr)

def _build_accelerator(args) -> Accelerator:
    if args.dtype == "bf16": mp = "bf16"
    elif args.dtype == "fp16": mp = "fp16"
    else: mp = "no"

    auto_wrap = None
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        auto_wrap = {"transformer_layer_cls_to_wrap": (LlamaDecoderLayer, MistralDecoderLayer)}
    except Exception:
        pass

    mode = getattr(args, "dist_mode", "ddp")
    if mode == "fsdp" and FullyShardedDataParallelPlugin is not None:
        fsdp = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD", cpu_offload=False,
            limit_all_gathers=True, use_orig_params=True,
            auto_wrap_policy=auto_wrap,
        )
        return Accelerator(mixed_precision=mp, fsdp_plugin=fsdp)
    if mode == "deepspeed" and DeepSpeedPlugin is not None:
        ds = DeepSpeedPlugin(zero_stage=int(getattr(args, "zero_stage", 2)))
        return Accelerator(mixed_precision=mp, deepspeed_plugin=ds)
    return Accelerator(mixed_precision=mp)

def train(args):
    accelerator = _build_accelerator(args)
    set_seed(args.seed)
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_tokenizer, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.dtype == "fp32": dtype = torch.float32
    elif args.dtype == "bf16": dtype = torch.bfloat16
    elif args.dtype == "fp16": dtype = torch.float16
    else:
        dtype = (torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
                 else (torch.float16 if device == "cuda" else torch.float32))

    if args.no_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception: pass
    if args.sdpa_math_only and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception: pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=False, torch_dtype=dtype, cache_dir=args.cache_dir_model
    )

    mistral_token_added = ensure_mistral_special_token(tokenizer, model)
    try:
        mistral_start_id = tokenizer.convert_tokens_to_ids(MISTRAL_ASSIST_START)
        if isinstance(mistral_start_id, list): mistral_start_id = mistral_start_id[0] if mistral_start_id else None
    except Exception:
        mistral_start_id = None

    impl = "eager"
    if accelerator.is_main_process:
        print("[Note] Forcing attn_implementation='eager' for all models (stability mode).")
    for k in ("attn_implementation", "_attn_implementation"):
        try:
            setattr(model.config, k, impl)
            try: setattr(_get_inner_model(model).config, k, impl)
            except Exception: pass
        except Exception: pass

    use_lora = str(args.use_lora).lower() in ("1","true","yes","y")
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lcfg = LoraConfig(r=int(args.lora_r), lora_alpha=int(args.lora_alpha),
                              lora_dropout=float(args.lora_dropout),
                              bias="none", task_type=TaskType.CAUSAL_LM,
                              target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"])
            model = get_peft_model(model, lcfg)
        except Exception as e:
            raise RuntimeError(f"peft init failed: {e}")
    else:
        for p in model.parameters(): p.requires_grad = True

    train_mistral_start_emb = (_is_mistral_template(tokenizer) and (mistral_start_id is not None) and (mistral_start_id >= 0))
    if train_mistral_start_emb:
        try:
            inp_emb = _get_input_embeddings(model)
            inp_emb.weight.requires_grad = True
            if accelerator.is_main_process:
                print(f"[Info] Enabling training for Mistral start token embedding id={int(mistral_start_id)}")
        except Exception as _e:
            if accelerator.is_main_process:
                print(f"[Warn] Failed to enable training for start token embedding: {_e}")

    ds_all = QADataset(args.data_file, tokenizer,
                       max_doc_tokens=int(getattr(args, "max_doc_tokens", 2048)),
                       explode=bool(getattr(args, "explode", True)),
                       group_by_question=bool(getattr(args, "group_by_question", True)),
                       seed=int(args.seed))
    val_size = max(1, int(0.1 * len(ds_all)))
    train_size = max(1, len(ds_all) - val_size)
    train_set, val_set = random_split(ds_all, [train_size, val_size])
    pin_mem = torch.cuda.is_available()
    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                          pin_memory=pin_mem, persistent_workers=True if 4>0 else False,
                          multiprocessing_context="forkserver", collate_fn=collate_identity)
    dl_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=pin_mem, persistent_workers=True if 2>0 else False,
                        multiprocessing_context="forkserver", collate_fn=collate_identity)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model, optim, dl_train, dl_val = accelerator.prepare(model, optim, dl_train, dl_val)

    if accelerator.is_main_process:
        try:
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
        except Exception: dev_name = str(device)
        try: param_dtype = next(model.parameters()).dtype
        except Exception: param_dtype = None
        try: bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        except Exception: bf16_ok = False
        try:
            tf32_matmul = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
            tf32_cudnn  = torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else False
        except Exception:
            tf32_matmul = tf32_cudnn = None
        try:
            flash = torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else None
            mem_e = torch.backends.cuda.mem_efficient_sdp_enabled() if torch.cuda.is_available() else None
            math  = torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else None
        except Exception:
            flash = mem_e = math = None
        print("[Env] torch:", torch.__version__, "cuda:", torch.version.cuda)
        print("[Env] device:", dev_name)
        print("[Env] requested dtype:", dtype, "| actual param dtype:", param_dtype, "| bf16_support:", bf16_ok)
        print("[Env] TF32 matmul:", tf32_matmul, "cudnn:", tf32_cudnn)
        print("[Env] SDPA flags → flash:", flash, "mem_efficient:", mem_e, "math:", math)

    total_train_steps = args.epochs * max(1, len(dl_train))
    warmup_steps = args.warmup_steps if int(args.warmup_steps) > 0 else int(args.warmup_ratio * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optim,
                                                num_warmup_steps=max(0, warmup_steps),
                                                num_training_steps=max(1, total_train_steps))

    system_prompt = "You are a helpful assistant that answers questions based on the given document. "
    K = max(0, int(args.prefill_layers))
    debug_print_done = False

    def compute_loss_on_sample(q: str, d: str, resp: str, debug: bool = False) -> torch.Tensor:
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)
        sys_only   = tokens_from_messages(tokenizer, [{"role":"system", "content":system_prompt}], device, add_generation_prompt=False)

        L_sys = sys_only.size(1); L_all = ids_phase1.size(1); L_doc = L_all - L_sys
        assert L_doc > 0

        inner = _get_inner_model(model)
        full_layers: nn.ModuleList = inner.layers
        n_layers = len(full_layers)
        K_eff = max(0, min(K, n_layers))
        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        with torch.no_grad():
            out_sys_low = inner(input_ids=sys_only, attention_mask=torch.ones_like(sys_only), use_cache=True, return_dict=True)
        pkv_sys_low = out_sys_low.past_key_values
        inner.layers = full_layers

        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        dc_low_in = dc_from_subset(pkv_sys_low, list(range(K_eff))) if K_eff > 0 else DynamicCache()
        with torch.no_grad():
            out_low = inner(input_ids=ids_phase1[:, L_sys:], past_key_values=dc_low_in,
                            attention_mask=None, use_cache=True, return_dict=True)
        pkv_low = out_low.past_key_values
        inner.layers = full_layers

        combined = DynamicCache()
        num_kv, head_dim, kv_dtype = _kv_meta_from_model(model)
        for li in range(n_layers):
            if li < K_eff:
                k_sys, v_sys = pkv_get(pkv_sys_low, li)
                k_sys_slice = k_sys[:, :, :L_sys, :]
                v_sys_slice = v_sys[:, :, :L_sys, :]
                k_low, v_low = pkv_get(pkv_low, li)
                k_doc = k_low[:, :, -L_doc:, :]
                v_doc = v_low[:, :, -L_doc:, :]
                combined.update(torch.cat([k_sys_slice, k_doc], dim=2).contiguous(),
                                torch.cat([v_sys_slice, v_doc], dim=2).contiguous(), li)
            else:
                k_empty, v_empty = _make_empty_kv(1, num_kv, head_dim, device, kv_dtype)
                combined.update(k_empty, v_empty, li)

        msgs_ass = msgs + [{"role": "assistant", "content": resp}]
        full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
        assistant_ids = full_ids[:, ids_hdr.size(1):]
        if assistant_ids.numel() == 0:
            return torch.tensor(float('nan'), device=device)

        hdr_tail = ids_hdr[:, L_all:]
        if hdr_tail.numel() > 0:
            seed = hdr_tail
        else:
            tok_id = tokenizer.convert_tokens_to_ids(MISTRAL_ASSIST_START)
            seed = torch.tensor([[int(tok_id)]], device=device, dtype=ids_hdr.dtype) if tok_id is not None else ids_phase1[:, -1:]
        inp = torch.cat([seed, assistant_ids], dim=1)
        lab = inp.clone(); lab[:, :seed.size(1)] = -100
        attn_mask = None

        def _forward_hf_loss():
            # 동적 오프셋 패치 적용
            with lopa_cache_position_patch(model, combined):
                out_local = model(input_ids=inp, past_key_values=combined,
                                  attention_mask=attn_mask, labels=lab,
                                  use_cache=True, return_dict=True)
            return out_local.loss if out_local.loss is not None else None

        if debug and accelerator.is_main_process:
            try:
                s_no_hdr = apply_chat_template(tokenizer, msgs, add_generation_prompt=False)
                s_with_hdr = apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
                s_full = apply_chat_template(tokenizer, msgs_ass, add_generation_prompt=False)
                header_len = ids_hdr.size(1) - L_all
                seed_dec = tokenizer.decode(seed[0], skip_special_tokens=False)
                assist_dec = tokenizer.decode(assistant_ids[0][: min(256, assistant_ids.size(1))], skip_special_tokens=False)
                print("\n===== DEBUG: First Training Sample (epoch=1, sample=1) =====")
                print(f"Model layers={n_layers} | K_eff={K_eff}")
                print(f"Lengths: L_sys={L_sys}, L_doc={L_doc}, header={header_len}, assist={assistant_ids.size(1)}")
                print(f"Seed tokens len={seed.size(1)} | decoded: {seed_dec[:160].replace('\n',' ')}")
                print(f"Assistant head: {assist_dec.replace('\n',' ')[:240]}")
                print("-- Combined KV past lengths --")
                for li in range(n_layers):
                    k_comb, _ = pkv_get(combined, li)
                    print(f"layer {li:02d}: past_seq={int(k_comb.shape[2])}")
                print("==========================================================\n")
            except Exception as e:
                print(f"[Debug print error] {e}")

        loss_val = _forward_hf_loss()
        if (loss_val is None) or (not torch.isfinite(loss_val)):
            flash0 = mem0 = math0 = None
            if torch.cuda.is_available():
                try:
                    flash0 = torch.backends.cuda.flash_sdp_enabled()
                    mem0 = torch.backends.cuda.mem_efficient_sdp_enabled()
                    math0 = torch.backends.cuda.math_sdp_enabled()
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception: pass
            loss_val = _forward_hf_loss()
            if torch.cuda.is_available() and None not in (flash0, mem0, math0):
                try:
                    torch.backends.cuda.enable_flash_sdp(bool(flash0))
                    torch.backends.cuda.enable_mem_efficient_sdp(bool(mem0))
                    torch.backends.cuda.enable_math_sdp(bool(math0))
                except Exception: pass
        return loss_val if loss_val is not None else torch.zeros((), device=device, dtype=torch.float32)

    def _tile_cache_for_batch(pkv_in: DynamicCache, batch: int) -> DynamicCache:
        n_layers_local = pkv_len(pkv_in)
        dc = DynamicCache()
        for li in range(n_layers_local):
            k, v = pkv_get(pkv_in, li)
            k_rep = k.repeat(batch, 1, 1, 1).contiguous()
            v_rep = v.repeat(batch, 1, 1, 1).contiguous()
            dc.update(k_rep, v_rep, li)
        return dc

    def compute_loss_on_group(qid: str, q: str, d: str, responses: List[str], debug: bool = False) -> torch.Tensor:
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)
        sys_only   = tokens_from_messages(tokenizer, [{"role":"system", "content":system_prompt}], device, add_generation_prompt=False)

        L_sys = sys_only.size(1); L_all = ids_phase1.size(1); L_doc = L_all - L_sys
        assert L_doc > 0

        inner = _get_inner_model(model)
        full_layers: nn.ModuleList = inner.layers
        n_layers_local = len(full_layers)
        K_eff = max(0, min(K, n_layers_local))
        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        with torch.no_grad():
            out_sys_low = inner(input_ids=sys_only, attention_mask=torch.ones_like(sys_only),
                                use_cache=True, return_dict=True)
        pkv_sys_low = out_sys_low.past_key_values
        dc_low_in = dc_from_subset(pkv_sys_low, list(range(K_eff))) if K_eff > 0 else DynamicCache()
        with torch.no_grad():
            out_low = inner(input_ids=ids_phase1[:, L_sys:], past_key_values=dc_low_in,
                            attention_mask=None, use_cache=True, return_dict=True)
        pkv_low = out_low.past_key_values
        inner.layers = full_layers

        combined = DynamicCache()
        num_kv, head_dim, kv_dtype = _kv_meta_from_model(model)
        for li in range(n_layers_local):
            if li < K_eff:
                k_sys, v_sys = pkv_get(pkv_sys_low, li)
                k_sys_slice = k_sys[:, :, :L_sys, :]
                v_sys_slice = v_sys[:, :, :L_sys, :]
                k_low, v_low = pkv_get(pkv_low, li)
                k_doc = k_low[:, :, -L_doc:, :]
                v_doc = v_low[:, :, -L_doc:, :]
                combined.update(torch.cat([k_sys_slice, k_doc], dim=2).contiguous(),
                                torch.cat([v_sys_slice, v_doc], dim=2).contiguous(), li)
            else:
                k_empty, v_empty = _make_empty_kv(1, num_kv, head_dim, device, kv_dtype)
                combined.update(k_empty, v_empty, li)

        hdr_tail = ids_hdr[:, L_all:]
        seed = hdr_tail if hdr_tail.numel() > 0 else ids_phase1[:, -1:]
        seed_len = seed.size(1)

        assist_list: List[torch.Tensor] = []
        for resp in responses:
            if not isinstance(resp, str) or not resp.strip():
                continue
            msgs_ass = msgs + [{"role": "assistant", "content": resp}]
            full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
            a = full_ids[:, ids_hdr.size(1):]
            if a.numel() > 0: assist_list.append(a)
        G = len(assist_list)
        if G == 0: return torch.tensor(float('nan'), device=device)

        a_lens = [int(x.size(1)) for x in assist_list]
        losses = []
        for a in assist_list:
            try:
                inp_i = torch.cat([seed, a], dim=1)
                labels_i = inp_i.clone(); labels_i[:, :seed_len] = -100
                with lopa_cache_position_patch(model, combined):
                    out_i = model(input_ids=inp_i, past_key_values=combined,
                                  attention_mask=None, use_cache=True, return_dict=True, labels=labels_i)
                if out_i.loss is None or not torch.isfinite(out_i.loss): continue
                losses.append(out_i.loss)
            except RuntimeError:
                if torch.cuda.is_available():
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                continue
        if not losses: return torch.tensor(float('nan'), device=device)
        group_loss = torch.stack(losses, dim=0).mean()

        if debug and accelerator.is_main_process:
            try:
                a_max = max(a_lens) if a_lens else 0
                print(f"[debug-group] qid={qid} | responses={G} | seed_len={seed_len} | a_max={a_max} (sequential)")
            except Exception: pass
        return group_loss

    best_loss = float("inf")
    best_dir = Path(args.save_best_dir); best_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if accelerator.is_main_process and args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=f"lopa_pure-K{K}", config=vars(args))
        except Exception: run = None

    def _iter_items(batch): return list(batch) if isinstance(batch, list) else [batch]

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        train_nan_skipped = 0
        train_iter = tqdm(dl_train, desc=f"Epoch {epoch} [train]", leave=False) if accelerator.is_main_process else dl_train
        for bidx, batch in enumerate(train_iter):
            items = _iter_items(batch)
            if not items: continue
            loss_accum = 0.0; loss_vals = []; valid_in_step = 0
            for iidx, it in enumerate(items):
                do_debug = (epoch == 1) and (not debug_print_done) and (bidx == 0) and (iidx == 0)
                try:
                    if len(it) == 4 and isinstance(it[3], list):
                        qid, q, d, rs = it
                        loss_i = compute_loss_on_group(str(qid), q, d, rs, debug=do_debug)
                    elif len(it) == 4:
                        qid, q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r, debug=do_debug)
                    else:
                        q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r, debug=do_debug)
                except Exception as _e:
                    print(f"[Warn] batch item parse failed: {_e}")
                    continue
                if do_debug: debug_print_done = True
                if not torch.isfinite(loss_i):
                    train_nan_skipped += 1; continue
                loss_accum += float(loss_i.detach().item()); loss_vals.append(float(loss_i.detach().item()))
                accelerator.backward(loss_i / max(1, args.batch_size))
                if train_mistral_start_emb:
                    try:
                        emb = _get_input_embeddings(model)
                        g = getattr(getattr(emb, "weight", None), "grad", None)
                        if g is not None:
                            with torch.no_grad():
                                mask = torch.zeros_like(g)
                                mask[int(mistral_start_id)].fill_(1.0)
                                g.mul_(mask)
                    except Exception: pass
                valid_in_step += 1
            if valid_in_step > 0:
                if float(args.grad_clip) and args.grad_clip > 0:
                    try: accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    except Exception: pass
                optim.step(); scheduler.step()
                tr_loss_sum += loss_accum; tr_count += valid_in_step
            optim.zero_grad(set_to_none=True)
            if accelerator.is_main_process and run is not None:
                try:
                    step_loss = (sum(loss_vals) / max(1, len(loss_vals))) if loss_vals else None
                    try: lr = float(scheduler.get_last_lr()[0])
                    except Exception: lr = optim.param_groups[0]["lr"]
                    payload = {"step": global_step, "lr": lr, "train/nan_skipped_step": (len(items) - valid_in_step)}
                    if step_loss is not None: payload["train/step_loss"] = step_loss
                    run.log(payload)
                except Exception: pass
            global_step += 1
        tr_avg = tr_loss_sum / max(1, tr_count)

        model.eval()
        va_loss_sum, va_count = 0.0, 0
        val_nan_skipped = 0
        with torch.no_grad():
            for batch in dl_val:
                items = _iter_items(batch)
                if not items: continue
                for it in items:
                    try:
                        if len(it) == 4 and isinstance(it[3], list):
                            qid, q, d, rs = it
                            loss_i = compute_loss_on_group(str(qid), q, d, rs, debug=False)
                        elif len(it) == 4:
                            qid, q, d, r = it
                            loss_i = compute_loss_on_sample(q, d, r, debug=False)
                        else:
                            q, d, r = it
                            loss_i = compute_loss_on_sample(q, d, r, debug=False)
                    except Exception:
                        continue
                    if not torch.isfinite(loss_i):
                        val_nan_skipped += 1; continue
                    va_loss_sum += float(loss_i.detach().item()); va_count += 1
        va_avg = va_loss_sum / max(1, va_count)

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] train_avg={tr_avg:.6f} | valid_avg={va_avg:.6f} | skipped(nan): train={train_nan_skipped}, valid={val_nan_skipped}")
            if run is not None:
                try:
                    run.log({"epoch": epoch, "train/avg": tr_avg, "valid/avg": va_avg,
                             "train/nan_skipped": train_nan_skipped, "valid/nan_skipped": val_nan_skipped})
                except Exception: pass

        if va_avg < best_loss:
            best_loss = va_avg
            if accelerator.is_main_process:
                for p in best_dir.glob("*"):
                    try:
                        if p.is_file(): p.unlink()
                        elif p.is_dir():
                            import shutil; shutil.rmtree(p)
                    except Exception: pass
                base_dir = best_dir / "base"; base_dir.mkdir(parents=True, exist_ok=True)
                try:
                    to_unwrap = accelerator.unwrap_model(model)
                    is_peft = False
                    try:
                        from peft import PeftModel  # type: ignore
                        is_peft = isinstance(to_unwrap, PeftModel)
                    except Exception: is_peft = False
                    if use_lora and is_peft:
                        base_clean = AutoModelForCausalLM.from_pretrained(
                            args.model_name, trust_remote_code=False, torch_dtype=dtype, cache_dir=args.cache_dir_model
                        )
                        _ = ensure_mistral_special_token(tokenizer, base_clean)
                        try:
                            if _is_mistral_template(tokenizer):
                                tok_id = tokenizer.convert_tokens_to_ids(MISTRAL_ASSIST_START)
                                if tok_id is not None and tok_id >= 0:
                                    src_emb = _get_input_embeddings(to_unwrap)
                                    dst_emb = _get_input_embeddings(base_clean)
                                    with torch.no_grad():
                                        if tok_id < dst_emb.weight.size(0) and tok_id < src_emb.weight.size(0):
                                            dst_emb.weight.data[tok_id].copy_(src_emb.weight.data[tok_id])
                        except Exception: pass
                        _strip_automap(base_clean.config)
                        base_clean.save_pretrained(base_dir, safe_serialization=True)
                    else:
                        to_save = to_unwrap
                        _strip_automap(to_save.config)
                        to_save.save_pretrained(base_dir, safe_serialization=True)
                except Exception as e:
                    raise RuntimeError(f"Failed saving base backbone: {e}")
                if use_lora:
                    try:
                        (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir / "lora")
                    except Exception as e:
                        print(f"[Warn] failed to save LoRA: {e}")
                try: tokenizer.save_pretrained(best_dir)
                except Exception: pass
                try:
                    if getattr(model, "generation_config", None) is not None:
                        model.generation_config.to_json_file(str(best_dir / "generation_config.json"))
                except Exception: pass
                print(f"[Best] Saved to {best_dir} (val={best_loss:.6f})")
                try:
                    helper_src = Path(__file__).parent / "infer_lopa_pure.py"
                    if helper_src.is_file():
                        import shutil; shutil.copy2(helper_src, best_dir / "infer_lopa_pure.py")
                except Exception: pass

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.push_to_hub and args.hf_repo_id:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN not set; export HF_TOKEN=... to push to hub")
        create_repo(args.hf_repo_id, exist_ok=True, private=args.private_repo, token=token)
        upload_folder(repo_id=args.hf_repo_id, folder_path=str(best_dir),
                      token=token, commit_message="LoPA pure trainer upload", allow_patterns=["*"])
        print(f"✅ Uploaded to hub: {args.hf_repo_id}")

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()

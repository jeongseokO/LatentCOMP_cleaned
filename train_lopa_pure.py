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
    # explode default True; add --no_explode to disable
    p.add_argument("--explode", action="store_true", default=True)
    p.add_argument("--no_explode", dest="explode", action="store_false")
    # group by question_id: yields one sample per record with list of responses
    p.add_argument("--group_by_question", action="store_true", default=True)
    p.add_argument("--no_group_by_question", dest="group_by_question", action="store_false")
    # HF Hub
    p.add_argument("--hf_repo_id", type=str, default=None, help="repo id to upload best/ (e.g., user/repo)")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")
    # distributed / sharding
    p.add_argument("--dist_mode", type=str, choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    p.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage (if dist_mode=deepspeed)")
    return p


class QADataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_doc_tokens: int = 2048,
        explode: bool = True,
        group_by_question: bool = True,
        seed: int = 42,
    ):
        """Dataset for records with `responses` list[str] or `response` str.

        - group_by_question=True(default): one sample per record, value=(question_id, question, document, [responses])
        - if group_by_question=False and explode=True: expand responses to individual samples
        - if group_by_question=False and explode=False: only first non-empty response
        - Filters empty question/document and overly long documents
        """
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
                # question id (prefer explicit; fallback to auto index)
                qid = rec.get("question_id", rec.get("id", None))
                if qid is None:
                    qid = f"auto_{auto_idx}"
                    auto_idx += 1
                try:
                    qid = str(qid)
                except Exception:
                    qid = f"auto_{auto_idx}"; auto_idx += 1
                # filter by doc length
                try:
                    n_tok = len(tokenizer(d, add_special_tokens=False).input_ids)
                except Exception:
                    n_tok = 0
                if n_tok > max_doc_tokens:
                    continue

                # collect candidates
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
                    # one record with all responses
                    self.recs.append((qid, q, d, cands))
                else:
                    if explode:
                        for a in cands:
                            self.recs.append((qid, q, d, a))
                    else:
                        # fall back to the first candidate only
                        self.recs.append((qid, q, d, cands[0]))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        return self.recs[idx]


def collate_identity(batch):
    """Return batch as-is (list of samples). Avoids default tuple-of-lists transforms
    for mixed Python types (str, list[str]). Keeps our parsing simple and robust.
    """
    return batch


def build_messages(system: str, document: str, question: str, include_query: bool = True):
    user = f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    """Render chat with robust fallback across templates (Llama3 vs Mistral).

    - Prefer tokenizer.apply_chat_template(..., add_generation_prompt=...)
    - If that signature is unsupported, inspect tokenizer.chat_template and only append
      an assistant header for Llama-3 style; for Mistral/INST style, append nothing.
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
            else:  # Mistral/INST or unknown → do not append explicit header
                s += ""
        return s


def tokens_from_messages(tokenizer, messages, device, add_generation_prompt=False):
    s = apply_chat_template(tokenizer, messages, add_generation_prompt)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def pkv_len(pkv) -> int:
    if hasattr(pkv, "key_cache"):
        return len(pkv.key_cache)
    if hasattr(pkv, "layers"):
        return len(pkv.layers)
    try:
        return len(pkv)
    except Exception:
        return 0


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
    """Return the base model object that holds .layers (e.g., LlamaModel/MistralModel).
    Handles Accelerate wrappers and PEFT.
    """
    # unwrap accelerate
    if hasattr(m, "module"):
        m = m.module
    # unwrap peft
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
    # common HF layouts
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model
    if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
        return m.transformer
    raise AttributeError("Could not locate inner base model with a .layers attribute")


def _kv_meta_from_model(model_like):
    """Return (num_kv_heads, head_dim, dtype) inferred from model config/params.
    Assumes uniform heads across layers (true for Llama/Mistral).
    """
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


def _build_accelerator(args) -> Accelerator:
    # derive mixed precision from dtype option
    if args.dtype == "bf16":
        mp = "bf16"
    elif args.dtype == "fp16":
        mp = "fp16"
    else:
        mp = "no"

    # Optional FSDP auto-wrap policy for transformer decoders
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
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
            limit_all_gathers=True,
            use_orig_params=True,
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

    # tokenizer
    tok_src = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        cache_dir=args.cache_dir_tokenizer,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Ensure Mistral assistant-start special token exists and resize model later

    # model dtype selection
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = (
            torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
                torch.float16 if device == "cuda" else torch.float32
            )
        )
    # optional: disable TF32 / fix SDPA kernel for stability
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=False,
        torch_dtype=dtype,
        cache_dir=args.cache_dir_model,
    )
    # If Mistral template, add special token and resize embeddings
    _ = ensure_mistral_special_token(tokenizer, model)
    # attn backend — force eager for all models (stability first)
    impl = "eager"
    if accelerator.is_main_process:
        print("[Note] Forcing attn_implementation='eager' for all models (stability mode).")
    for k in ("attn_implementation", "_attn_implementation"):
        try:
            setattr(model.config, k, impl)
            try:
                setattr(_get_inner_model(model).config, k, impl)
            except Exception:
                pass
        except Exception:
            pass

    # LoRA
    use_lora = str(args.use_lora).lower() in ("1", "true", "yes", "y")
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lcfg = LoraConfig(
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lcfg)
            # base frozen; LoRA params trainable
        except Exception as e:
            raise RuntimeError(f"peft not available or failed to init LoRA: {e}")
    else:
        for p in model.parameters():
            p.requires_grad = True

    # data
    ds_all = QADataset(
        args.data_file,
        tokenizer,
        max_doc_tokens=int(getattr(args, "max_doc_tokens", 2048)),
        explode=bool(getattr(args, "explode", True)),
        group_by_question=bool(getattr(args, "group_by_question", True)),
        seed=int(args.seed),
    )
    val_size = max(1, int(0.1 * len(ds_all)))
    train_size = max(1, len(ds_all) - val_size)
    train_set, val_set = random_split(ds_all, [train_size, val_size])
    pin_mem = torch.cuda.is_available()
    dl_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_mem,
        persistent_workers=True if 4 > 0 else False,
        multiprocessing_context="forkserver",
        collate_fn=collate_identity,
    )
    dl_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_mem,
        persistent_workers=True if 2 > 0 else False,
        multiprocessing_context="forkserver",
        collate_fn=collate_identity,
    )

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Prepare with accelerate
    model, optim, dl_train, dl_val = accelerator.prepare(model, optim, dl_train, dl_val)
    # --- Diagnostics: print compute dtype / backend flags (main process only)
    if accelerator.is_main_process:
        try:
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
        except Exception:
            dev_name = str(device)
        try:
            param_dtype = next(model.parameters()).dtype
        except Exception:
            param_dtype = None
        try:
            bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        except Exception:
            bf16_ok = False
        try:
            tf32_matmul = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
            tf32_cudnn  = torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else False
        except Exception:
            tf32_matmul, tf32_cudnn = None, None
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
    # Scheduler: cosine with warmup
    total_train_steps = args.epochs * max(1, len(dl_train))
    warmup_steps = args.warmup_steps if int(args.warmup_steps) > 0 else int(args.warmup_ratio * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=max(0, warmup_steps),
        num_training_steps=max(1, total_train_steps),
    )

    system_prompt = "You are a helpful assistant that answers questions based on the given document. "
    K = max(0, int(args.prefill_layers))
    debug_print_done = False  # print detailed feed only for epoch1/sample1

    def compute_loss_on_sample(q: str, d: str, resp: str, debug: bool = False) -> torch.Tensor:
        # messages
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)  # [1, L_sys+doc]
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)   # [1, L_sys+doc+header]
        sys_only   = tokens_from_messages(tokenizer, [{"role":"system", "content":system_prompt}], device, add_generation_prompt=False)

        L_sys = sys_only.size(1)
        L_all = ids_phase1.size(1)
        L_doc = L_all - L_sys
        assert L_doc > 0

        # Phase-1: system-only prefill (only lower-K layers)
        inner = _get_inner_model(model)
        full_layers: nn.ModuleList = inner.layers
        n_layers = len(full_layers)
        K_eff = max(0, min(K, n_layers))
        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        with torch.no_grad():
            out_sys_low = inner(
                input_ids=sys_only,
                attention_mask=torch.ones_like(sys_only),
                use_cache=True,
                return_dict=True,
            )
        pkv_sys_low = out_sys_low.past_key_values
        inner.layers = full_layers

        # Phase-1: doc pass only for lower-K layers
        full_layers: nn.ModuleList = inner.layers
        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        dc_low_in = dc_from_subset(pkv_sys_low, list(range(K_eff))) if K_eff > 0 else DynamicCache()
        with torch.no_grad():
            out_low = inner(
                input_ids=ids_phase1[:, L_sys:],
                past_key_values=dc_low_in,
                attention_mask=None,
                use_cache=True,
                return_dict=True,
            )
        pkv_low = out_low.past_key_values
        inner.layers = full_layers

        # Combine caches: lower => sys+doc; upper => no sys/no doc (start empty and grow from header)
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
                k_cat = torch.cat([k_sys_slice, k_doc], dim=2).contiguous()
                v_cat = torch.cat([v_sys_slice, v_doc], dim=2).contiguous()
                combined.update(k_cat, v_cat, li)
            else:
                # Upper layers start with empty past (no sys/doc)
                k_empty, v_empty = _make_empty_kv(1, num_kv, head_dim, device, kv_dtype)
                combined.update(k_empty, v_empty, li)

        # Build assistant continuation ids from response
        msgs_ass = msgs + [{"role": "assistant", "content": resp}]
        full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
        # assistant ids = after header
        assistant_ids = full_ids[:, ids_hdr.size(1):]
        if assistant_ids.numel() == 0:
            # No target tokens → skip this sample by returning NaN (caller skips non-finite losses)
            return torch.tensor(float('nan'), device=device)

        # Do NOT push header into past during training.
        # Instead, include header (if any) or the explicit Mistral assist-start token as the first input token(s),
        # and mask them out so the first predicted target is A0.
        hdr_tail = ids_hdr[:, L_all:]  # header-only sequence; with Mistral we append <Mistral_start> so len>=1
        if hdr_tail.numel() > 0:
            seed = hdr_tail
        else:
            # Fallback: explicit special token id (in case template path missed)
            tok_id = tokenizer.convert_tokens_to_ids(MISTRAL_ASSIST_START)
            seed = torch.tensor([[int(tok_id)]], device=device, dtype=ids_hdr.dtype) if tok_id is not None else ids_phase1[:, -1:]
        inp = torch.cat([seed, assistant_ids], dim=1)  # [seed, A0, A1, ...]
        lab = inp.clone()
        # Mask seed tokens from loss
        lab[:, :seed.size(1)] = -100

        # Important: allow per-layer past length mismatch by not forcing a shared 1D mask
        attn_mask = None

        # Use HF standard CLM loss (handles 1-token shift internally). Header is ignored via -100.
        def _forward_hf_loss():
            out_local = model(
                input_ids=inp,
                past_key_values=combined,
                attention_mask=attn_mask,
                labels=lab,
                use_cache=True,
                return_dict=True,
            )
            return out_local.loss if out_local.loss is not None else None

        # Optional detailed debug print for first sample of training
        if debug and accelerator.is_main_process:
            try:
                # Render strings for visibility
                s_no_hdr = apply_chat_template(tokenizer, msgs, add_generation_prompt=False)
                s_with_hdr = apply_chat_template(tokenizer, msgs, add_generation_prompt=True)
                s_full = apply_chat_template(tokenizer, msgs_ass, add_generation_prompt=False)
                header_len = ids_hdr.size(1) - L_all
                seed_dec = tokenizer.decode(seed[0], skip_special_tokens=False)
                assist_dec = tokenizer.decode(assistant_ids[0][: min(256, assistant_ids.size(1))], skip_special_tokens=False)
                print("\n===== DEBUG: First Training Sample (epoch=1, sample=1) =====")
                print(f"Model: {getattr(model.config, 'name_or_path', None)} | Layers={n_layers} | K_eff={K_eff}")
                print(f"Doc/Query lengths (tokens): L_sys={L_sys}, L_doc={L_doc}, header={header_len}, assist={assistant_ids.size(1)}")
                # Token id snapshots
                try:
                    doc_head = min(16, L_doc)
                    print(f"sys_only ids[:16]: {sys_only[0, :min(16, L_sys)].tolist()}")
                    print(f"doc part ids[:16]: {ids_phase1[0, L_sys:L_sys+doc_head].tolist()}")
                    print(f"header ids[:16]: {hdr_tail[0, :min(16, hdr_tail.size(1))].tolist() if hdr_tail.numel()>0 else []}")
                    print(f"seed ids[:16]: {seed[0, :min(16, seed.size(1))].tolist()}")
                    print(f"assistant ids head[:16]: {assistant_ids[0, :min(16, assistant_ids.size(1))].tolist()}")
                except Exception:
                    pass
                print("-- system prompt (full) --\n" + system_prompt)
                # Print lengths and chunk long outputs to avoid log line truncation
                print(f"Rendered lengths (chars): no_hdr={len(s_no_hdr)}, with_hdr={len(s_with_hdr)}, full={len(s_full)}")
                def _print_chunked(label: str, text: str, chunk: int = 4000):
                    print(f"-- {label} --")
                    for i in range(0, len(text), chunk):
                        print(text[i:i+chunk])
                _print_chunked("rendered without header (full)", s_no_hdr)
                _print_chunked("rendered with header (full)", s_with_hdr)
                _print_chunked("full target rendered (full)", s_full)
                # Also dump to files to guarantee full capture
                try:
                    out_dir = Path(getattr(args, "save_best_dir", "./_best_ckpt")) / "debug_first_sample"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "render_no_header.txt").write_text(s_no_hdr, encoding="utf-8")
                    (out_dir / "render_with_header.txt").write_text(s_with_hdr, encoding="utf-8")
                    (out_dir / "full_target.txt").write_text(s_full, encoding="utf-8")
                    print(f"[debug] wrote full renders to {out_dir}")
                except Exception as _e:
                    print(f"[debug] file dump failed: {_e}")
                print(f"Seed tokens len={seed.size(1)} | decoded: {seed_dec[:160].replace('\n', ' ')}")
                print(f"Assistant target len={assistant_ids.size(1)} | decoded head: {assist_dec.replace('\n',' ')[:240]}")
                # Combined cache lengths per layer
                print("-- Combined KV past lengths per layer --")
                for li in range(n_layers):
                    k_comb, _ = pkv_get(combined, li)
                    print(f"layer {li:02d}: past_seq={int(k_comb.shape[2])}")
                print("Attention mask during label loss: None (per-layer past handled internally)")
                print(f"Label -100 count (masked seed): {int((lab == -100).sum().item())}")
                print("==========================================================\n")
            except Exception as e:
                print(f"[Debug print error] {e}")

        loss_val = _forward_hf_loss()
        if (loss_val is None) or (not torch.isfinite(loss_val)):
            # Fallback: force math SDPA for this batch
            flash0 = mem0 = math0 = None
            if torch.cuda.is_available():
                try:
                    flash0 = torch.backends.cuda.flash_sdp_enabled()
                    mem0 = torch.backends.cuda.mem_efficient_sdp_enabled()
                    math0 = torch.backends.cuda.math_sdp_enabled()
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass
            loss_val = _forward_hf_loss()
            if torch.cuda.is_available() and None not in (flash0, mem0, math0):
                try:
                    torch.backends.cuda.enable_flash_sdp(bool(flash0))
                    torch.backends.cuda.enable_mem_efficient_sdp(bool(mem0))
                    torch.backends.cuda.enable_math_sdp(bool(math0))
                except Exception:
                    pass
        return loss_val if loss_val is not None else torch.zeros((), device=device, dtype=torch.float32)

    def _tile_cache_for_batch(pkv_in: DynamicCache, batch: int) -> DynamicCache:
        """Repeat past_key_values along batch dim to match group size."""
        n_layers_local = pkv_len(pkv_in)
        dc = DynamicCache()
        for li in range(n_layers_local):
            k, v = pkv_get(pkv_in, li)
            # k/v: [1, heads, past, dim] → [batch, heads, past, dim]
            k_rep = k.repeat(batch, 1, 1, 1).contiguous()
            v_rep = v.repeat(batch, 1, 1, 1).contiguous()
            dc.update(k_rep, v_rep, li)
        return dc

    def compute_loss_on_group(qid: str, q: str, d: str, responses: List[str], debug: bool = False) -> torch.Tensor:
        # messages and ids
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)
        sys_only   = tokens_from_messages(tokenizer, [{"role":"system", "content":system_prompt}], device, add_generation_prompt=False)

        L_sys = sys_only.size(1)
        L_all = ids_phase1.size(1)
        L_doc = L_all - L_sys
        assert L_doc > 0

        # Prefill (system then doc for lower K) — optimized to lower-K only
        inner = _get_inner_model(model)
        full_layers: nn.ModuleList = inner.layers
        n_layers_local = len(full_layers)
        K_eff = max(0, min(K, n_layers_local))
        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        with torch.no_grad():
            out_sys_low = inner(input_ids=sys_only, attention_mask=torch.ones_like(sys_only), use_cache=True, return_dict=True)
        pkv_sys_low = out_sys_low.past_key_values
        dc_low_in = dc_from_subset(pkv_sys_low, list(range(K_eff))) if K_eff > 0 else DynamicCache()
        with torch.no_grad():
            out_low = inner(
                input_ids=ids_phase1[:, L_sys:],
                past_key_values=dc_low_in,
                attention_mask=None,
                use_cache=True,
                return_dict=True,
            )
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

        # Seed tokens (header tail or fallback)
        hdr_tail = ids_hdr[:, L_all:]
        if hdr_tail.numel() > 0:
            seed = hdr_tail  # [1, H]
        else:
            tok_id = tokenizer.convert_tokens_to_ids(MISTRAL_ASSIST_START)
            seed = torch.tensor([[int(tok_id)]], device=device, dtype=ids_hdr.dtype) if tok_id is not None else ids_phase1[:, -1:]
        seed_len = seed.size(1)

        # Build assistant ids per response
        assist_list: List[torch.Tensor] = []
        for resp in responses:
            if not isinstance(resp, str) or not resp.strip():
                continue
            msgs_ass = msgs + [{"role": "assistant", "content": resp}]
            full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
            a = full_ids[:, ids_hdr.size(1):]
            if a.numel() > 0:
                assist_list.append(a)
        G = len(assist_list)
        if G == 0:
            return torch.tensor(float('nan'), device=device)

        # Always compute sequentially per response to minimize memory usage
        a_lens = [int(x.size(1)) for x in assist_list]
        losses = []
        for a in assist_list:
            try:
                inp_i = torch.cat([seed, a], dim=1)  # [1, H+Li]
                labels_i = inp_i.clone()
                labels_i[:, :seed_len] = -100
                out_i = model(input_ids=inp_i, past_key_values=combined, attention_mask=None, use_cache=True, return_dict=True, labels=labels_i)
                if out_i.loss is None or not torch.isfinite(out_i.loss):
                    continue
                losses.append(out_i.loss)
            except RuntimeError as e:
                # Best-effort OOM mitigation per response
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue
        if not losses:
            return torch.tensor(float('nan'), device=device)
        group_loss = torch.stack(losses, dim=0).mean()

        if debug and accelerator.is_main_process:
            try:
                a_max = max(a_lens) if a_lens else 0
                print(f"[debug-group] qid={qid} | responses={G} | seed_len={seed_len} | a_max={a_max} (sequential)")
            except Exception:
                pass
        return group_loss

    # training loop
    best_loss = float("inf")
    best_dir = Path(args.save_best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    # wandb (optional)
    run = None
    if accelerator.is_main_process and args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=f"lopa_pure-K{K}", config=vars(args))
        except Exception:
            run = None

    def _iter_items(batch):
        # With collate_identity, batch is always a list of per-sample items.
        return list(batch) if isinstance(batch, list) else [batch]

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        train_nan_skipped = 0
        train_iter = dl_train
        if accelerator.is_main_process:
            train_iter = tqdm(dl_train, desc=f"Epoch {epoch} [train]", leave=False)
        for bidx, batch in enumerate(train_iter):
            items = _iter_items(batch)
            if not items:
                continue
            loss_accum = 0.0
            loss_vals = []
            valid_in_step = 0
            for iidx, it in enumerate(items):
                do_debug = (epoch == 1) and (not debug_print_done) and (bidx == 0) and (iidx == 0)
                # Accept both grouped and single-response samples
                try:
                    if len(it) == 4 and isinstance(it[3], list):
                        qid, q, d, rs = it
                        loss_i = compute_loss_on_group(str(qid), q, d, rs, debug=do_debug)
                    elif len(it) == 4:
                        qid, q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r, debug=do_debug)
                    else:
                        # Legacy triple (q, d, r)
                        q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r, debug=do_debug)
                except Exception as _e:
                    print(f"[Warn] batch item parse failed: {_e}")
                    continue
                if do_debug:
                    debug_print_done = True
                if not torch.isfinite(loss_i):
                    train_nan_skipped += 1
                    continue
                loss_accum += float(loss_i.detach().item())
                loss_vals.append(float(loss_i.detach().item()))
                accelerator.backward(loss_i / max(1, args.batch_size))
                valid_in_step += 1
            if valid_in_step > 0:
                # grad clip + step
                if float(args.grad_clip) and args.grad_clip > 0:
                    try:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    except Exception:
                        pass
                optim.step(); scheduler.step()
                tr_loss_sum += loss_accum
                tr_count += valid_in_step
            # Always zero grads at step end
            optim.zero_grad(set_to_none=True)
            # step-level logging (per optimizer step)
            if accelerator.is_main_process and run is not None:
                try:
                    step_loss = (sum(loss_vals) / max(1, len(loss_vals))) if loss_vals else None
                    # fetch current lr
                    try:
                        lr = float(scheduler.get_last_lr()[0])
                    except Exception:
                        lr = optim.param_groups[0]["lr"]
                    payload = {"step": global_step, "lr": lr, "train/nan_skipped_step": (len(items) - valid_in_step)}
                    if step_loss is not None:
                        payload["train/step_loss"] = step_loss
                    run.log(payload)
                except Exception:
                    pass
            global_step += 1
        tr_avg = tr_loss_sum / max(1, tr_count)

        # validation
        model.eval()
        va_loss_sum, va_count = 0.0, 0
        val_nan_skipped = 0
        with torch.no_grad():
            for batch in dl_val:
                items = _iter_items(batch)
                if not items:
                    continue
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
                        val_nan_skipped += 1
                        continue
                    va_loss_sum += float(loss_i.detach().item())
                    va_count += 1
        va_avg = va_loss_sum / max(1, va_count)

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] train_avg={tr_avg:.6f} | valid_avg={va_avg:.6f} | skipped(nan): train={train_nan_skipped}, valid={val_nan_skipped}")
            if run is not None:
                try:
                    run.log({
                        "epoch": epoch,
                        "train/avg": tr_avg,
                        "valid/avg": va_avg,
                        "train/nan_skipped": train_nan_skipped,
                        "valid/nan_skipped": val_nan_skipped,
                    })
                except Exception:
                    pass

        if va_avg < best_loss:
            best_loss = va_avg
            if accelerator.is_main_process:
                # Save best
                for p in best_dir.glob("*"):
                    try:
                        if p.is_file(): p.unlink()
                        elif p.is_dir():
                            import shutil; shutil.rmtree(p)
                    except Exception:
                        pass
                # Save current backbone (captures resized embeddings for special tokens) to base/
                base_dir = best_dir / "base"
                base_dir.mkdir(parents=True, exist_ok=True)
                try:
                    inner = _get_inner_model(model)
                    # Try not to save PEFT adapters in base/
                    to_save = inner
                    to_save.save_pretrained(base_dir, safe_serialization=True)
                except Exception as e:
                    raise RuntimeError(f"Failed saving base backbone: {e}")
                # Save LoRA adapter if present
                if use_lora:
                    try:
                        (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir / "lora")
                    except Exception as e:
                        print(f"[Warn] failed to save LoRA: {e}")
                # Save tokenizer + generation config
                try:
                    tokenizer.save_pretrained(best_dir)
                except Exception:
                    pass
                try:
                    if getattr(model, "generation_config", None) is not None:
                        model.generation_config.to_json_file(str(best_dir / "generation_config.json"))
                except Exception:
                    pass
                print(f"[Best] Saved to {best_dir} (val={best_loss:.6f})")
                # Drop a lightweight inference helper next to best/ for convenience
                try:
                    helper_src = Path(__file__).parent / "infer_lopa_pure.py"
                    if helper_src.is_file():
                        import shutil
                        shutil.copy2(helper_src, best_dir / "infer_lopa_pure.py")
                except Exception:
                    pass

        accelerator.wait_for_everyone()

    # Hub upload (optional, main process only)
    if accelerator.is_main_process and args.push_to_hub and args.hf_repo_id:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN not set; export HF_TOKEN=... to push to hub")
        create_repo(args.hf_repo_id, exist_ok=True, private=args.private_repo, token=token)
        upload_folder(
            repo_id=args.hf_repo_id,
            folder_path=str(best_dir),
            token=token,
            commit_message="LoPA pure trainer upload",
            allow_patterns=["*"],
        )
        print(f"✅ Uploaded to hub: {args.hf_repo_id}")


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

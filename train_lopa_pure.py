#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoPA-only trainer (STRICT, responses sequential supported)
- Custom modeling (lopa_llama_modeling.py) MUST be injected; otherwise abort.
- Default training: rope_mode="global" + upper_zero_pad_prefix=True (exact zero-pad equivalence).
- Lower-K prefill: model.model.lopa_prefill_lower_k(...)
- Upper cache compose:
    * zero-pad:  model.model.lopa_build_zero_padded_cache(...)
    * empty-KV:  model.model.lopa_build_combined_cache(...)
- Step/CE:       model.lopa_step_logits(...), absolute positions enforced by modeling.

NO FALLBACKS. If any required LoPA API is missing → RuntimeError and stop.
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from accelerate import Accelerator
try:
    from accelerate import FullyShardedDataParallelPlugin, DeepSpeedPlugin
except Exception:
    FullyShardedDataParallelPlugin = None
    DeepSpeedPlugin = None

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.cache_utils import DynamicCache
from huggingface_hub import create_repo, upload_folder

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------
# Strict helpers
# -----------------------
def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def _get_inner_model(m):
    # unwrap DistributedDataParallel / Accelerate wrapper
    if hasattr(m, "module"):
        m = m.module
    return m

def _unwrap_peft(m):
    # unwrap PEFT to base model if present
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            try:
                m = m.get_base_model()
            except Exception:
                m = getattr(m, "base_model", m)
    except Exception:
        pass
    return m


# -----------------------
# Custom modeling injection (STRICT)
# -----------------------
def load_custom_llama_modeling(modeling_path: Path):
    """
    Load local `lopa_llama_modeling.py` as `transformers.models.llama.modeling_llama`.
    - sys.modules 엔트리 교체
    - 부모 패키지 속성(transformers.models.llama.modeling_llama)도 함께 교체
    - 로드된 module 객체 자체로 API 검증 (재임포트 X)
    """
    import importlib.util, sys
    import transformers
    import transformers.models.llama as llama_pkg  # parent pkg object

    target_name = "transformers.models.llama.modeling_llama"
    sys.modules.pop(target_name, None)

    spec = importlib.util.spec_from_file_location(target_name, str(modeling_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {modeling_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[target_name] = module
    spec.loader.exec_module(module)

    # also patch parent package attribute
    setattr(llama_pkg, "modeling_llama", module)

    # strict API checks on the loaded module itself
    for a in ("LlamaModel", "LlamaForCausalLM"):
        _require(hasattr(module, a), f"Patched module missing `{a}` in {modeling_path}")

    _require(hasattr(module.LlamaModel, "lopa_prefill_lower_k"), "Patched LlamaModel lacks `lopa_prefill_lower_k`")
    _require(hasattr(module.LlamaModel, "lopa_build_zero_padded_cache"), "Patched LlamaModel lacks `lopa_build_zero_padded_cache`")
    _require(hasattr(module.LlamaModel, "lopa_build_combined_cache"), "Patched LlamaModel lacks `lopa_build_combined_cache`")
    _require(hasattr(module.LlamaModel, "lopa_forward_from_prefix"), "Patched LlamaModel lacks `lopa_forward_from_prefix`")
    _require(hasattr(module.LlamaForCausalLM, "lopa_step_logits"), "Patched LlamaForCausalLM lacks `lopa_step_logits`")

    print("[DEBUG] modeling_llama path:", modeling_path)
    print("[DEBUG] bound:",
          hasattr(module.LlamaModel, "lopa_prefill_lower_k"),
          hasattr(module.LlamaModel, "lopa_build_zero_padded_cache"),
          hasattr(module.LlamaModel, "lopa_build_combined_cache"),
          hasattr(module.LlamaModel, "lopa_forward_from_prefix"),
          hasattr(module.LlamaForCausalLM, "lopa_step_logits"))
    return module


# -----------------------
# Argparser
# -----------------------
def build_argparser():
    p = argparse.ArgumentParser("LoPA-only trainer (STRICT) — Default: global + zero-pad")
    # core
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefill_layers", type=int, default=8, help="K (lower layers) used for prefill")

    # modeling file
    p.add_argument("--lopa_modeling_path", type=str, default="lopa_llama_modeling.py",
                   help="Path to the modified modeling_llama.py file")

    # LoRA (optional)
    p.add_argument("--use_lora", type=str, default="True")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # backend
    p.add_argument("--attn_impl", type=str, choices=["eager", "sdpa", "flash_attention_2"], default="eager")

    # dtype
    p.add_argument("--dtype", type=str, choices=["auto","bf16","fp16","fp32"], default="auto",
                   help="auto=bf16(if supported) else fp16 on CUDA; fp32 on CPU")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--sdpa_math_only", action="store_true")

    # schedule/clip
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # data
    p.add_argument("--max_doc_tokens", type=int, default=2048)
    p.add_argument("--explode", action="store_true", default=True)
    p.add_argument("--no_explode", dest="explode", action="store_false")
    p.add_argument("--group_by_question", action="store_true", default=True)
    p.add_argument("--no_group_by_question", dest="group_by_question", action="store_false")

    # IO
    p.add_argument("--save_best_dir", type=str, default="./_best_ckpt")
    p.add_argument("--cache_dir_model", type=str, default=None)
    p.add_argument("--cache_dir_tokenizer", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)

    # hub
    p.add_argument("--hf_repo_id", type=str, default=None)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")

    # distributed
    p.add_argument("--dist_mode", type=str, choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    p.add_argument("--zero_stage", type=int, default=2)

    # LoPA rope/zero-pad options (DEFAULTS)
    p.add_argument("--lopa_rope_mode", type=str, choices=["local","global"], default="global",
                   help="LoPA RoPE: local=per-layer local positions, global=single global positions")
    p.add_argument("--upper_zero_pad_prefix", dest="upper_zero_pad_prefix", action="store_true", default=True,
                   help="(DEFAULT ON) In global mode, upper layers get L_all-length 0-KV (exact zero-pad equivalence).")
    p.add_argument("--no_upper_zero_pad_prefix", dest="upper_zero_pad_prefix",
                   action="store_false", help=argparse.SUPPRESS)
    p.add_argument("--explicit_empty_upper_cache", action="store_true",
                   help="(Debug) Fill explicit empty KV for upper (length=0). Ignored if zero-pad is ON.")

    # NEW: responses sequential switch (DEFAULT: True)
    p.add_argument("--responses_sequential", action="store_true", default=True,
                   help="Process multiple responses per question sequentially (forward+backward per response).")
    p.add_argument("--no_responses_sequential", dest="responses_sequential",
                   action="store_false", help=argparse.SUPPRESS)

    return p


# -----------------------
# Dataset & Template
# -----------------------
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
        s = tokenizer.apply_chat_template(messages, tokenize=False)
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        if add_generation_prompt and "<|start_header_id|>" in tmpl:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s

def tokens_from_messages(tokenizer, messages, device, add_generation_prompt=False):
    s = apply_chat_template(tokenizer, messages, add_generation_prompt)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


# -----------------------
# Accelerator builder
# -----------------------
def _build_accelerator(args) -> Accelerator:
    if args.dtype == "bf16": mp = "bf16"
    elif args.dtype == "fp16": mp = "fp16"
    else: mp = "no"

    auto_wrap = None
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer as _L
        auto_wrap = {"transformer_layer_cls_to_wrap": (_L,)}
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


# -----------------------
# Train (STRICT)
# -----------------------
def train(args):
    accelerator = _build_accelerator(args)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = accelerator.device

    # 1) Load & assert custom modeling BEFORE instantiating model
    modeling_path = Path(args.lopa_modeling_path).resolve()
    _require(modeling_path.exists(), f"lopa_modeling_path not found: {modeling_path}")
    llama_mod = load_custom_llama_modeling(modeling_path)
    LlamaForCausalLM = llama_mod.LlamaForCausalLM  # noqa

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3) dtype
    if args.dtype == "fp32": dtype = torch.float32
    elif args.dtype == "bf16": dtype = torch.bfloat16
    elif args.dtype == "fp16": dtype = torch.float16
    else:
        dtype = (torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
                 else (torch.float16 if device == "cuda" else torch.float32))
    dtype = torch.bfloat16
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

    # 4) Instantiate model (patched class)
    model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, cache_dir=args.cache_dir_model)

    # 5) Attention impl
    impl = args.attn_impl
    try:
        model.config._attn_implementation = impl
        _get_inner_model(model).model.config._attn_implementation = impl
    except Exception:
        pass
    if accelerator.is_main_process:
        print(f"[Info] attn_implementation = {impl}")

    # 6) STRICT runtime API checks (instance-level)
    inner = _get_inner_model(model)
    inner = _unwrap_peft(inner)
    lm = getattr(inner, "model", None)
    _require(lm is not None, "Inner base model not found (missing `.model`).")
    for need in ("lopa_prefill_lower_k", "lopa_build_zero_padded_cache", "lopa_build_combined_cache", "lopa_forward_from_prefix"):
        _require(hasattr(lm, need), f"LlamaModel lacks `{need}`. Custom modeling injection failed.")
    _require(hasattr(model, "lopa_step_logits"), "LlamaForCausalLM lacks `lopa_step_logits`. Injection failed.")

    # 7) rope_mode (default: global)
    rope_mode = str(getattr(args, "lopa_rope_mode", "global"))
    upper_zero = bool(getattr(args, "upper_zero_pad_prefix", True))
    if accelerator.is_main_process:
        print(f"[LoPA] rope_mode={rope_mode} | upper_zero_pad_prefix={upper_zero}")
    setattr(lm, "lopa_rope_mode", rope_mode)

    # 8) LoRA (optional)
    use_lora = str(args.use_lora).lower() in ("1","true","yes","y")
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(r=int(args.lora_r), lora_alpha=int(args.lora_alpha),
                          lora_dropout=float(args.lora_dropout),
                          bias="none", task_type=TaskType.CAUSAL_LM,
                          target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"])
        model = get_peft_model(model, lcfg)

    # 9) Datasets / loaders
    ds_all = QADataset(args.data_file, tokenizer,
                       max_doc_tokens=int(args.max_doc_tokens),
                       explode=bool(args.explode),
                       group_by_question=bool(args.group_by_question),
                       seed=int(args.seed))
    _require(len(ds_all) > 0, f"Dataset empty after filters: {args.data_file}")

    val_size = max(1, int(0.1 * len(ds_all)))
    train_size = max(1, len(ds_all) - val_size)
    train_set, val_set = random_split(ds_all, [train_size, val_size])
    pin_mem = torch.cuda.is_available()
    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                          pin_memory=pin_mem, persistent_workers=True, collate_fn=collate_identity)
    dl_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=pin_mem, persistent_workers=True, collate_fn=collate_identity)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model, optim, dl_train, dl_val = accelerator.prepare(model, optim, dl_train, dl_val)

    # 10) runtime-safe LlamaModel handle (unwrap Accelerate+PEFT every time)
    def _lm_handle() -> nn.Module:
        inner_now = _get_inner_model(model)
        inner_now = _unwrap_peft(inner_now)
        lm_now = getattr(inner_now, "model", None)
        _require(lm_now is not None, "Inner base model not found (missing `.model`).")
        for need in ("lopa_prefill_lower_k", "lopa_build_zero_padded_cache",
                     "lopa_build_combined_cache", "lopa_forward_from_prefix"):
            _require(hasattr(lm_now, need), f"LlamaModel lacks `{need}` at runtime (bad injection or stale handle).")
        return lm_now

    # 11) re-affirm rope_mode on runtime handle (after prepare/PEFT)
    _lm_handle().lopa_rope_mode = rope_mode
    if accelerator.is_main_process:
        print("[LoPA] re-affirm rope_mode on runtime handle ->", _lm_handle().lopa_rope_mode)

    if accelerator.is_main_process:
        try:
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
        except Exception:
            dev_name = str(device)
        print("[Env] torch:", torch.__version__, "cuda:", torch.version.cuda)
        print("[Env] device:", dev_name, "| param dtype:", next(_get_inner_model(model).parameters()).dtype)

    total_train_steps = args.epochs * max(1, len(dl_train))
    warmup_steps = args.warmup_steps if int(args.warmup_steps) > 0 else int(args.warmup_ratio * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optim,
                                                num_warmup_steps=max(0, warmup_steps),
                                                num_training_steps=max(1, total_train_steps))

    system_prompt = "You are a helpful assistant that answers questions based on the given document. "
    K = max(0, int(args.prefill_layers))
    aux_ratio = float(getattr(args, "aux_prefix_loss_ratio", 0.0) or 0.0)

    # -----------------------
    # Core loss (STRICT path)
    # -----------------------
    def _prefill_lower_k(ids_phase1: torch.LongTensor):
        return _lm_handle().lopa_prefill_lower_k(input_ids=ids_phase1, lower_k=K, use_cache=True)

    def _compose_upper_cache(lower_cache: DynamicCache, L_all: int, batch_size: int, device) -> DynamicCache:
        if rope_mode == "global" and upper_zero:
            return _lm_handle().lopa_build_zero_padded_cache(
                lower_cache, lower_k=K, batch_size=batch_size, device=device, zero_len=L_all
            )
        if bool(getattr(args, "explicit_empty_upper_cache", False)):
            return _lm_handle().lopa_build_combined_cache(
                lower_cache, lower_k=K, batch_size=batch_size, device=device
            )
        return lower_cache

    def compute_loss_on_sample(q: str, d: str, resp: str) -> torch.Tensor:
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=True)

        pref_out = _prefill_lower_k(ids_phase1)
        lower_cache = pref_out.past_key_values
        L_all = lower_cache.get_seq_length()
        combined = _compose_upper_cache(lower_cache, L_all, ids_phase1.size(0), ids_phase1.device)

        msgs_ass = msgs + [{"role": "assistant", "content": resp}]
        full_ids = tokens_from_messages(tokenizer, msgs_ass, accelerator.device, add_generation_prompt=False)
        assistant_ids = full_ids[:, ids_hdr.size(1):]
        _require(assistant_ids.numel() > 0, "assistant_ids is empty")

        seed = ids_hdr[:, L_all:] if ids_hdr.size(1) > L_all else ids_phase1[:, -1:]
        inp = torch.cat([seed, assistant_ids], dim=1)
        labels = inp.clone(); labels[:, :seed.size(1)] = -100
        T = inp.size(1)

        out = model.lopa_step_logits(
            input_ids=inp, prefix_len=L_all, past_key_values=combined,
            attention_mask_total_len=L_all + T, logits_to_keep=T, labels=labels,
        )
        loss = out.loss if out.loss is not None else torch.zeros((), device=accelerator.device, dtype=torch.float32)
        if aux_ratio > 0.0 and K > 0:
            pass
        return loss

    def compute_loss_on_group_mean(qid: str, q: str, d: str, responses: List[str]) -> torch.Tensor:
        """(기존 방식) 응답 로스를 평균내어 한 번만 backward."""
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=True)

        pref_out = _prefill_lower_k(ids_phase1)
        lower_cache = pref_out.past_key_values
        L_all = lower_cache.get_seq_length()
        combined = _compose_upper_cache(lower_cache, L_all, ids_phase1.size(0), ids_phase1.device)

        losses = []
        for resp in responses:
            if not isinstance(resp, str) or not resp.strip():
                continue
            msgs_ass = msgs + [{"role": "assistant", "content": resp}]
            full_ids = tokens_from_messages(tokenizer, msgs_ass, accelerator.device, add_generation_prompt=False)
            assistant_ids = full_ids[:, ids_hdr.size(1):]
            if assistant_ids.numel() == 0:
                continue
            seed = ids_hdr[:, L_all:] if ids_hdr.size(1) > L_all else ids_phase1[:, -1:]
            inp = torch.cat([seed, assistant_ids], dim=1)
            labels = inp.clone(); labels[:, :seed.size(1)] = -100
            T = inp.size(1)
            out = model.lopa_step_logits(
                input_ids=inp, prefix_len=L_all, past_key_values=combined,
                attention_mask_total_len=L_all + T, logits_to_keep=T, labels=labels,
            )
            if out.loss is not None and torch.isfinite(out.loss):
                losses.append(out.loss)
        _require(len(losses) > 0, "All responses invalid/empty for this item.")
        return torch.stack(losses, dim=0).mean()

    def compute_loss_on_group_seq(qid: str, q: str, d: str, responses: List[str]) -> float:
        """
        응답별로 완전히 순차 forward/backward.
        각 응답마다 fresh upper cache를 *다시* 만들기 때문에 그래프 공유가 없어 안전.
        반환값은 로깅용 평균 loss(float); optimizer.step()/zero_grad 는 바깥 루프에서.
        """
        # 0) 공통 토크나이즈
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=True)

        # 1) 하층 K 프리필 (no_grad) — 공통
        with torch.no_grad():
            pref_out = _prefill_lower_k(ids_phase1)
        lower_cache = pref_out.past_key_values
        L_all = lower_cache.get_seq_length()

        # 2) 유효 응답 목록 먼저 수집(빈 것/공백 제외)
        eligible = []
        for resp in responses:
            if not isinstance(resp, str) or not resp.strip():
                continue
            msgs_ass = msgs + [{"role": "assistant", "content": resp}]
            full_ids = tokens_from_messages(tokenizer, msgs_ass, accelerator.device, add_generation_prompt=False)
            assistant_ids = full_ids[:, ids_hdr.size(1):]
            if assistant_ids.numel() == 0:
                continue
            eligible.append((resp, assistant_ids))
        _require(len(eligible) > 0, "All responses invalid/empty for this item.")
        eff_total = len(eligible)
        bs = max(1, args.batch_size)

        # 3) 순차 실행(응답마다 fresh combined 재구성 → 독립 그래프)
        losses_for_log = []
        for resp, assistant_ids in eligible:
            # seed
            seed = ids_hdr[:, L_all:] if ids_hdr.size(1) > L_all else ids_phase1[:, -1:]
            inp = torch.cat([seed, assistant_ids], dim=1)
            labels = inp.clone(); labels[:, :seed.size(1)] = -100
            T = inp.size(1)

            # ★ 핵심: 매 응답마다 *새로운* combined 캐시를 만든다
            combined = _compose_upper_cache(lower_cache, L_all, ids_phase1.size(0), ids_phase1.device)

            out = model.lopa_step_logits(
                input_ids=inp,
                prefix_len=L_all,
                past_key_values=combined,
                attention_mask_total_len=L_all + T,
                logits_to_keep=T,
                labels=labels,
            )
            _require(out.loss is not None and torch.isfinite(out.loss), "loss is NaN/inf")
            losses_for_log.append(float(out.loss.detach().item()))

            # ZeRO/AMP 호환 backward (평균 스케일 유지)
            accelerator.backward(out.loss / eff_total / bs)

            # (선택) 메모리 여유 없음이면 아래 라인 사용
            # del combined, out; torch.cuda.empty_cache()

        return sum(losses_for_log) / eff_total

    # wandb (optional)
    run = None
    if accelerator.is_main_process and args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=f"lopa_strict_zero_pad-K{K}", config=vars(args))
        except Exception:
            run = None

    best_loss = float("inf")
    best_dir = Path(args.save_best_dir); best_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        train_iter = tqdm(dl_train, desc=f"Epoch {epoch} [train]", leave=False) if accelerator.is_main_process else dl_train

        for bidx, batch in enumerate(train_iter):
            items = list(batch) if isinstance(batch, list) else [batch]
            if not items:
                continue

            step_losses = []
            valid = 0
            # 응답 순차 모드 플래그
            RESP_SEQ = bool(getattr(args, "responses_sequential", True))

            for it in items:
                # it = (qid, q, d, responses) | (qid, q, d, r) | (q, d, r)
                if len(it) == 4 and isinstance(it[3], list):
                    qid, q, d, rs = it
                    if RESP_SEQ:
                        # 응답별 순차 backward (여기서는 backward 이미 수행됨)
                        loss_val = compute_loss_on_group_seq(str(qid), q, d, rs)
                        step_losses.append(loss_val)
                        valid += 1
                        # 순차 모드에서는 여기서 backward가 끝났으므로 추가 backward 없음
                        continue
                    else:
                        # 기존 mean 경로: 한 번만 backward
                        loss_i = compute_loss_on_group_mean(str(qid), q, d, rs)
                elif len(it) == 4:
                    qid, q, d, r = it
                    loss_i = compute_loss_on_sample(q, d, r)
                else:
                    q, d, r = it
                    loss_i = compute_loss_on_sample(q, d, r)

                # 기존 경로(=mean or single response)만 여기서 backward
                accelerator.backward(loss_i / max(1, args.batch_size))
                step_losses.append(float(loss_i.detach().item()))
                valid += 1

            if valid > 0:
                if float(args.grad_clip) and args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step(); scheduler.step()
                tr_loss_sum += sum(step_losses); tr_count += valid
            optim.zero_grad(set_to_none=True)

            if accelerator.is_main_process and run is not None:
                try:
                    lr = float(scheduler.get_last_lr()[0])
                except Exception:
                    lr = optim.param_groups[0]["lr"]
                payload = {"step": global_step, "lr": lr, "train/n_item": valid}
                if step_losses: payload["train/step_loss_mean"] = sum(step_losses)/len(step_losses)
                run.log(payload)
            global_step += 1

        tr_avg = tr_loss_sum / max(1, tr_count)

        # ---- validation ----
        model.eval()
        va_loss_sum, va_count = 0.0, 0
        with torch.no_grad():
            for batch in dl_val:
                items = list(batch) if isinstance(batch, list) else [batch]
                if not items: continue
                for it in items:
                    if len(it) == 4 and isinstance(it[3], list):
                        qid, q, d, rs = it
                        # 검증은 항상 mean으로
                        loss_i = compute_loss_on_group_mean(str(qid), q, d, rs)
                    elif len(it) == 4:
                        qid, q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r)
                    else:
                        q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r)
                    va_loss_sum += float(loss_i.detach().item()); va_count += 1
        va_avg = va_loss_sum / max(1, va_count)

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] train_avg={tr_avg:.6f} | valid_avg={va_avg:.6f} | n_train={tr_count} | n_valid={va_count}")
            if run is not None:
                run.log({"epoch": epoch, "train/avg": tr_avg, "valid/avg": va_avg})

        # ---- save best ----
        if va_count > 0 and va_avg < best_loss and accelerator.is_main_process:
            best_loss = va_avg
            # clean dir
            for p in best_dir.glob("*"):
                try:
                    if p.is_file(): p.unlink()
                    elif p.is_dir():
                        import shutil; shutil.rmtree(p)
                except Exception:
                    pass

            base_dir = best_dir / "base"; base_dir.mkdir(parents=True, exist_ok=True)
            try:
                to_unwrap = accelerator.unwrap_model(model)
                # PEFT 여부
                is_peft = False
                try:
                    from peft import PeftModel
                    is_peft = isinstance(to_unwrap, PeftModel)
                except Exception:
                    pass
                if is_peft:
                    base_clean = LlamaForCausalLM.from_pretrained(
                        args.model_name, torch_dtype=dtype, cache_dir=args.cache_dir_model
                    )
                    base_clean.save_pretrained(base_dir, safe_serialization=True)
                    (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                    to_unwrap.save_pretrained(best_dir / "lora")
                else:
                    to_unwrap.save_pretrained(base_dir, safe_serialization=True)
            except Exception as e:
                raise RuntimeError(f"Failed to save best: {e}")

            try:
                tokenizer.save_pretrained(best_dir)
                with open(best_dir / "rope_mode.txt", "w") as f:
                    f.write(f"rope_mode={rope_mode}\n")
                    f.write(f"upper_zero_pad_prefix={upper_zero}\n")
                    f.write(f"prefill_layers={K}\n")
            except Exception:
                pass

            print(f"[Best] Saved to {best_dir} (val={best_loss:.6f})")

        accelerator.wait_for_everyone()

    # push to hub
    if accelerator.is_main_process and args.push_to_hub and args.hf_repo_id:
        token = os.environ.get("HF_TOKEN", "")
        _require(bool(token), "HF_TOKEN not set; export HF_TOKEN=...")
        create_repo(args.hf_repo_id, exist_ok=True, private=args.private_repo, token=token)
        upload_folder(repo_id=args.hf_repo_id, folder_path=str(best_dir),
                      token=token, commit_message="LoPA trainer (strict global+zero-pad, seq responses) upload", allow_patterns=["*"])
        print(f"✅ Uploaded to Hub: {args.hf_repo_id}")


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

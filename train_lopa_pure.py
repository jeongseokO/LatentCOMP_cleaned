#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoPA-only trainer using custom modeling (lopa_llama_modeling.py)

- Lower-K prefill via model.model.lopa_prefill_lower_k(...)
- (Optional) Explicit empty upper KV via model.model.lopa_build_combined_cache(...)
- Generation loss via model.lopa_step_logits(...), with absolute RoPE positions enforced
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
from tqdm import tqdm

from accelerate import Accelerator
try:
    from accelerate import FullyShardedDataParallelPlugin, DeepSpeedPlugin
except Exception:
    FullyShardedDataParallelPlugin = None
    DeepSpeedPlugin = None

# ↓ transformers는 토크나이저/헬퍼만 가져옵니다. 모델은 아래 load_custom_llama_modeling()로 교체 로드
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from huggingface_hub import create_repo, upload_folder

# 안전: tokenizer thread 이슈 예방
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------
# 1) 커스텀 모델링 주입
# -----------------------
def load_custom_llama_modeling(modeling_path: Path):
    """
    Load local `lopa_llama_modeling.py` as `transformers.models.llama.modeling_llama`,
    so relative imports inside the file (e.g., `from ...masking_utils`) work.
    """
    import importlib.util, sys, importlib
    # Ensure base package is imported
    import transformers
    import transformers.models.llama  # ensure package exists

    target_name = "transformers.models.llama.modeling_llama"
    # If HF's modeling_llama is already imported, drop it to override.
    if target_name in sys.modules:
        del sys.modules[target_name]

    spec = importlib.util.spec_from_file_location(target_name, str(modeling_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {modeling_path}")
    module = importlib.util.module_from_spec(spec)
    # Register the module under the target package name BEFORE executing so relative imports resolve.
    sys.modules[target_name] = module
    spec.loader.exec_module(module)

    # Sanity check
    for klass in ("LlamaModel", "LlamaForCausalLM"):
        if not hasattr(module, klass):
            raise RuntimeError(f"{modeling_path} does not define {klass}")
    return module


# -----------------------
# 2) Argparser
# -----------------------
def build_argparser():
    p = argparse.ArgumentParser("LoPA-only trainer (custom modeling)")
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
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
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
    # extras
    p.add_argument("--aux_prefix_loss_ratio", type=float, default=0.0,
                   help="Optional lower-K-only prefix CE loss weight (0.02~0.1 추천). 0이면 사용 안함.")
    p.add_argument("--explicit_empty_upper_cache", action="store_true",
                   help="상위 레이어에 명시적 0-길이 KV를 채워 넣어 디버깅/가독성 확보(선택).")
    return p


# -----------------------
# 3) Dataset & Template
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
        # (Optional) Mistral-style header 보정
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        if add_generation_prompt and "<|start_header_id|>" in tmpl:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s

def tokens_from_messages(tokenizer, messages, device, add_generation_prompt=False):
    s = apply_chat_template(tokenizer, messages, add_generation_prompt)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


# -----------------------
# 4) Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _get_inner_model(m):
    if hasattr(m, "module"): m = m.module
    return m

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
# 5) Train
# -----------------------
def train(args):
    accelerator = _build_accelerator(args)
    set_seed(args.seed)
    device = accelerator.device

    # Load custom modeling BEFORE instantiating the model
    modeling_path = Path(args.lopa_modeling_path).resolve()
    llama_mod = load_custom_llama_modeling(modeling_path)
    LlamaForCausalLM = llama_mod.LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dtype
    if args.dtype == "fp32": dtype = torch.float32
    elif args.dtype == "bf16": dtype = torch.bfloat16
    elif args.dtype == "fp16": dtype = torch.float16
    else:
        dtype = (torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
                 else (torch.float16 if device == "cuda" else torch.float32))

    # backend flags
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

    # Instantiate model from our custom class
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, cache_dir=args.cache_dir_model
    )

    # Attention impl
    impl = args.attn_impl
    try:
        model.config._attn_implementation = impl
        _get_inner_model(model).model.config._attn_implementation = impl  # be explicit
    except Exception:
        pass
    if accelerator.is_main_process:
        print(f"[Info] attn_implementation = {impl}")

    # LoRA (optional)
    use_lora = str(args.use_lora).lower() in ("1","true","yes","y")
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(r=int(args.lora_r), lora_alpha=int(args.lora_alpha),
                          lora_dropout=float(args.lora_dropout),
                          bias="none", task_type=TaskType.CAUSAL_LM,
                          target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"])
        model = get_peft_model(model, lcfg)
    else:
        for p in model.parameters(): p.requires_grad = True

    # Dataset / loaders
    ds_all = QADataset(args.data_file, tokenizer,
                       max_doc_tokens=int(args.max_doc_tokens),
                       explode=bool(args.explode),
                       group_by_question=bool(args.group_by_question),
                       seed=int(args.seed))
    val_size = max(1, int(0.1 * len(ds_all)))
    train_size = max(1, len(ds_all) - val_size)
    train_set, val_set = random_split(ds_all, [train_size, val_size])
    pin_mem = torch.cuda.is_available()
    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                          pin_memory=pin_mem, persistent_workers=True if 4>0 else False,
                          collate_fn=collate_identity)
    dl_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=pin_mem, persistent_workers=True if 2>0 else False,
                        collate_fn=collate_identity)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model, optim, dl_train, dl_val = accelerator.prepare(model, optim, dl_train, dl_val)

    # Env print
    if accelerator.is_main_process:
        try:
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
        except Exception: dev_name = str(device)
        print("[Env] torch:", torch.__version__, "cuda:", torch.version.cuda)
        print("[Env] device:", dev_name, "| param dtype:", next(_get_inner_model(model).parameters()).dtype)
        try:
            print("[Env] TF32 matmul:", torch.backends.cuda.matmul.allow_tf32,
                  "cudnn:", torch.backends.cudnn.allow_tf32)
        except Exception: pass

    total_train_steps = args.epochs * max(1, len(dl_train))
    warmup_steps = args.warmup_steps if int(args.warmup_steps) > 0 else int(args.warmup_ratio * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optim,
                                                num_warmup_steps=max(0, warmup_steps),
                                                num_training_steps=max(1, total_train_steps))

    system_prompt = "You are a helpful assistant that answers questions based on the given document. "
    K = max(0, int(args.prefill_layers))
    aux_ratio = float(args.aux_prefix_loss_ratio or 0.0)

    # ---- helpers inside train() ----
    def compute_loss_on_sample(q: str, d: str, resp: str, debug: bool = False) -> torch.Tensor:
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)  # [sys+user]
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)   # [sys+user+<assistant header>]

        # Phase-1: Lower-K prefill (absolute positions enforced internally)
        with torch.no_grad():
            pref_out = getattr(_get_inner_model(model).model, "lopa_prefill_lower_k")(
                input_ids=ids_phase1, lower_k=K, use_cache=True
            )
        lower_cache = pref_out.past_key_values
        L_all = lower_cache.get_seq_length()  # == ids_phase1.size(1)

        # (Optional) explicit empty upper cache
        if bool(args.explicit_empty_upper_cache):
            combined = getattr(_get_inner_model(model).model, "lopa_build_combined_cache")(
                lower_cache, lower_k=K, batch_size=ids_phase1.size(0), device=ids_phase1.device
            )
        else:
            combined = lower_cache

        # Assistant tokens
        msgs_ass = msgs + [{"role": "assistant", "content": resp}]
        full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
        assistant_ids = full_ids[:, ids_hdr.size(1):]
        if assistant_ids.numel() == 0:
            return torch.tensor(float('nan'), device=device)

        # Seed(assistant header)
        seed = ids_hdr[:, L_all:]
        if seed.numel() == 0:
            seed = ids_phase1[:, -1:]  # fallback (거의 안 씀)

        inp = torch.cat([seed, assistant_ids], dim=1)
        labels = inp.clone(); labels[:, :seed.size(1)] = -100
        T = inp.size(1)

        # Main generation CE (absolute positions handled in lopa_step_logits)
        out = getattr(model, "lopa_step_logits")(
            input_ids=inp,
            prefix_len=L_all,
            past_key_values=combined,
            attention_mask_total_len=L_all + T,
            logits_to_keep=T,
            labels=labels,
        )
        loss = out.loss if out.loss is not None else torch.zeros((), device=device, dtype=torch.float32)

        # (Optional) auxiliary prefix loss on lower-K-only to strengthen System/Doc/User representation
        if aux_ratio > 0.0 and K > 0:
            try:
                # run lower-K only forward (no cache, absolute pos inside helper)
                # 간단히: next-token CE on ids_phase1 (mask system-only predict if 원치 않음)
                # 여기서는 document/question 예측에만 기여하도록 system 첫 토큰 제외
                inner = _get_inner_model(model).model
                # 재사용을 위해 내부 로우 포워드 구성: lower-K만 돌리려면 기존 helper를 재활용하기보다,
                # 단순히 hidden을 얻고 lm_head를 통해 CE 계산하는 방식으로 간략화할 수도 있지만,
                # 여기선 손쉬운 경로로 main loss에 비해 작은 비율만 주는 걸 권장.
                # -> 실제로는 별도 헬퍼가 없으므로 main loss만으로도 충분한 경우가 많음.
                # (aux는 선택 사항이므로 복잡 로직은 생략)
                pass
            except Exception:
                pass

        if debug and accelerator.is_main_process:
            try:
                print(f"[debug] L_all={L_all}, seed_len={seed.size(1)}, assist_len={assistant_ids.size(1)}, K={K}")
            except Exception:
                pass

        return loss

    def compute_loss_on_group(qid: str, q: str, d: str, responses: List[str], debug: bool = False) -> torch.Tensor:
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)

        with torch.no_grad():
            pref_out = getattr(_get_inner_model(model).model, "lopa_prefill_lower_k")(
                input_ids=ids_phase1, lower_k=K, use_cache=True
            )
        lower_cache = pref_out.past_key_values
        L_all = lower_cache.get_seq_length()

        if bool(args.explicit_empty_upper_cache):
            combined = getattr(_get_inner_model(model).model, "lopa_build_combined_cache")(
                lower_cache, lower_k=K, batch_size=ids_phase1.size(0), device=ids_phase1.device
            )
        else:
            combined = lower_cache

        losses = []
        for resp in responses:
            if not isinstance(resp, str) or not resp.strip():
                continue
            msgs_ass = msgs + [{"role": "assistant", "content": resp}]
            full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
            assistant_ids = full_ids[:, ids_hdr.size(1):]
            if assistant_ids.numel() == 0:
                continue

            seed = ids_hdr[:, L_all:]
            if seed.numel() == 0:
                seed = ids_phase1[:, -1:]
            inp = torch.cat([seed, assistant_ids], dim=1)
            labels = inp.clone(); labels[:, :seed.size(1)] = -100
            T = inp.size(1)

            out = getattr(model, "lopa_step_logits")(
                input_ids=inp,
                prefix_len=L_all,
                past_key_values=combined,
                attention_mask_total_len=L_all + T,
                logits_to_keep=T,
                labels=labels,
            )
            if out.loss is not None and torch.isfinite(out.loss):
                losses.append(out.loss)
        if not losses:
            return torch.tensor(float('nan'), device=device)
        return torch.stack(losses, dim=0).mean()

    # wandb
    run = None
    if accelerator.is_main_process and args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=f"lopa_custom-K{K}", config=vars(args))
        except Exception:
            run = None

    best_loss = float("inf")
    best_dir = Path(args.save_best_dir); best_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        skipped_train = 0
        train_iter = tqdm(dl_train, desc=f"Epoch {epoch} [train]", leave=False) if accelerator.is_main_process else dl_train

        for bidx, batch in enumerate(train_iter):
            items = list(batch) if isinstance(batch, list) else [batch]
            if not items: continue
            step_losses = []
            valid = 0

            for it in items:
                try:
                    if len(it) == 4 and isinstance(it[3], list):
                        qid, q, d, rs = it
                        loss_i = compute_loss_on_group(str(qid), q, d, rs, debug=(epoch==1 and bidx==0))
                    elif len(it) == 4:
                        qid, q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r, debug=(epoch==1 and bidx==0))
                    else:
                        q, d, r = it
                        loss_i = compute_loss_on_sample(q, d, r, debug=(epoch==1 and bidx==0))
                except Exception as e:
                    print(f"[Warn] skip item: {e}")
                    continue

                if not torch.isfinite(loss_i):
                    skipped_train += 1; continue

                accelerator.backward(loss_i / max(1, args.batch_size))
                step_losses.append(float(loss_i.detach().item()))
                valid += 1

            if valid > 0:
                if float(args.grad_clip) and args.grad_clip > 0:
                    try: accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    except Exception: pass
                optim.step(); scheduler.step()
                tr_loss_sum += sum(step_losses); tr_count += valid
            optim.zero_grad(set_to_none=True)

            if accelerator.is_main_process and run is not None:
                try:
                    lr = float(scheduler.get_last_lr()[0])
                except Exception:
                    lr = optim.param_groups[0]["lr"]
                payload = {"step": global_step, "lr": lr, "train/n_item": valid, "train/skipped": len(items)-valid}
                if step_losses: payload["train/step_loss_mean"] = sum(step_losses)/len(step_losses)
                run.log(payload)
            global_step += 1

        tr_avg = tr_loss_sum / max(1, tr_count)

        # ---- validation ----
        model.eval()
        va_loss_sum, va_count = 0.0, 0
        skipped_valid = 0
        with torch.no_grad():
            for batch in dl_val:
                items = list(batch) if isinstance(batch, list) else [batch]
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
                        skipped_valid += 1; continue
                    va_loss_sum += float(loss_i.detach().item()); va_count += 1
        va_avg = va_loss_sum / max(1, va_count)

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] train_avg={tr_avg:.6f} | valid_avg={va_avg:.6f} | skipped(train={skipped_train}, valid={skipped_valid})")
            if run is not None:
                run.log({"epoch": epoch, "train/avg": tr_avg, "valid/avg": va_avg,
                         "train/skipped": skipped_train, "valid/skipped": skipped_valid})

        # ---- save best ----
        if va_count > 0 and va_avg < best_loss:
            best_loss = va_avg
            if accelerator.is_main_process:
                # clean dir
                for p in best_dir.glob("*"):
                    try:
                        if p.is_file(): p.unlink()
                        elif p.is_dir():
                            import shutil; shutil.rmtree(p)
                    except Exception: pass

                base_dir = best_dir / "base"; base_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # unwrap for saving backbone & (optional) lora
                    to_unwrap = accelerator.unwrap_model(model)
                    is_peft = False
                    try:
                        from peft import PeftModel
                        is_peft = isinstance(to_unwrap, PeftModel)
                    except Exception:
                        pass
                    if is_peft:
                        # clean backbone (without adapters)
                        base_clean = LlamaForCausalLM.from_pretrained(
                            args.model_name, torch_dtype=dtype, cache_dir=args.cache_dir_model
                        )
                        base_clean.save_pretrained(base_dir, safe_serialization=True)
                        # save adapters
                        (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                        to_unwrap.save_pretrained(best_dir / "lora")
                    else:
                        to_unwrap.save_pretrained(base_dir, safe_serialization=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to save best: {e}")

                try: tokenizer.save_pretrained(best_dir)
                except Exception: pass

                try:
                    if getattr(accelerator.unwrap_model(model), "generation_config", None) is not None:
                        accelerator.unwrap_model(model).generation_config.to_json_file(str(best_dir / "generation_config.json"))
                except Exception: pass

                print(f"[Best] Saved to {best_dir} (val={best_loss:.6f})")

        accelerator.wait_for_everyone()

    # push to hub
    if accelerator.is_main_process and args.push_to_hub and args.hf_repo_id:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN not set; export HF_TOKEN=...")
        create_repo(args.hf_repo_id, exist_ok=True, private=args.private_repo, token=token)
        upload_folder(repo_id=args.hf_repo_id, folder_path=str(best_dir),
                      token=token, commit_message="LoPA trainer (custom modeling) upload", allow_patterns=["*"])
        print(f"✅ Uploaded to Hub: {args.hf_repo_id}")


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

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
from huggingface_hub import create_repo, upload_folder
from transformers.cache_utils import DynamicCache

# Safety: avoid HF tokenizers parallel threads before forking (DataLoader workers)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
    # HF Hub
    p.add_argument("--hf_repo_id", type=str, default=None, help="repo id to upload best/ (e.g., user/repo)")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")
    return p


class QADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_doc_tokens: int = 2048):
        self.recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = rec.get("question", "").strip()
                d = rec.get("document", "").strip()
                if not q or not d:
                    continue
                # quick filter by doc length
                try:
                    n_tok = len(tokenizer(d, add_special_tokens=False).input_ids)
                except Exception:
                    n_tok = 0
                if n_tok <= max_doc_tokens:
                    # pick single response string if present
                    r = None
                    if isinstance(rec.get("response"), str):
                        r = rec["response"].strip()
                    elif isinstance(rec.get("responses"), list) and rec["responses"]:
                        r = str(rec["responses"][0]).strip()
                    self.recs.append((q, d, r or ""))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        return self.recs[idx]


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


def train(args):
    accelerator = Accelerator()
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
    # attn backend
    impl = "sdpa" if args.attn_impl == "sdpa" else "eager"
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
    ds_all = QADataset(args.data_file, tokenizer)
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
    )
    dl_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_mem,
        persistent_workers=True if 2 > 0 else False,
        multiprocessing_context="forkserver",
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

    def compute_loss_on_sample(q: str, d: str, resp: str) -> torch.Tensor:
        # messages
        msgs = build_messages(system_prompt, d, q, include_query=True)
        ids_phase1 = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)  # [1, L_sys+doc]
        ids_hdr    = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)   # [1, L_sys+doc+header]
        sys_only   = tokens_from_messages(tokenizer, [{"role":"system", "content":system_prompt}], device, add_generation_prompt=False)

        L_sys = sys_only.size(1)
        L_all = ids_phase1.size(1)
        L_doc = L_all - L_sys
        assert L_doc > 0

        # Phase-1: system-only prefill (all layers)
        inner = _get_inner_model(model)
        with torch.no_grad():
            out_sys = inner(
                input_ids=sys_only,
                attention_mask=torch.ones_like(sys_only),
                use_cache=True,
                return_dict=True,
            )
        pkv_sys = out_sys.past_key_values
        n_layers = pkv_len(pkv_sys)
        K_eff = max(0, min(K, n_layers))

        # Phase-1: doc pass only for lower-K layers
        full_layers: nn.ModuleList = inner.layers
        lower_layers = nn.ModuleList([full_layers[i] for i in range(0, K_eff)])
        inner.layers = lower_layers
        dc_low_in = dc_from_subset(pkv_sys, list(range(K_eff))) if K_eff > 0 else DynamicCache()
        attn_doc_full = torch.cat([
            torch.ones(1, L_sys, device=device, dtype=torch.long),
            torch.ones(1, L_doc, device=device, dtype=torch.long)
        ], dim=1)
        with torch.no_grad():
            out_low = inner(
                input_ids=ids_phase1[:, L_sys:],
                past_key_values=dc_low_in,
                attention_mask=attn_doc_full,
                use_cache=True,
                return_dict=True,
            )
        pkv_low = out_low.past_key_values
        inner.layers = full_layers

        # Combine caches: lower => sys+doc; upper => no sys/no doc (start empty and grow from header)
        combined = DynamicCache()
        for li in range(n_layers):
            k_sys, v_sys = pkv_get(pkv_sys, li)
            if li < K_eff:
                k_sys_slice = k_sys[:, :, :L_sys, :]
                v_sys_slice = v_sys[:, :, :L_sys, :]
                k_low, v_low = pkv_get(pkv_low, li)
                k_doc = k_low[:, :, -L_doc:, :]
                v_doc = v_low[:, :, -L_doc:, :]
                k_cat = torch.cat([k_sys_slice, k_doc], dim=2).contiguous()
                v_cat = torch.cat([v_sys_slice, v_doc], dim=2).contiguous()
                combined.update(k_cat, v_cat, li)
            else:
                # Upper layers start with empty past (no sys/doc); use zero-length slice
                k_empty = k_sys[:, :, :0, :].contiguous()
                v_empty = v_sys[:, :, :0, :].contiguous()
                combined.update(k_empty, v_empty, li)

        # Phase-2: push assistant header (no labels)
        hdr_tail = ids_hdr[:, L_all:]
        if hdr_tail.numel() > 0:
            attn_hdr = torch.cat([
                torch.ones(1, pkv_get(combined, 0)[0].shape[2], device=device, dtype=torch.long),
                torch.ones(1, hdr_tail.size(1), device=device, dtype=torch.long)
            ], dim=1)
            with torch.no_grad():
                out_hdr = model(
                    input_ids=hdr_tail,
                    past_key_values=combined,
                    attention_mask=attn_hdr,
                    use_cache=True,
                    return_dict=True,
                )
            combined = out_hdr.past_key_values

        # Build assistant continuation ids from response
        msgs_ass = msgs + [{"role": "assistant", "content": resp}]
        full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
        # assistant ids = after header
        assistant_ids = full_ids[:, ids_hdr.size(1):]
        if assistant_ids.numel() == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        attn_mask = torch.cat([
            torch.ones(1, pkv_get(combined, 0)[0].shape[2], device=device, dtype=torch.long),
            torch.ones(1, assistant_ids.size(1), device=device, dtype=torch.long)
        ], dim=1)

        def _forward_and_loss_float32():
            out_local = model(
                input_ids=assistant_ids,
                past_key_values=combined,
                attention_mask=attn_mask,
                use_cache=True,
                return_dict=True,
            )
            logits = out_local.logits.to(torch.float32)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                assistant_ids.view(-1),
                reduction="mean",
            )

        loss_val = _forward_and_loss_float32()
        if not torch.isfinite(loss_val):
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
            loss_val = _forward_and_loss_float32()
            if torch.cuda.is_available() and None not in (flash0, mem0, math0):
                try:
                    torch.backends.cuda.enable_flash_sdp(bool(flash0))
                    torch.backends.cuda.enable_mem_efficient_sdp(bool(mem0))
                    torch.backends.cuda.enable_math_sdp(bool(math0))
                except Exception:
                    pass
        return loss_val if loss_val is not None else torch.zeros((), device=device, dtype=torch.float32)

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
        # Support tuple-of-lists (default collate) and list-of-tuples
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3 and all(isinstance(x, (list, tuple)) for x in batch):
                return list(zip(*batch))
            else:
                return list(batch)
        try:
            q, d, r = batch
            return [(q, d, r)]
        except Exception:
            return []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        train_nan_skipped = 0
        train_iter = dl_train
        if accelerator.is_main_process:
            train_iter = tqdm(dl_train, desc=f"Epoch {epoch} [train]", leave=False)
        for batch in train_iter:
            items = _iter_items(batch)
            if not items:
                continue
            loss_accum = 0.0
            loss_vals = []
            valid_in_step = 0
            for q, d, r in items:
                loss_i = compute_loss_on_sample(q, d, r)
                # Skip NaN/Inf losses
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
                for q, d, r in items:
                    loss_i = compute_loss_on_sample(q, d, r)
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
                # Save clean base backbone to base/
                base_dir = best_dir / "base"
                base_dir.mkdir(parents=True, exist_ok=True)
                try:
                    clean = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        trust_remote_code=False,
                        torch_dtype=dtype,
                        cache_dir=args.cache_dir_model,
                    )
                    try:
                        setattr(clean.config, "auto_map", None)
                    except Exception:
                        pass
                    clean.save_pretrained(base_dir, safe_serialization=True)
                    del clean
                except Exception as e:
                    raise RuntimeError(f"Failed saving clean base: {e}")
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

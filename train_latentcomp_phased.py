#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phased LatentCOMP trainer (clean version)

Implements the exact three-phase flow you described with original-position RoPE:

Phase 0: Compute SYS
Phase 1: Compute SYS + User(...), keeping positions as original; then drop Document
Phase 2: Reuse SYS + User (from Phase 1, without Document) and train on Assistant(Response)

Variants:
- include_query=True:
  Phase 1: SYS, User(Document, Question, Specials, Question)
  Phase 2: SYS, User(Specials, Question), Assistant(Response)

- include_query=False:
  Phase 1: SYS, User(Document, Specials, Question)
  Phase 2: SYS, User(Specials, Question), Assistant(Response)

- include_specials=False:
  Phase 1: SYS, User(Document, Question)
  Phase 2: SYS, User(Question), Assistant(Response)

Notes:
- Positions are preserved by adding a "gap" between the cached past length and the position_ids.
- Scheduling, accelerate, and simple packaging follow the cleaned trainer conventions.
- This trainer focuses on clarity and the specified phase logic; it does not implement
  dynamic support blocks or multi-document splitting. It handles a single document string per row.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from accelerate import Accelerator
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb


# -----------------------------
# Utility: argparse helpers
# -----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def build_argparser():
    p = argparse.ArgumentParser("Phased LatentCOMP Trainer (clean)")
    # Core
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    # Phased options
    p.add_argument("--include_query", type=str2bool, default=True)
    p.add_argument("--include_specials", type=str2bool, default=True)
    p.add_argument("--latent_token_num", type=int, default=10)
    p.add_argument("--pos_mode", type=str, choices=["original"], default="original")

    # Prefill layers (kept for API compatibility; this clean version evaluates full layers)
    p.add_argument("--prefill_layers", type=int, default=0, help="Reserved; full-layer compute in this clean trainer")

    # Scheduler
    p.add_argument("--lr_scheduler", type=str, choices=["cosine","linear","one_cycle","exponential","step","plateau","none"], default="cosine")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--lr_step_size", type=int, default=1)
    p.add_argument("--plateau_patience", type=int, default=1)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--one_cycle_pct_start", type=float, default=0.1)
    p.add_argument("--one_cycle_div_factor", type=float, default=25.0)
    p.add_argument("--one_cycle_final_div_factor", type=float, default=1e4)

    # Saving / Hub
    p.add_argument("--save_best_dir", type=str, default="./_best_ckpt")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_repo_id", type=str, default=None)
    p.add_argument("--private_repo", action="store_true")

    # Cache dirs
    p.add_argument("--cache_dir_tokenizer", type=str, default=None)
    p.add_argument("--cache_dir_model", type=str, default=None)

    # W&B
    p.add_argument("--wandb_project", type=str, default="latentcomp-phased-clean")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # LoRA
    p.add_argument("--use_lora", type=str2bool, default=False)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Data filtering
    p.add_argument("--max_doc_tokens", type=int, default=12500)
    return p


# -----------------------------
# DynamicCache shim utils
# -----------------------------
try:
    from transformers import DynamicCache
    USE_DYNAMIC_CACHE = True
except Exception:
    DynamicCache = None  # type: ignore
    USE_DYNAMIC_CACHE = False


def dc_num_layers(dc):
    if USE_DYNAMIC_CACHE and isinstance(dc, DynamicCache):
        return len(dc.layers) if hasattr(dc, "layers") else len(dc.key_cache)
    return len(dc)


def dc_get_kv(dc, idx):
    if USE_DYNAMIC_CACHE and isinstance(dc, DynamicCache):
        if hasattr(dc, "layers"):
            layer = dc.layers[idx]
            return layer.keys, layer.values
        return dc.key_cache[idx], dc.value_cache[idx]
    return dc[idx]


def dc_update(dc, k, v, idx):
    if USE_DYNAMIC_CACHE and isinstance(dc, DynamicCache):
        dc.update(k, v, idx)
    else:
        dc.append((k, v))


def dc_from_legacy(pkv):
    if USE_DYNAMIC_CACHE and not isinstance(pkv, DynamicCache):
        dc = DynamicCache()
        for i, (k, v) in enumerate(pkv):
            dc.update(k, v, i)
        return dc
    return pkv


def dc_repeat(dc, B: int):
    if USE_DYNAMIC_CACHE and isinstance(dc, DynamicCache):
        out = DynamicCache()
        for li in range(dc_num_layers(dc)):
            k, v = dc_get_kv(dc, li)
            dc_update(out, k.repeat(B, 1, 1, 1), v.repeat(B, 1, 1, 1), li)
        return out
    else:
        return [(k.repeat(B, 1, 1, 1), v.repeat(B, 1, 1, 1)) for (k, v) in dc]


def dc_first_seq_len(dc, default_len: int = 0) -> int:
    try:
        n = dc_num_layers(dc)
        for li in range(n):
            k, v = dc_get_kv(dc, li)
            if k is not None and hasattr(k, "shape") and len(k.shape) >= 3:
                return int(k.shape[2])
        return int(default_len)
    except Exception:
        return int(default_len)


# -----------------------------
# Dataset (keeps rows with shorter docs)
# -----------------------------
class TriviaQADataset(Dataset):
    def __init__(self, path, tokenizer, max_doc_tokens=20_000):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc = rec.get("document", "")
                n_tok = len(tokenizer(doc, add_special_tokens=False).input_ids)
                if n_tok <= max_doc_tokens:
                    # normalize response fields
                    if "responses" not in rec or not isinstance(rec["responses"], list):
                        rec["responses"] = []
                    if not isinstance(rec.get("response", None), str):
                        rec["response"] = ""
                    self.records.append(rec)
        print(f"[Dataset] kept {len(self.records)} samples (≤{max_doc_tokens} tokens)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r.get("question", ""), r.get("document", ""), r.get("responses", []), r.get("response", "")


def str_collate(batch):
    qs, ds, rlists, rstrs = zip(*batch)
    return list(qs), list(ds), list(rlists), list(rstrs)


# -----------------------------
# Phase helpers
# -----------------------------
def build_latent_tokens(L: int) -> List[str]:
    return [f"<|Latent{i}|>" for i in range(1, L + 1)]


def build_special_seq(latents: List[str]) -> str:
    # Concatenate without spaces, consistent with additional_special_tokens usage
    return "".join(latents)


def find_special_span(ids: torch.Tensor, special_id_set: set) -> Optional[Tuple[int, int]]:
    positions = [i for i, t in enumerate(ids.tolist()) if t in special_id_set]
    if not positions:
        return None
    return min(positions), max(positions) + 1


def find_last_subsequence(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle or not hay or len(needle) > len(hay):
        return None
    n = len(needle)
    for i in range(len(hay) - n, -1, -1):
        if hay[i:i+n] == needle:
            return i
    return None


def main():
    args = build_argparser().parse_args()

    # Seed & wandb
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name or "latentcomp-phased", config=vars(args))

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir_tokenizer,
        use_fast=True,
    )
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Add specials if requested
    added_specials: List[str] = []
    special_id_set: set = set()
    if args.include_specials:
        added_specials = build_latent_tokens(args.latent_token_num)
        tok.add_special_tokens({"additional_special_tokens": added_specials})
        special_id_set = set(tok.convert_tokens_to_ids(t) for t in added_specials)
        print(f"[Tokenizer] Added {len(added_specials)} latent specials")
    else:
        print("[Tokenizer] include_specials=False → no additional specials added")

    # Model (partial-layer-capable remote code in this folder)
    is_mistral = ("mistral" in args.model_name.lower())
    if is_mistral:
        from modeling_mistral_partial import MistralForCausalLM as PartialModel
    else:
        from modeling_partial_layer import LlamaForCausalLM as PartialModel

    model = PartialModel.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir_model,
    )

    # Prefer FA2, then sdpa, then eager
    def _prefer_attn_impl(m):
        for impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                m.config.attn_implementation = impl
                setattr(m.config, "_attn_implementation", impl)
                return True
            except Exception:
                continue
        return False
    _ = _prefer_attn_impl(model)

    model.resize_token_embeddings(len(tok))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.eos_token_id

    # LoRA config / freezing
    lora_enabled = bool(args.use_lora)
    if not args.include_specials and not lora_enabled:
        print("[Note] include_specials=False with no LoRA → enabling LoRA to allow learning")
        lora_enabled = True

    if lora_enabled:
        from peft import LoraConfig, get_peft_model, TaskType
        lconf = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"],
        )
        model = get_peft_model(model, lconf)
    else:
        for p in model.parameters():
            p.requires_grad = False

    # Only allow gradients on special token embedding rows (if specials are used)
    inp_emb = model.get_input_embeddings()
    if args.include_specials:
        inp_emb.weight.requires_grad = True
        specials = set(tok.convert_tokens_to_ids(t) for t in (tok.additional_special_tokens or []))

        def mask_grad(grad):
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            out = torch.zeros_like(grad)
            H = out.size(0)
            for sid in specials:
                if sid is not None and 0 <= sid < H:
                    out[sid] = grad[sid]
            return out

        inp_emb.weight.register_hook(mask_grad)
    else:
        # No specials → do not train embeddings
        inp_emb.weight.requires_grad = False

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        trainable = [inp_emb.weight]  # safety
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    accelerator = Accelerator()
    device = accelerator.device

    # Dataset
    full_set = TriviaQADataset(args.data_file, tok, max_doc_tokens=args.max_doc_tokens)
    val_size = max(1, int(0.1 * len(full_set)))
    train_size = len(full_set) - val_size
    train_set, val_set = random_split(full_set, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=str_collate)
    val_loader   = DataLoader(val_set,   batch_size=1,               shuffle=False, collate_fn=str_collate)

    # Prepare accelerate (avoid moving already-sharded model)
    if hasattr(model, "hf_device_map") and isinstance(getattr(model, "hf_device_map"), dict):
        optimizer, train_loader, val_loader = accelerator.prepare(optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # Schedulers
    num_train_examples = max(1, len(train_set))
    total_training_steps = max(1, num_train_examples * args.epochs)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * total_training_steps)

    def build_scheduler(opt):
        name = args.lr_scheduler
        if name == "cosine":
            return get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps), "step"
        if name == "linear":
            return get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps), "step"
        if name == "one_cycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr, total_steps=total_training_steps,
                pct_start=args.one_cycle_pct_start, div_factor=args.one_cycle_div_factor,
                final_div_factor=args.one_cycle_final_div_factor, anneal_strategy="cos"
            ), "step"
        if name == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma), "step"
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma), "step"
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                              patience=args.plateau_patience, factor=args.plateau_factor, verbose=True), "epoch"
        if name == "none":
            return None, None
        raise ValueError(f"Unknown lr_scheduler: {name}")

    scheduler, sched_mode = build_scheduler(optimizer)

    # Phase 0: System cache
    system_prompt = "You are a helpful assistant that answers questions based on the given document. "
    sys_chat = tok.apply_chat_template([{"role": "system", "content": system_prompt}], tokenize=False)
    sys_ids = tok(sys_chat, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    sys_mask = torch.ones_like(sys_ids)
    with torch.no_grad():
        sys_out = model(input_ids=sys_ids, attention_mask=sys_mask, use_cache=True, return_dict=True)
    system_cache = dc_from_legacy(sys_out.past_key_values)
    L_sys = sys_ids.size(1)
    n_layers = dc_num_layers(system_cache)

    # Best saver
    best_dir = Path(args.save_best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    min_val = float("inf")
    best_epoch = -1

    # Helper: build Phase 1 cache (returns combined cache, past_len, pos_base, user_prefix_for_phase2)
    def build_phase1_cache(document: str, question: str):
        # Construct Phase 1 user content according to flags
        specials_text = build_special_seq(added_specials) if args.include_specials else ""
        if args.include_specials:
            if args.include_query:
                user_p1 = f"Document:\n{document}\n\nQuestion: {question}\n{specials_text}Question: {question}"
            else:
                user_p1 = f"Document:\n{document}\n\n{specials_text}Question: {question}"
        else:
            # include_specials=False → always Document + Question
            user_p1 = f"Document:\n{document}\n\nQuestion: {question}"

        msgs1 = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_p1}]
        full_ids1 = tok(tok.apply_chat_template(msgs1, tokenize=False), add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

        # Identify slice to keep (Specials+trailing Q) or (Q only)
        if args.include_specials:
            span = find_special_span(full_ids1, special_id_set)
            if span is None:
                # Fallback: if specials not found (edge-case), keep last Question only
                q_ids = tok(f"Question: {question}", add_special_tokens=False).input_ids
                start_q = find_last_subsequence(full_ids1.tolist(), q_ids)
                if start_q is None:
                    # worst-case: keep nothing (will return None cache)
                    return None
                keep_start = start_q
            else:
                sp_start, sp_end = span
                keep_start = sp_start
        else:
            q_ids = tok(f"Question: {question}", add_special_tokens=False).input_ids
            start_q = find_last_subsequence(full_ids1.tolist(), q_ids)
            if start_q is None:
                return None
            keep_start = start_q

        # Split user tokens (exclude system) into prefix and tail
        # user_all = tokens after system
        user_all = full_ids1[L_sys:]
        # keep_start within full_ids1; prefix within user_all starts at index (keep_start - L_sys)
        rel_keep_start = max(0, int(keep_start - L_sys))
        tail_ids = user_all[rel_keep_start:]

        with torch.no_grad():
            out_user = model(
                input_ids=user_all.unsqueeze(0),
                attention_mask=torch.ones(1, L_sys + user_all.size(0), device=device),
                past_key_values=system_cache,
                use_cache=True,
                return_dict=True,
            )
        user_pkv = dc_from_legacy(out_user.past_key_values)

        # Extract only tail (Specials+Q or Q-only)
        K = int(tail_ids.size(0))
        combined = DynamicCache() if (USE_DYNAMIC_CACHE and isinstance(system_cache, DynamicCache)) else []
        for li in range(n_layers):
            k_sys, v_sys = dc_get_kv(system_cache, li)
            k_all, v_all = dc_get_kv(user_pkv, li)
            k_tail = k_all[:, :, -K:, :] if K > 0 else k_all[:, :, :0, :]
            v_tail = v_all[:, :, -K:, :] if K > 0 else v_all[:, :, :0, :]
            dc_update(combined, torch.cat([k_sys, k_tail], dim=2), torch.cat([v_sys, v_tail], dim=2), li)

        # Compute position base: add gap to preserve original geometry
        past_len = dc_first_seq_len(combined, default_len=(L_sys + K))
        gap = max(0, int(keep_start - L_sys))  # tokens between system end and start of kept tail
        pos_base = (past_len + gap) if args.pos_mode == "original" else past_len

        # Phase 2 user prefix text (without Document): Specials+Question or Question-only
        if args.include_specials:
            user_p2 = f"{specials_text}Question: {question}"
        else:
            user_p2 = f"Question: {question}"

        return {
            "combined": combined,
            "past_len": past_len,
            "pos_base": pos_base,
            "user_p2": user_p2,
        }

    # Compute loss for a single sample (single response)
    def forward_one(question: str, document: str, response: str):
        ph1 = build_phase1_cache(document, question)
        if ph1 is None:
            # No useful span found; return zero-loss (skip)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Build Phase 2 messages
        msgs2 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ph1["user_p2"]},
            {"role": "assistant", "content": response},
        ]
        full_ids2 = tok(tok.apply_chat_template(msgs2, tokenize=False), add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

        # Extract assistant ids (suffix after sys+user)
        ids_sys_user = tok(tok.apply_chat_template(msgs2[:-1], tokenize=False), add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        prefix_len = ids_sys_user.size(0)
        assistant_ids = full_ids2[prefix_len:]
        if assistant_ids.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        B = 1
        La = assistant_ids.size(0)
        input_ids = assistant_ids.unsqueeze(0)
        labels = assistant_ids.unsqueeze(0).clone()
        pos_ids = torch.arange(ph1["pos_base"], ph1["pos_base"] + La, device=device, dtype=torch.long).unsqueeze(0)

        batch_cache = dc_repeat(ph1["combined"], B)
        attn_mask = torch.cat([
            torch.ones(B, ph1["past_len"], device=device, dtype=torch.long),
            torch.ones(B, La, device=device, dtype=torch.long)
        ], dim=1)

        out = model(
            input_ids=input_ids,
            past_key_values=batch_cache,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            labels=labels,
            return_dict=True,
        )
        return out.loss

    # Train/valid loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum, tr_count = 0.0, 0
        for qs, ds, rlists, rstrs in train_loader:
            for i in range(len(qs)):
                # choose a response: prefer provided single, else first from list
                resp = rstrs[i] if isinstance(rstrs[i], str) and rstrs[i] else (rlists[i][0] if rlists[i] else "")
                loss = forward_one(qs[i], ds[i], resp)
                if torch.isnan(loss) or torch.isinf(loss):
                    wandb.log({"train/nan_or_inf": 1})
                    continue

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)

                # sanitize embedding grad if present
                if inp_emb.weight.grad is not None:
                    inp_emb.weight.grad.data = torch.nan_to_num(inp_emb.weight.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

                # optional grad clipping
                try:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                except Exception:
                    pass

                optimizer.step()
                if scheduler is not None and sched_mode == "step":
                    scheduler.step()

                global_step += 1
                tr_sum += float(loss.item())
                tr_count += 1
                wandb.log({"step": global_step, "train/loss": float(loss.item()), "lr": optimizer.param_groups[0]["lr"]})

        train_avg = (tr_sum / max(1, tr_count))

        # Validation
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for qs, ds, rlists, rstrs in val_loader:
                for i in range(len(qs)):
                    resp = rstrs[i] if isinstance(rstrs[i], str) and rstrs[i] else (rlists[i][0] if rlists[i] else "")
                    loss = forward_one(qs[i], ds[i], resp)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    val_sum += float(loss.item())
                    val_count += 1

        val_avg = (val_sum / max(1, val_count)) if val_count > 0 else float("inf")
        wandb.log({"epoch": epoch, "train/avg": train_avg, "valid/avg": None if val_avg==float("inf") else val_avg})

        if scheduler is not None and sched_mode == "epoch":
            scheduler.step(val_avg)

        print(f"[Epoch {epoch}] train_avg={train_avg:.4f} | valid_avg={val_avg if val_avg!=float('inf') else 'inf'}")

        # Save best (base-only under best/base, LoRA under best/lora)
        if accelerator.is_main_process and val_avg < min_val:
            min_val = val_avg
            best_epoch = epoch
            # clear directory
            for p in best_dir.glob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)

            # unwrap base
            to_save = accelerator.unwrap_model(model)
            base_dir = best_dir / "base"
            base_dir.mkdir(parents=True, exist_ok=True)
            try:
                base_model = to_save.get_base_model() if hasattr(to_save, "get_base_model") else to_save
            except Exception:
                base_model = to_save

            # avoid leaking remote-code auto_map into base config
            try:
                setattr(base_model.config, "auto_map", None)
            except Exception:
                pass

            base_model.save_pretrained(base_dir, safe_serialization=True)

            # Save tokenizer at root (for chat template and specials)
            try:
                tok.save_pretrained(best_dir)
            except Exception:
                pass

            # Save LoRA if attached
            try:
                from peft import PeftModel
                if isinstance(to_save, PeftModel):
                    (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                    to_save.save_pretrained(best_dir / "lora")
            except Exception:
                pass

            # generation_config for convenience
            try:
                if getattr(base_model, "generation_config", None) is not None:
                    base_model.generation_config.to_json_file(str(best_dir / "generation_config.json"))
            except Exception:
                pass

            print(f"[Best] epoch {epoch} saved → {best_dir}")

    accelerator.wait_for_everyone()

    # Optional Hub push of best_dir
    if accelerator.is_main_process and args.push_to_hub and args.hf_repo_id:
        from huggingface_hub import create_repo, upload_folder
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN not set; export HF_TOKEN=...")
        create_repo(args.hf_repo_id, exist_ok=True, private=args.private_repo, token=token)
        upload_folder(
            repo_id=args.hf_repo_id,
            folder_path=str(best_dir),
            token=token,
            commit_message=f"best epoch {best_epoch} (val={min_val:.4f})",
            allow_patterns=["*"],
        )
        print(f"✅ Hub upload complete: {args.hf_repo_id}")


if __name__ == "__main__":
    main()


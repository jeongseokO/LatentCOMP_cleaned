#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoPA-only trainer (TRI pipeline, PEFT-safe, balanced mode, FIXED teacher forcing)

Balanced(수정 요지):
- System/User(S/U): grad 유지 (응답마다 프리필을 grad로 재계산 → 그래프 공유 없음)
- Assistant: [header + content]를 **한 번에** teacher forcing (write_cache=True)
  * header 구간은 labels=-100로 손실 제외
  * 같은 호출 안에서 KV가 연결되므로 Assistant 토큰이 **S/U/헤더**를 실제 참조

추가:
- --max_response_tokens (기본 1024): 응답(assistant) 토큰 길이 상한

### [Math-aware changes]
- document가 빈 문자열이면 수학 문제로 간주하여, 수학용 system/user 템플릿을 사용
- 데이터셋 로더가 document==""를 허용(기존엔 제거)
"""

from __future__ import annotations
import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple
from copy import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from latentcomp_cleaned.dist_utils import (
    build_accelerator,
    extract_input_embedding_weight,
    extract_module_parameter,
    gather_state_dict,
)


# -----------------------
# STRICT helpers
# -----------------------
def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def _get_inner_model(m):
    if hasattr(m, "module"):
        m = m.module
    return m

def _unwrap_peft(m):
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

def _infer_lora_targets_from_model(model: nn.Module) -> list[str]:
    want = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    seen = set()
    for name, _m in model.named_modules():
        base = name.split(".")[-1]
        if base in want:
            seen.add(base)
    if not seen:
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                base = name.split(".")[-1]
                for w in want:
                    if w in base:
                        seen.add(w)
    return sorted(seen)


# -----------------------
# Custom modeling injection (TRI, STRICT)
# -----------------------
def load_custom_llama_modeling_TRI(modeling_path: Path):
    import importlib.util, sys
    import transformers
    import transformers.models.llama as llama_pkg

    target_name = "transformers.models.llama.modeling_llama"
    sys.modules.pop(target_name, None)

    spec = importlib.util.spec_from_file_location(target_name, str(modeling_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {modeling_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[target_name] = module
    spec.loader.exec_module(module)

    setattr(llama_pkg, "modeling_llama", module)

    for a in ("LlamaModel", "LlamaForCausalLM"):
        _require(hasattr(module, a), f"Patched module missing `{a}` in {modeling_path}")

    need_LM = ["tri_prefill_system_all", "tri_prefill_user_lower", "tri_build_caches", "tri_forward_assistant"]
    for n in need_LM:
        _require(hasattr(module.LlamaModel, n), f"Patched LlamaModel lacks `{n}`")

    need_TOP = ["tri_build_caches", "tri_forward_assistant", "tri_step_logits"]
    for n in need_TOP:
        _require(hasattr(module.LlamaForCausalLM, n), f"Patched LlamaForCausalLM lacks `{n}`")

    print("[DEBUG] modeling_llama path:", modeling_path)
    print("[DEBUG] TRI bound:",
          {n: hasattr(module.LlamaModel, n) for n in need_LM},
          {n: hasattr(module.LlamaForCausalLM, n) for n in need_TOP})
    return module


# -----------------------
# Argparser
# -----------------------
def build_argparser():
    p = argparse.ArgumentParser("LoPA trainer (TRI pipeline, balanced)")
    # core
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefill_layers", type=int, default=16, help="K (lower layers) used for User prefill")

    # modeling file
    p.add_argument("--lopa_modeling_path", type=str, default="lopa_llama_modeling.py",
                   help="Path to TRI-enabled modeling_llama.py")

    # LoRA (optional)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", nargs="*",
                   default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    # backend
    p.add_argument("--attn_impl", type=str, choices=["eager", "sdpa", "flash_attention_2"], default="flash_attention_2")

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
    p.add_argument("--max_response_tokens", type=int, default=1024,
                   help="cap assistant tokens per sample to this many tokens")
    p.add_argument("--num_specials", type=int, default=0,
                   help="Number of Latent special tokens to inject per sample.")
    p.add_argument("--special_add_to", type=str, choices=["none", "user", "assistant"], default="none",
                   help="Where to place Latent specials (user tail or assistant head).")
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
    p.add_argument("--hf_repo_org", type=str, default=None)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")

    # distributed
    p.add_argument("--dist_mode", type=str, choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    p.add_argument("--zero_stage", type=int, default=2)

    # responses execution mode
    p.add_argument("--responses_sequential", action="store_true", default=True,
                   help="각 응답을 독립 그래프로 순차 학습(backward 포함)")
    p.add_argument("--no_responses_sequential", dest="responses_sequential", action="store_false")

    p.add_argument("--train_method", type=str, default="lopa")

    ### [Math-aware changes]
    p.add_argument("--math_force_final_hash_rule", action="store_true", default=False,
                   help="수학 system 프롬프트에 '#### 숫자' 규칙을 명시적으로 강화")

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
                d = (rec.get("document") or "")
                # ### [Math-aware changes] — document가 비어도 허용
                d = d.strip()

                if not q:
                    continue  # 질문은 반드시 필요

                # 문서 토큰 길이 필터 (비어있으면 0이므로 통과)
                try:
                    n_tok = len(tokenizer(d, add_special_tokens=False).input_ids)
                except Exception:
                    n_tok = 0
                if n_tok > max_doc_tokens:
                    continue

                # 응답 수집
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

                qid = rec.get("question_id", rec.get("id", None))
                if qid is None:
                    qid = f"auto_{auto_idx}"; auto_idx += 1
                try:
                    qid = str(qid)
                except Exception:
                    qid = f"auto_{auto_idx}"; auto_idx += 1

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

### [Math-aware changes] — 메시지 빌더를 DocQA/Math로 분기
def build_messages_docqa(system: str, document: str, question: str):
    user = f"Document:\n{document}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def build_messages_math(system: str, question: str):
    user = question
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

def lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
    L = min(a.size(1), b.size(1))
    eq = (a[0, :L] == b[0, :L])
    nz = (~eq).nonzero(as_tuple=False)
    return int(nz[0, 0]) if nz.numel() else L


# -----------------------
# Accelerator builder
# -----------------------
def _fork_cache_view(pkv):
    new = copy(pkv)
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        new.key_cache  = list(pkv.key_cache)
        new.value_cache = list(pkv.value_cache)
    elif hasattr(pkv, "layers"):
        new.layers = list(getattr(pkv, "layers") or [])
    return new


# -----------------------
# Train (TRI, BALANCED+TF FIX)
# -----------------------
def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # 1) Patch modeling BEFORE loading model
    modeling_path = Path(getattr(args, "lopa_modeling_path", "lopa_llama_modeling.py")).resolve()
    _require(modeling_path.exists(), f"lopa_modeling_path not found: {modeling_path}")
    llama_mod = load_custom_llama_modeling_TRI(modeling_path)
    LlamaForCausalLM = llama_mod.LlamaForCausalLM  # noqa

    wrap_cls = None
    if hasattr(llama_mod, "LlamaDecoderLayer"):
        wrap_cls = (llama_mod.LlamaDecoderLayer,)

    accelerator = build_accelerator(args, fsdp_wrap_cls=wrap_cls)
    device = accelerator.device

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    num_specials = max(0, int(getattr(args, "num_specials", 0)))
    special_add_target = (getattr(args, "special_add_to", "none") or "none").lower()
    if special_add_target not in {"user", "assistant"}:
        special_add_target = "none"
    special_tokens = [f"<|Latent{i}|>" for i in range(1, num_specials + 1)] if num_specials > 0 else []
    specials_joined = " ".join(special_tokens) if special_tokens else ""
    train_special_embeddings = bool(special_tokens) and special_add_target in {"user", "assistant"}
    tokens_to_add = [tok for tok in special_tokens if tok not in tokenizer.all_special_tokens]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    # 3) dtype
    if args.dtype == "fp32": dtype = torch.float32
    elif args.dtype == "bf16": dtype = torch.bfloat16
    elif args.dtype == "fp16": dtype = torch.float16
    else:
        dtype = (torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
                 else (torch.float16 if device == "cuda" else torch.float32))
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else dtype
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
            torch.backends.cudnn.allow_tf32 = not args.no_tf32
        except Exception:
            pass
        if args.sdpa_math_only:
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

    # 4) Model
    ds_plugin = None
    accel_state = getattr(accelerator, "state", None)
    if accel_state is not None:
        ds_plugin = getattr(accel_state, "deepspeed_plugin", None)
    init_ctx = nullcontext()
    if ds_plugin is not None and getattr(ds_plugin, "zero_stage", 0) >= 3:
        init_ctx = getattr(ds_plugin, "zero3_init_context_manager", lambda: nullcontext())()
    with init_ctx:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            cache_dir=args.cache_dir_model,
        )
    if train_special_embeddings:
        try:
            current = model.get_input_embeddings().weight.size(0)
        except Exception:
            current = None
        if current is None or current != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    # 5) Attention impl
    impl = args.attn_impl
    try:
        model.config._attn_implementation = impl
        _get_inner_model(model).model.config._attn_implementation = impl
    except Exception:
        pass
    if accelerator.is_main_process:
        print(f"[Info] attn_implementation = {impl}")

    # 6) TRI instance-level checks
    for need in ("tri_build_caches", "tri_forward_assistant", "tri_step_logits"):
        _require(hasattr(model, need), f"LlamaForCausalLM instance lacks `{need}`. TRI injection failed.")

    # 7) LoRA (optional)
    use_lora = bool(getattr(args, "use_lora", False))
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        cli_targets = list(getattr(args, "lora_targets", []))
        if not cli_targets:
            core = _unwrap_peft(_get_inner_model(model))
            found = _infer_lora_targets_from_model(core)
            cli_targets = found or ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

        if accelerator.is_main_process:
            print(f"[LoRA] target_modules = {cli_targets}")

        lcfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=cli_targets,
        )
        model = get_peft_model(model, lcfg)

    if train_special_embeddings:
        input_emb = model.get_input_embeddings()
        if input_emb is not None:
            input_emb.weight.requires_grad = True
        output_emb = getattr(model, "lm_head", None)
        if output_emb is not None and hasattr(output_emb, "weight"):
            output_emb.weight.requires_grad = True

    # 8) Data
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
    accel_state = getattr(accelerator, "state", None)
    ds_engine = None
    if accel_state is not None:
        ds_engine = getattr(accel_state, "deepspeed_engine", None)
    if ds_engine is None and hasattr(model, "deepspeed_engine"):
        ds_engine = getattr(model, "deepspeed_engine")
    inference_model = None
    if ds_engine is not None and hasattr(ds_engine, "module"):
        inference_model = ds_engine.module
    if inference_model is None and hasattr(model, "module"):
        inference_model = model.module
    if inference_model is None:
        inference_model = model

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

    ### [Math-aware changes] — 두 가지 시스템 프롬프트
    system_prompt_doc = (
        "You are a helpful assistant that answers questions based on the given document."
    )
    system_prompt_math = (
        "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {numeric answer}' format, where the answer is solely a number."
    )

    K = max(0, int(args.prefill_layers))  # lower_k
    MAX_R = max(1, int(getattr(args, "max_response_tokens", 1024)))

    # -----------------------
    # Chat-template TRI helpers
    # -----------------------
    def _build_segments(q: str, d: str):
        """
        Return:
          msgs, S_ids, SU_ids, SU_gen, user_delta, assistant_header_delta
        with Math/DocQA-aware messages.
        """
        is_math = (len(d.strip()) == 0)

        if is_math:
            msgs = build_messages_math(system_prompt_math, q)
        else:
            msgs = build_messages_docqa(system_prompt_doc, d, q)

        if specials_joined and special_add_target == "user":
            user_content = msgs[-1]["content"]
            sep = "\n\n" if not user_content.endswith("\n") else ""
            msgs[-1]["content"] = f"{user_content}{sep}{specials_joined}"

        S_ids   = tokens_from_messages(tokenizer, msgs[:1], accelerator.device, add_generation_prompt=False)
        SU_ids  = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=False)
        SU_gen  = tokens_from_messages(tokenizer, msgs, accelerator.device, add_generation_prompt=True)

        l_su = lcp_len(S_ids, SU_ids)
        user_delta = SU_ids[:, l_su:SU_ids.size(1)]
        assistant_header_delta = SU_gen[:, SU_ids.size(1):]  # header only
        return msgs, S_ids, SU_ids, SU_gen, user_delta, assistant_header_delta

    def _assistant_content_delta(msgs, resp: str, SU_gen_ids):
        if specials_joined and special_add_target == "assistant":
            if resp:
                lead_space = " " if not resp[0].isspace() else ""
                resp = f"{specials_joined}{lead_space}{resp}"
            else:
                resp = specials_joined
        msgs_ass = msgs + [{"role": "assistant", "content": resp}]
        full_ids = tokens_from_messages(tokenizer, msgs_ass, accelerator.device, add_generation_prompt=False)
        assistant_delta = full_ids[:, SU_gen_ids.size(1):]
        if assistant_delta.size(1) > MAX_R:
            assistant_delta = assistant_delta[:, :MAX_R]
        return assistant_delta

    # -----------------------
    # Core losses (teacher forcing: header+content, write_cache=True)
    # -----------------------
    def _tf_call(pkv, S_len: int, U_len: int, header_delta: torch.Tensor, assistant_delta: torch.Tensor):
        if assistant_delta.size(1) > MAX_R:
            assistant_delta = assistant_delta[:, :MAX_R]
        if assistant_delta.size(1) < 2:
            return None
        inp = torch.cat([header_delta, assistant_delta], dim=1)          # [H + A]
        labels = inp.clone()
        labels[:, :header_delta.size(1)] = -100                           # header는 로스 제외
        out = inference_model.tri_step_logits(
            assistant_ids=inp, lower_k=K, pkv=pkv, S=S_len, U=U_len,
            logits_to_keep=inp.size(1), labels=labels, write_cache=True
        )
        return out

    def compute_loss_on_sample(q: str, d: str, resp: str) -> torch.Tensor:
        msgs, S_ids, SU_ids, SU_gen, user_delta, header_delta = _build_segments(q, d)

        # (1) S/U 프리필 — grad ON (응답별 독립 그래프)
        pkv_su, S_len, U_len = inference_model.tri_build_caches(system_ids=S_ids, user_ids=user_delta, lower_k=K)

        # (2) Assistant delta 구성 (+ cap)
        assistant_delta = _assistant_content_delta(msgs, resp, SU_gen)
        if assistant_delta.size(1) < 2:
            return torch.zeros((), device=accelerator.device, dtype=torch.float32)

        # (3) header+content 한 번에 teacher forcing
        out = _tf_call(pkv_su, S_len, U_len, header_delta, assistant_delta)
        if out is None or out.loss is None:
            return torch.zeros((), device=accelerator.device, dtype=torch.float32)
        return out.loss

    def compute_loss_on_group_mean(qid: str, q: str, d: str, responses: List[str]) -> torch.Tensor:
        losses = []
        with torch.no_grad():
            msgs, S_ids, SU_ids, SU_gen, user_delta, header_delta = _build_segments(q, d)
            for resp in responses:
                if not isinstance(resp, str) or not resp.strip(): continue
                pkv_su, S_len, U_len = inference_model.tri_build_caches(S_ids, user_delta, lower_k=K)
                ad = _assistant_content_delta(msgs, resp, SU_gen)
                if ad.size(1) < 2:
                    continue
                out = _tf_call(pkv_su, S_len, U_len, header_delta, ad)
                if out is not None and out.loss is not None and torch.isfinite(out.loss):
                    losses.append(out.loss.detach())
        _require(len(losses) > 0, "All responses invalid/empty for this item.")
        return torch.stack(losses, dim=0).mean()

    def compute_loss_on_group_seq(qid: str, q: str, d: str, responses: List[str]) -> float:
        msgs, S_ids, SU_ids, SU_gen, user_delta, header_delta = _build_segments(q, d)

        eligible = []
        for resp in responses:
            if not isinstance(resp, str) or not resp.strip(): continue
            ad = _assistant_content_delta(msgs, resp, SU_gen)
            if ad.size(1) < 2:
                continue
            eligible.append(ad)
        _require(len(eligible) > 0, "All responses invalid/empty for this item.")

        losses_for_log, eff_total = [], len(eligible)
        bs = max(1, args.batch_size)

        for ad in eligible:
            pkv_su, S_len, U_len = inference_model.tri_build_caches(system_ids=S_ids, user_ids=user_delta, lower_k=K)
            out = _tf_call(pkv_su, S_len, U_len, header_delta, ad)
            _require(out is not None and out.loss is not None and torch.isfinite(out.loss), "loss is NaN/inf")
            lval = float(out.loss.detach().item())
            losses_for_log.append(lval)
            accelerator.backward(out.loss / eff_total / bs)
            del pkv_su, out

        return sum(losses_for_log) / eff_total

    # wandb (optional)
    run = None
    wandb_name = getattr(args, "wandb_run_name", None) or f"lopa_tri_balanced_K{K}_R{MAX_R}"
    if accelerator.is_main_process and args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=wandb_name, config=vars(args))
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
            RESP_SEQ = bool(getattr(args, "responses_sequential", True))

            for it in items:
                if len(it) == 4 and isinstance(it[3], list):
                    qid, q, d, rs = it
                    if RESP_SEQ:
                        loss_val = compute_loss_on_group_seq(str(qid), q, d, rs)
                        step_losses.append(loss_val); valid += 1
                        continue
                    else:
                        loss_i = compute_loss_on_group_mean(str(qid), q, d, rs)
                elif len(it) == 4:
                    qid, q, d, r = it
                    loss_i = compute_loss_on_sample(q, d, r)
                    accelerator.backward(loss_i / max(1, args.batch_size))
                else:
                    q, d, r = it
                    loss_i = compute_loss_on_sample(q, d, r)
                    accelerator.backward(loss_i / max(1, args.batch_size))

                step_losses.append(float(loss_i.detach().item())); valid += 1

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
                state_dict = gather_state_dict(accelerator, model)
                try:
                    to_unwrap = accelerator.unwrap_model(model)
                except Exception:
                    to_unwrap = model

                is_peft = False
                try:
                    from peft import PeftModel
                    is_peft = isinstance(to_unwrap, PeftModel)
                except Exception:
                    pass

                if is_peft:
                    from transformers.models.llama.modeling_llama import LlamaForCausalLM as _LM
                    base_clean = _LM.from_pretrained(args.model_name, torch_dtype=dtype, cache_dir=args.cache_dir_model)
                    if train_special_embeddings:
                        base_clean.resize_token_embeddings(len(tokenizer))
                        embed_weight = extract_input_embedding_weight(to_unwrap, state_dict)
                        clean_inp = base_clean.get_input_embeddings()
                        if embed_weight is not None and clean_inp is not None:
                            clean_inp.weight.data.copy_(embed_weight.to(clean_inp.weight.dtype))
                        trained_base = _unwrap_peft(to_unwrap)
                        trained_head = getattr(trained_base, "lm_head", None)
                        clean_head = getattr(base_clean, "lm_head", None)
                        if clean_head is not None and trained_head is not None and hasattr(clean_head, "weight"):
                            head_weight = extract_module_parameter(to_unwrap, trained_head, state_dict, "weight")
                            if head_weight is not None:
                                clean_head.weight.data.copy_(head_weight.to(clean_head.weight.dtype))
                    base_clean.save_pretrained(base_dir, safe_serialization=True)
                    (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                    try:
                        from peft import get_peft_model_state_dict
                        adapter_state = get_peft_model_state_dict(to_unwrap)
                        adapter_state = {k: v.detach().cpu() for k, v in adapter_state.items()}
                    except Exception:
                        adapter_state = None
                    to_unwrap.save_pretrained(best_dir / "lora", state_dict=adapter_state)
                else:
                    to_unwrap.save_pretrained(base_dir, safe_serialization=True, state_dict=state_dict)
            except Exception as e:
                raise RuntimeError(f"Failed to save best: {e}")

            try:
                tokenizer.save_pretrained(best_dir)
                with open(best_dir / "tri_info.txt", "w") as f:
                    f.write(f"lower_k={K}\n")
                    f.write(f"max_response_tokens={MAX_R}\n")
                    f.write("rope=global (TRI)\n")
                    f.write("teacher_forcing=header+content write_cache=True\n")
                    f.write("zero_padding=off, alpha=off\n")
                    f.write("prompt_modes=DocQA|Math(#### rule)\n")
            except Exception:
                pass

            print(f"[Best] Saved to {best_dir} (val={best_loss:.6f})")

        accelerator.wait_for_everyone()

    # push to hub
    if accelerator.is_main_process and args.push_to_hub:
        token = os.environ.get("HF_TOKEN", "")
        _require(bool(token), "HF_TOKEN not set; export HF_TOKEN=...")

        repo_id = getattr(args, "hf_repo_id", None)
        if not repo_id:
            org = getattr(args, "hf_repo_org", None)
            base_name = Path(args.save_best_dir).resolve().name or "lopa-tri-balanced-tf"
            repo_id = f"{org}/{base_name}" if org else base_name

        from huggingface_hub import create_repo, upload_folder
        create_repo(repo_id, exist_ok=True, private=bool(getattr(args, "private_repo", False)), token=token)
        upload_folder(repo_id=repo_id, folder_path=str(best_dir),
                      token=token, commit_message="LoPA(TRI, teacher forcing fixed, response cap, math-aware prompts)", allow_patterns=["*"])
        print(f"✅ Uploaded to Hub: {repo_id}")


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

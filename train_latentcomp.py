#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_with_rope_position_original_geometry_batched_responses.py
# ──────────────────────────────────────────────────────────────────────────
# Original-geometry RoPE 학습 (원본 틀 유지) + SUPPORT/단일응답/헤더 옵션 (개정판)
# ──────────────────────────────────────────────────────────────────────────

import gc
import os, json, random, shutil, math, argparse
import sys
from pathlib import Path
from typing import List, Tuple, Any
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from accelerate import Accelerator
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from huggingface_hub import create_repo, upload_folder
import wandb

# Ensure we import modeling_* from this cleaned folder (isolate from legacy tree)
try:
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent
    for _p in (_ROOT, _HERE):
        _ps = str(_p)
        if _ps not in sys.path:
            sys.path.insert(0, _ps)
except Exception:
    pass

# DynamicCache 지원 여부
try:
    from transformers import DynamicCache
    USE_DYNAMIC_CACHE = True
except ImportError:
    USE_DYNAMIC_CACHE = False
    print("DynamicCache not available, using legacy format")

# ──────────────────────────────────────────────────────────────────────────
# argparse 유틸
# ──────────────────────────────────────────────────────────────────────────
def str2bool(v):
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):  return True
    if s in ("false", "f", "0", "no", "n"):  return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")

def build_argparser():
    p = argparse.ArgumentParser(
        description="Original-geometry RoPE + (Latent, SUPPORT) training with fixed Latent count and uniform SUPPORT blocks"
    )
    # 핵심
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    # LR Scheduler 옵션
    p.add_argument("--lr_scheduler", type=str, choices=["cosine","linear","one_cycle","exponential","step","plateau","none"], default="cosine")
    p.add_argument("--warmup_steps", type=int, default=0, help="0이면 warmup_ratio로 자동 계산")
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="총 step 대비 비율(0이면 미사용)")
    p.add_argument("--min_lr", type=float, default=0.0, help="(예약) 하한")
    p.add_argument("--lr_gamma", type=float, default=0.5, help="exponential/step 감쇠 계수")
    p.add_argument("--lr_step_size", type=int, default=1, help="stepLR 주기(스텝 단위)")
    p.add_argument("--plateau_patience", type=int, default=1)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--one_cycle_pct_start", type=float, default=0.1)
    p.add_argument("--one_cycle_div_factor", type=float, default=25.0)
    p.add_argument("--one_cycle_final_div_factor", type=float, default=1e4)

    # HF Hub
    p.add_argument("--hf_repo_id", type=str, default=None,
                   help="업로드할 레포 (예: user/repo). 내부에서 접미사가 붙을 수 있음")
    p.add_argument("--push_to_hub", action="store_true", help="지정 시 Hub에 업로드")
    p.add_argument("--private_repo", action="store_true", help="지정 시 private 레포로 생성")

    # 옵션
    p.add_argument("--include_query", type=str2bool, default=True)
    p.add_argument("--max_doc_tokens", type=int, default=12500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--pos_mode", type=str, choices=["original","compressed"], default="original")

    p.add_argument("--support", type=str2bool, default=False, help="SUPPORT 블록 사용 여부")
    p.add_argument("--batch", type=str2bool, default=True)
    p.add_argument("--header", type=str, choices=["user","assistant"], default="user")
    p.add_argument("--only_one_token", type=str2bool, default=False, help="SUPPORT 1개 토큰만 사용(Ablation)")
    p.add_argument("--dynamic_support", type=str2bool, default=False, help="전체 문서 길이에 비례해 K(블록당 SUPPORT 개수) 자동 산출")
    p.add_argument("--compress_rate", type=int, default=128, help="Dynamic일 때 K≈ceil(M / (compress_rate * L)) 계산에 사용")

    # 새 옵션
    p.add_argument("--latent_token_num", type=int, default=10)
    p.add_argument("--support_token_num", type=int, default=3)
    p.add_argument("--document_sep", type=str2bool, default=False)

    # 부분 레이어 프리필(Phase1) 옵션
    p.add_argument("--prefill_layers", type=int, default=0, help="Phase1에서 사용할 하위 레이어 수 (0이면 비활성, 전체 레이어)")

    p.add_argument("--save_best_dir", type=str, default="./_best_ckpt")

    # 캐시 디렉토리
    p.add_argument("--cache_dir_tokenizer", type=str, default="/data2/jeongseokoh/hub/tokenizer")
    p.add_argument("--cache_dir_model", type=str, default="/data2/jeongseokoh/hub/model")

    # W&B
    p.add_argument("--wandb_project", type=str, default="rope-original-geometry")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # LoRA 옵션
    p.add_argument("--use_lora", type=str2bool, default=False, help="LoRA 어댑터로 K/V/MLP(gate/up/down) 학습")
    p.add_argument("--lora_r", type=int, default=4, help="LoRA rank (요청: 4)")
    p.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    return p

# ──────────────────────────────────────────────────────────────────────────
# DynamicCache helpers
# ──────────────────────────────────────────────────────────────────────────
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
    else:
        return dc[idx]

def dc_update(dc, k, v, idx):
    if USE_DYNAMIC_CACHE and isinstance(dc, DynamicCache):
        dc.update(k, v, idx)
    else:
        dc.append((k, v))

def dc_from_legacy(pkv_legacy):
    if USE_DYNAMIC_CACHE and not isinstance(pkv_legacy, DynamicCache):
        dc = DynamicCache()
        for li, (k, v) in enumerate(pkv_legacy):
            dc.update(k, v, li)
        return dc
    return pkv_legacy

def dc_repeat(dc, B):
    if USE_DYNAMIC_CACHE and isinstance(dc, DynamicCache):
        out = DynamicCache()
        for li in range(dc_num_layers(dc)):
            k, v = dc_get_kv(dc, li)
            dc_update(out, k.repeat(B, 1, 1, 1), v.repeat(B, 1, 1, 1), li)
        return out
    else:
        return [(k.repeat(B, 1, 1, 1), v.repeat(B, 1, 1, 1)) for (k, v) in dc]

def dc_first_seq_len(dc, default_len: int = 0) -> int:
    """Return sequence length (past length) from the first valid layer in cache.
    Falls back to default_len if none found.
    """
    try:
        n = dc_num_layers(dc)
        for li in range(n):
            k, v = dc_get_kv(dc, li)
            if k is not None and hasattr(k, "shape") and len(k.shape) >= 3:
                return int(k.shape[2])
        return int(default_len)
    except Exception:
        return int(default_len)

# ──────────────────────────────────────────────────────────────────────────
# 1) Dataset
# ──────────────────────────────────────────────────────────────────────────
class TriviaQADataset(Dataset):
    def __init__(self, path, tokenizer, max_doc_tokens=20_000):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                n_tok = len(tokenizer(rec["document"], add_special_tokens=False).input_ids)
                if n_tok <= max_doc_tokens:
                    if "responses" not in rec or not isinstance(rec["responses"], list):
                        rec["responses"] = []
                    if not isinstance(rec.get("response", None), str):
                        rec["response"] = ""
                    self.records.append(rec)
        print(f"[Dataset] kept {len(self.records)} samples (≤{max_doc_tokens} tokens)")

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        return r["question"], r["document"], r["responses"], r["response"]

def str_collate(batch):
    qs, ds, rlists, rstrs = zip(*batch)
    return list(qs), list(ds), list(rlists), list(rstrs)

# ──────────────────────────────────────────────────────────────────────────
def main():
    args = build_argparser().parse_args()

    # seed / wandb
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    include_query_bool = args.include_query
    use_support       = args.support
    use_batch         = args.batch
    only_one          = args.only_one_token
    dynamic_sup       = args.dynamic_support
    header_mode       = "user"  # unify: always user
    # Deprecated: document_sep option is no longer used. We always insert a doc-separator
    # token between per-document SPECIAL chunks only for include_query=False (offline) path.
    use_docsep        = False

    # 모델 플래그
    IS_MISTRAL = ("mistral" in args.model_name.lower())

    # Header is unified to 'user' for all models
    if args.header != "user":
        print("[Header] Forcing header='user' (unified setting)")

    # repo id 접미어
    og_suffix = "" if args.pos_mode == "compressed" else "-og"
    sup_suffix = (f"L{args.latent_token_num}xDyn(cr{args.compress_rate})"
                  if dynamic_sup else f"L{args.latent_token_num}xS{args.support_token_num}")
    docsep_suffix  = ""  # deprecated
    only_one_suffix = "-only-one-token" if only_one else ""
    q_suffix = "-no-query" if not include_query_bool else ""
    # partial-layer prefill 표시
    try:
        pl = int(args.prefill_layers)
    except Exception:
        pl = 0
    partial_suffix = f"-partial{pl}" if pl > 0 else ""
    # Mistral이면 강제로 user 접미사 사용
    effective_header = "user" if IS_MISTRAL else header_mode
    mode_suffix = f"{effective_header}{og_suffix}"
    lora_suffix = f"-lora" if args.use_lora else ""
    base_suffix = f"{mode_suffix}-{sup_suffix}{docsep_suffix}{q_suffix}{only_one_suffix}{partial_suffix}{lora_suffix}"

    HF_REPO_ID = f"{args.hf_repo_id}_{base_suffix}" if args.hf_repo_id else None
    run_name = HF_REPO_ID or f"run-{base_suffix}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4) Special tokens
    support_token = "<|SUPPORT|>"
    doc_sep_token = "<|DOC_SEP|>"

    def build_latent_markers(L: int, use_docsep_flag: bool) -> List[str]:
        # document_sep mode removed; always use Latent markers
        return [f"<|Latent{i}|>" for i in range(1, L + 1)]

    if only_one:
        special_tokens_vocab = [support_token, doc_sep_token]
        print(f"[Config] Only-one 모드: SUPPORT 토큰 1개만 사용 (Latent 임베딩 미사용)")
    else:
        latent_markers = build_latent_markers(args.latent_token_num, use_docsep)
        special_tokens_vocab = latent_markers + ([support_token] if use_support else []) + [doc_sep_token]
        print(f"[Config] Latent {args.latent_token_num}개" + (" + SUPPORT 1개" if use_support else " (SUPPORT 미사용)"))

    print(f"총 special tokens 추가: {len(special_tokens_vocab)}개")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_vocab})

    # 샘플 분포 프린트(일부)
    print("\n실제 데이터에서 문서 수/토큰 길이 분포 확인 중...")
    doc_separator_print = "\n\n----[DOC SEP]----\n\n"
    doc_count_distribution = {}
    with open(args.data_file, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx >= 1000: break
            try:
                rec = json.loads(line)
                document = rec["document"]
                if doc_separator_print in document:
                    docs = [d.strip() for d in document.split(doc_separator_print) if d.strip()]
                    doc_count = len(docs)
                else:
                    doc_count = 1
                doc_count_distribution[doc_count] = doc_count_distribution.get(doc_count, 0) + 1
                if line_idx < 3:
                    total_tokens = len(tokenizer(document, add_special_tokens=False).input_ids)
                    if dynamic_sup:
                        K = max(1, math.ceil(total_tokens / max(1, args.compress_rate * args.latent_token_num)))
                        print(f"샘플 {line_idx}: 문서 {doc_count}개 (총 {total_tokens} 토큰) → Dynamic K={K} (L={args.latent_token_num})")
                    else:
                        K = max(0, args.support_token_num)
                        print(f"샘플 {line_idx}: 문서 {doc_count}개 (총 {total_tokens} 토큰) → Static K={K} (L={args.latent_token_num})")
            except Exception as e:
                print(f"샘플 {line_idx} 파싱 오류: {e}")
                continue

    checked = min(1000, line_idx+1 if 'line_idx' in locals() else 0)
    print(f"\n문서 수 분포 (처음 {checked}개 샘플):")
    for doc_count in sorted(doc_count_distribution.keys()):
        count = doc_count_distribution[doc_count]
        denom = max(1, checked)
        print(f"  {doc_count}개 문서: {count}개 샘플 ({count*100/denom:.1f}%)")

    # 5) Model (single model with partial-layer support)
    if IS_MISTRAL:
        from modeling_mistral_partial import MistralForCausalLM as PartialAwareModel
    else:
        from modeling_partial_layer import LlamaForCausalLM as PartialAwareModel

    model = PartialAwareModel.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir_model,
    )
    # Prefer Flash-Attention-2 across the board; fallback to sdpa → eager
    def _prefer_fa2(m):
        tried = False
        for attr in ("attn_implementation", "_attn_implementation"):
            try:
                setattr(m.config, attr, "flash_attention_2")
                tried = True
            except Exception:
                pass
        # also set on inner base if present (PEFT wrappers)
        try:
            base = m.get_base_model() if hasattr(m, "get_base_model") else None
            if base is not None and hasattr(base, "config"):
                for attr in ("attn_implementation", "_attn_implementation"):
                    try:
                        setattr(base.config, attr, "flash_attention_2")
                    except Exception:
                        pass
        except Exception:
            pass
        # light test: access ALL_ATTENTION_FUNCTIONS mapping may raise if unsupported; if so, set sdpa/eager
        try:
            _ = getattr(model, "config", None)
        except Exception:
            pass
        return tried
    if not _prefer_fa2(model):
        # fallback attempts
        for impl in ("sdpa", "eager"):
            try:
                model.config.attn_implementation = impl
                setattr(model.config, "_attn_implementation", impl)
                break
            except Exception:
                continue

    model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # 단일 모델에서 부분 레이어 프리필/연결을 모두 처리합니다.
    try:
        n_total_layers_cfg = int(getattr(model.config, "num_hidden_layers", 0))
    except Exception:
        n_total_layers_cfg = 0

    # 6) (옵션) LoRA 구성 + Freeze except special-token embeddings
    lora_enabled = bool(args.use_lora)
    if lora_enabled:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as e:
            raise ImportError(f"LoRA를 사용하려면 peft가 필요합니다: {e}")
        # K/V/MLP(targets: k_proj, v_proj, gate_proj, up_proj, down_proj)
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        # get_peft_model은 기본적으로 LoRA 파라미터만 trainable로 설정하고, base는 freeze합니다.
    else:
        # LoRA 미사용 시: 전체 freeze
        for p in model.parameters():
            p.requires_grad = False
    # 특수 토큰 임베딩은 항상 학습
    inp_emb = model.get_input_embeddings()
    inp_emb.weight.requires_grad = True

    special_id_set = set(tokenizer.convert_tokens_to_ids(t) for t in special_tokens_vocab)
    def mask_grad(grad):
        # Sanitize incoming gradients to avoid NaN propagation, then keep only special-token rows
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.zeros_like(grad)
        H = out.size(0)
        for sid in special_id_set:
            if sid is not None and sid >= 0 and sid < H:
                out[sid] = grad[sid]
        return out
    inp_emb.weight.register_hook(mask_grad)

    # Optimizer는 학습 가능한 모든 파라미터(특수 임베딩 + LoRA 어댑터)로 구성
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        trainable_params = [inp_emb.weight]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    accelerator = Accelerator()
    device = accelerator.device

    # 7) Dataset & loaders
    full_set = TriviaQADataset(args.data_file, tokenizer, max_doc_tokens=args.max_doc_tokens)
    val_size   = max(1, int(0.1 * len(full_set)))
    train_size = len(full_set) - val_size
    train_set, val_set = random_split(full_set, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=str_collate)
    val_loader   = DataLoader(val_set,   batch_size=1,               shuffle=False, collate_fn=str_collate)

    # 8) prepare
    # If the model is already sharded across devices via HF device_map, avoid
    # giving it to accelerate (which would try to .to(accelerator.device)).
    if hasattr(model, "hf_device_map") and isinstance(getattr(model, "hf_device_map"), dict):
        optimizer, train_loader, val_loader = accelerator.prepare(optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # 스케줄러
    num_train_examples = len(train_set)
    total_training_steps = max(1, num_train_examples * args.epochs)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * total_training_steps)

    def build_scheduler(optimizer):
        name = args.lr_scheduler
        if name == "cosine":
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps), "step"
        if name == "linear":
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps), "step"
        if name == "one_cycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr, total_steps=total_training_steps,
                pct_start=args.one_cycle_pct_start, div_factor=args.one_cycle_div_factor,
                final_div_factor=args.one_cycle_final_div_factor, anneal_strategy="cos"
            ), "step"
        if name == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma), "step"
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma), "step"
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                              patience=args.plateau_patience, factor=args.plateau_factor, verbose=True), "epoch"
        if name == "none":
            return None, None
        raise ValueError(f"Unknown lr_scheduler: {name}")

    scheduler, sched_mode = build_scheduler(optimizer)

    # 9) Precompute system prompt KV
    system_prompt = "You are a helpful assistant that answers questions based on the given document. "
    system_chat = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}], tokenize=False)
    sys_ids = tokenizer(system_chat, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    sys_mask = torch.ones_like(sys_ids)

    with torch.no_grad():
        sys_out = model(input_ids=sys_ids, attention_mask=sys_mask, use_cache=True, return_dict=True)
    system_cache = dc_from_legacy(sys_out.past_key_values)
    if system_cache is None:
        raise ValueError("system past_key_values is None")
    L_sys = sys_ids.size(1)
    n_layers = dc_num_layers(system_cache)
    # clamp prefill_layers to [0, n_layers]
    prefill_layers = max(0, min(int(args.prefill_layers), n_layers))

    # best-only 저장 준비
    save_best_dir_path = f"{HF_REPO_ID}/best" if HF_REPO_ID else args.save_best_dir
    best_dir = Path(save_best_dir_path); best_dir.mkdir(parents=True, exist_ok=True)
    min_val_loss = float("inf"); best_epoch = -1

    # ──────────────────────────────────────────────────────────────────
    # Special 시퀀스/프리필드/포워드 헬퍼
    # ──────────────────────────────────────────────────────────────────
    def parse_documents(document: str) -> List[str]:
        doc_separator = "\n\n----[DOC SEP]----\n\n"
        docs = document.split(doc_separator)
        docs = [doc.strip() for doc in docs if doc.strip()]
        return docs

    def _compute_uniform_K(total_doc_tokens: int, L: int) -> int:
        target_blocks = total_doc_tokens / max(1, args.compress_rate)
        K = math.ceil(target_blocks / max(1, L) - 1.0)
        return max(1, K)

    def build_special_seq_for_doc(document: str) -> str:
        total_tokens = len(tokenizer(document, add_special_tokens=False).input_ids)
        if only_one:
            L = args.latent_token_num
            K = _compute_uniform_K(total_tokens, L) if dynamic_sup else max(0, args.support_token_num)
            return support_token * (L * K)
        L = args.latent_token_num
        markers = build_latent_markers(L, use_docsep)
        if not use_support:
            return "".join(markers)
        K = _compute_uniform_K(total_tokens, L) if dynamic_sup else max(0, args.support_token_num)
        sup_block = support_token * K if K > 0 else ""
        return "".join([m + sup_block for m in markers])

    def _assistant_header_span_from_user_p1(user_p1: str) -> Tuple[int, int]:
        pre_msgs = [{"role":"system","content":system_prompt},{"role":"user","content":user_p1}]
        ids_pre = tokenizer(tokenizer.apply_chat_template(pre_msgs, tokenize=False), add_special_tokens=False, return_tensors="pt").input_ids[0]
        ass_empty = pre_msgs + [{"role":"assistant","content":""}]
        ids_ass_empty = tokenizer(tokenizer.apply_chat_template(ass_empty, tokenize=False), add_special_tokens=False, return_tensors="pt").input_ids[0]
        return len(ids_pre), len(ids_ass_empty)

    # Mistral 전용 Phase1 프리필 + 결합
    def _prefill_and_build_combined_mistral(question: str, document: str):
        # Phase1 (Mistral): sys, user( doc, [q?], specials, [q?] )
        # - include_query_bool=True  → specials 뒤에 q까지 포함 (온라인/훈련용)
        # - include_query_bool=False → q 미포함, specials까지만 포함 (오프라인 캐싱용)
        special_seq = build_special_seq_for_doc(document)
        if include_query_bool:
            user_p1 = f"Document:\n{document}\n\nQuestion: {question}\n{special_seq}Question: {question}"

            messages1 = [{"role": "system", "content": system_prompt},
                         {"role": "user",   "content": user_p1}]

            ids1 = tokenizer(tokenizer.apply_chat_template(messages1, tokenize=False),
                             add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

            ids1_list = ids1.tolist()
            pos1_special = [i for i, tid in enumerate(ids1_list) if tid in special_id_set]
            if not pos1_special:
                return None, None, None, None, None

            start_sp = pos1_special[0]
            start_slice = start_sp

            prefix_ids  = ids1[:start_slice]
            special_ids = ids1[start_slice:]

            with torch.no_grad():
                out_prefix = model(
                    prefix_ids.unsqueeze(0),
                    torch.ones(1, prefix_ids.size(0), device=ids1.device),
                    use_cache=True,
                    return_dict=True,
                )
            prefix_pkv = out_prefix.past_key_values

            attn_mask_sp = torch.cat([
                torch.ones(1, prefix_ids.size(0), device=ids1.device),
                torch.ones(1, special_ids.size(0), device=ids1.device),
            ], dim=1)
            out_sp = model(
                special_ids.unsqueeze(0),
                attn_mask_sp,
                past_key_values=prefix_pkv,
                use_cache=True,
                return_dict=True,
            )

            kv_sp_full = dc_from_legacy(out_sp.past_key_values)
            special_len = special_ids.size(0)

            combined = DynamicCache() if (USE_DYNAMIC_CACHE and isinstance(kv_sp_full, DynamicCache)) else []
            for li in range(n_layers):
                k_sys, v_sys = dc_get_kv(system_cache, li)
                k_all, v_all = dc_get_kv(kv_sp_full, li)
                k_sp = k_all[:, :, -special_len:, :]
                v_sp = v_all[:, :, -special_len:, :]
                dc_update(combined, torch.cat([k_sys, k_sp], dim=2), torch.cat([v_sys, v_sp], dim=2), li)

            gap = start_slice - L_sys
            if gap < 0: gap = 0
            past_len = dc_first_seq_len(combined, default_len=L_sys)
            pos_base = (past_len + gap) if args.pos_mode == "original" else past_len
            return combined, past_len, pos_base, special_seq, (question,)
        else:
            # include_query=False → per-document specials only; concatenate with <|DOC_SEP|>
            if doc_separator_print in document:
                doc_list = [d.strip() for d in document.split(doc_separator_print) if d.strip()]
            else:
                doc_list = [document]

            combined = DynamicCache() if (USE_DYNAMIC_CACHE and isinstance(system_cache, DynamicCache)) else []
            for li in range(n_layers):
                k_sys, v_sys = dc_get_kv(system_cache, li)
                dc_update(combined, k_sys, v_sys, li)

            gap_max = 0
            special_chunks = []
            for di, d in enumerate(doc_list):
                sp_seq = build_special_seq_for_doc(d)
                special_chunks.append(sp_seq)
                user_p1 = f"Document:\n{d}\n\n{sp_seq}"
                messages1 = [{"role": "system", "content": system_prompt},
                             {"role": "user",   "content": user_p1}]
                ids1 = tokenizer(tokenizer.apply_chat_template(messages1, tokenize=False),
                                 add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
                ids1_list = ids1.tolist()
                pos1_special = [i for i, tid in enumerate(ids1_list) if tid in special_id_set]
                if not pos1_special:
                    continue
                start_sp = pos1_special[0]
                gap_max = max(gap_max, max(0, start_sp - L_sys))

                prefix_ids = ids1[:start_sp]
                special_ids = ids1[start_sp:]

                with torch.no_grad():
                    out_prefix = model(
                        prefix_ids.unsqueeze(0),
                        torch.ones(1, prefix_ids.size(0), device=ids1.device),
                        use_cache=True,
                        return_dict=True,
                    )
                prefix_pkv = out_prefix.past_key_values

                attn_mask_sp = torch.cat([
                    torch.ones(1, prefix_ids.size(0), device=ids1.device),
                    torch.ones(1, special_ids.size(0), device=ids1.device),
                ], dim=1)
                out_sp = model(
                    special_ids.unsqueeze(0),
                    attn_mask_sp,
                    past_key_values=prefix_pkv,
                    use_cache=True,
                    return_dict=True,
                )
                kv_sp_full = dc_from_legacy(out_sp.past_key_values)
                special_len = special_ids.size(0)
                for li in range(n_layers):
                    k_comb, v_comb = dc_get_kv(combined, li)
                    k_all, v_all = dc_get_kv(kv_sp_full, li)
                    k_sp = k_all[:, :, -special_len:, :]
                    v_sp = v_all[:, :, -special_len:, :]
                    dc_update(combined, torch.cat([k_comb, k_sp], dim=2), torch.cat([v_comb, v_sp], dim=2), li)

                if di < len(doc_list) - 1:
                    try:
                        sep_id = tokenizer.convert_tokens_to_ids(doc_sep_token)
                        past_cur_len = dc_get_kv(combined, 0)[0].shape[2] if USE_DYNAMIC_CACHE else combined[0][0].shape[2]
                        attn_mask = torch.ones(1, past_cur_len + 1, device=ids1.device)
                        out_sep = model(
                            torch.tensor([[sep_id]], device=ids1.device),
                            attn_mask,
                            past_key_values=combined,
                            use_cache=True,
                            return_dict=True,
                        )
                        combined = dc_from_legacy(out_sep.past_key_values)
                    except Exception:
                        pass

                # Free intermediate tensors to control memory (sequential doc processing)
                try:
                    del ids1, ids1_list, prefix_ids, special_ids, out_prefix, prefix_pkv, attn_mask_sp, out_sp, kv_sp_full
                except Exception:
                    pass
                try:
                    del out_sep, attn_mask
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            special_seq = doc_sep_token.join(special_chunks)
            gap = gap_max
            past_len = dc_first_seq_len(combined, default_len=L_sys)
            pos_base = (past_len + gap) if args.pos_mode == "original" else past_len
            return combined, past_len, pos_base, special_seq, (question,)

    def _prefill_and_build_combined_llama_like(question: str, document: str):
        # 기존 로직 유지 + include_query=False일 때 문서별 specials 분할·병합
        if include_query_bool:
            special_seq = build_special_seq_for_doc(document)
            if header_mode == "assistant":
                user_p1 = f"Document:\n{document}\n\nQuestion: {question}"
                messages1 = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_p1},
                    {"role": "assistant", "content": special_seq},
                ]
                messages1.append({"role": "user", "content": f"Question: {question}"})
                header_start, header_end = _assistant_header_span_from_user_p1(user_p1)
            else:
                user_p1 = f"Document:\n{document}\n\nQuestion: {question}" + special_seq + f"Question: {question}"
                messages1 = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_p1}]
                header_start, header_end = None, None

            ids1 = tokenizer(tokenizer.apply_chat_template(messages1, tokenize=False),
                             add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
            ids1_list = ids1.tolist()
            pos1_special = [i for i, tid in enumerate(ids1_list) if tid in special_id_set]
            if not pos1_special:
                return None, None, None, None, None
            start_sp = pos1_special[0]

            start_slice = header_start if header_mode == "assistant" else start_sp
            prefix_ids = ids1[:start_slice]
            special_ids = ids1[start_slice:]

            # Phase1: 문서 포함 경로(하위 prefill_layers까지만)
            if prefill_layers > 0 and prefill_layers < n_layers:
                # 하위 레이어까지의 prefix는 그래프 비연결(no_grad), special 구간은 grad on
                inner_base = None
                try:
                    inner_base = model.get_base_model() if lora_enabled and hasattr(model, "get_base_model") else model
                except Exception:
                    inner_base = model
                with torch.no_grad():
                    out_prefix = inner_base.model(
                        input_ids=prefix_ids.unsqueeze(0),
                        attention_mask=torch.ones(1, prefix_ids.size(0), device=ids1.device),
                        use_cache=True,
                        return_dict=True,
                        layers_limit=prefill_layers,
                    )
                prefix_pkv = dc_from_legacy(out_prefix.past_key_values)
                attn_mask_sp = torch.cat([
                    torch.ones(1, prefix_ids.size(0), device=ids1.device),
                    torch.ones(1, special_ids.size(0), device=ids1.device),
                ], dim=1)
                out_sp_low = inner_base.model(
                    input_ids=special_ids.unsqueeze(0),
                    attention_mask=attn_mask_sp,
                    past_key_values=prefix_pkv,
                    use_cache=True,
                    return_dict=True,
                    layers_limit=prefill_layers,
                )
                kv_sp_low = dc_from_legacy(out_sp_low.past_key_values)
            else:
                # 전체 레이어 경로: prefix만 no_grad, special은 grad on
                with torch.no_grad():
                    out_prefix = model(
                        prefix_ids.unsqueeze(0),
                        torch.ones(1, prefix_ids.size(0), device=ids1.device),
                        use_cache=True,
                        return_dict=True,
                    )
                prefix_pkv = dc_from_legacy(out_prefix.past_key_values)
                attn_mask_sp = torch.cat([
                    torch.ones(1, prefix_ids.size(0), device=ids1.device),
                    torch.ones(1, special_ids.size(0), device=ids1.device),
                ], dim=1)
                out_sp_full = model(
                    special_ids.unsqueeze(0),
                    attn_mask_sp,
                    past_key_values=prefix_pkv,
                    use_cache=True,
                    return_dict=True,
                )
                kv_sp_low = dc_from_legacy(out_sp_full.past_key_values)

            special_len = special_ids.size(0)

            if prefill_layers > 0 and prefill_layers < n_layers:
                # Phase2: 레이어 prefill_layers 이후를 "이어서 계산" (문서 없이)
                hidden_tail_low = out_sp_low.last_hidden_state  # [1, special_len, H]
                pos_ids_tail = torch.arange(start_slice, start_slice + special_len, device=ids1.device, dtype=torch.long).unsqueeze(0)
                # For FA2 varlen path, provide a 2D key-padding mask to avoid
                # _prepare_from_posids assuming zero-based positions.
                attn_mask_tail = torch.ones(1, special_len, device=ids1.device, dtype=torch.long)
                try:
                    inner_base = model.get_base_model() if lora_enabled and hasattr(model, "get_base_model") else model
                except Exception:
                    inner_base = model
                out_tail_upper = inner_base.model(
                    inputs_embeds=hidden_tail_low,  # layer prefill_layers 출력에서 이어서
                    attention_mask=attn_mask_tail,
                    position_ids=pos_ids_tail,
                    use_cache=True,
                    return_dict=True,
                    start_layer=prefill_layers,
                )
                kv_tail_upper = dc_from_legacy(out_tail_upper.past_key_values)

                # 혼합 캐시 생성: 하위 레이어는 doc경로, 상위 레이어는 tail-upper 경로
                combined = DynamicCache() if (USE_DYNAMIC_CACHE and isinstance(system_cache, DynamicCache)) else []
                for li in range(n_layers):
                    k_sys, v_sys = dc_get_kv(system_cache, li)
                    if li < prefill_layers:
                        k_all_low, v_all_low = dc_get_kv(kv_sp_low, li)
                        k_sp_low = k_all_low[:, :, -special_len:, :]
                        v_sp_low = v_all_low[:, :, -special_len:, :]
                        k_cat = torch.cat([k_sys, k_sp_low], dim=2)
                        v_cat = torch.cat([v_sys, v_sp_low], dim=2)
                    else:
                        k_all_up, v_all_up = dc_get_kv(kv_tail_upper, li)
                        k_sp_up = k_all_up[:, :, -special_len:, :]
                        v_sp_up = v_all_up[:, :, -special_len:, :]
                        k_cat = torch.cat([k_sys, k_sp_up], dim=2)
                        v_cat = torch.cat([v_sys, v_sp_up], dim=2)
                    dc_update(combined, k_cat, v_cat, li)
            else:
                # 기존 전체 레이어 프리필 결합 (부분 레이어 비활성/실패 시)
                combined = DynamicCache() if (USE_DYNAMIC_CACHE and isinstance(system_cache, DynamicCache)) else []
                for li in range(n_layers):
                    k_sys, v_sys = dc_get_kv(system_cache, li)
                    k_all, v_all = dc_get_kv(kv_sp_low, li)
                    if k_all is None or v_all is None:
                        dc_update(combined, k_sys, v_sys, li)
                    else:
                        k_sp = k_all[:, :, -special_len:, :]
                        v_sp = v_all[:, :, -special_len:, :]
                        dc_update(combined, torch.cat([k_sys, k_sp], dim=2), torch.cat([v_sys, v_sp], dim=2), li)

            gap = start_slice - L_sys
            if gap < 0: gap = 0
            # Safe past length: first valid layer length, fallback to L_sys + special_len
            past_len = dc_first_seq_len(combined, default_len=(L_sys + special_len))
            pos_base = (past_len + gap) if args.pos_mode == "original" else past_len
            return combined, past_len, pos_base, special_seq, (question,)
        else:
            # include_query=False → 문서별 specials만 생성해 이어붙임
            if doc_separator_print in document:
                doc_list = [d.strip() for d in document.split(doc_separator_print) if d.strip()]
            else:
                doc_list = [document]

            combined = DynamicCache() if (USE_DYNAMIC_CACHE and isinstance(system_cache, DynamicCache)) else []
            for li in range(n_layers):
                k_sys, v_sys = dc_get_kv(system_cache, li)
                dc_update(combined, k_sys, v_sys, li)

            gap_max = 0
            special_chunks = []
            for di, d in enumerate(doc_list):
                sp_seq = build_special_seq_for_doc(d)
                special_chunks.append(sp_seq)
                if header_mode == "assistant":
                    user_p1 = f"Document:\n{d}\n\n"
                    messages1 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_p1},
                        {"role": "assistant", "content": sp_seq},
                    ]
                    header_start, header_end = _assistant_header_span_from_user_p1(user_p1)
                    ids1 = tokenizer(tokenizer.apply_chat_template(messages1, tokenize=False),
                                     add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
                    start_slice = header_start
                else:
                    user_p1 = f"Document:\n{d}\n\n" + sp_seq
                    messages1 = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_p1}]
                    ids1 = tokenizer(tokenizer.apply_chat_template(messages1, tokenize=False),
                                     add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
                    ids1_list = ids1.tolist()
                    pos1_special = [i for i, tid in enumerate(ids1_list) if tid in special_id_set]
                    if not pos1_special:
                        continue
                    start_slice = pos1_special[0]

                gap_max = max(gap_max, max(0, start_slice - L_sys))

                prefix_ids = ids1[:start_slice]
                special_ids = ids1[start_slice:]

                with torch.no_grad():
                    out_prefix = model(
                        prefix_ids.unsqueeze(0),
                        torch.ones(1, prefix_ids.size(0), device=ids1.device),
                        use_cache=True,
                        return_dict=True,
                    )
                prefix_pkv = out_prefix.past_key_values

                attn_mask_sp = torch.cat([
                    torch.ones(1, prefix_ids.size(0), device=ids1.device),
                    torch.ones(1, special_ids.size(0), device=ids1.device),
                ], dim=1)
                out_sp = model(
                    special_ids.unsqueeze(0),
                    attn_mask_sp,
                    past_key_values=prefix_pkv,
                    use_cache=True,
                    return_dict=True,
                )
                kv_sp_full = dc_from_legacy(out_sp.past_key_values)
                special_len = special_ids.size(0)
                for li in range(n_layers):
                    k_comb, v_comb = dc_get_kv(combined, li)
                    k_all, v_all = dc_get_kv(kv_sp_full, li)
                    k_sp = k_all[:, :, -special_len:, :]
                    v_sp = v_all[:, :, -special_len:, :]
                    dc_update(combined, torch.cat([k_comb, k_sp], dim=2), torch.cat([v_comb, v_sp], dim=2), li)

                # Always insert doc-separator token between chunks (except after last)
                if di < len(doc_list) - 1:
                    try:
                        sep_id = tokenizer.convert_tokens_to_ids(doc_sep_token)
                        past_cur_len = dc_get_kv(combined, 0)[0].shape[2] if USE_DYNAMIC_CACHE else combined[0][0].shape[2]
                        attn_mask = torch.ones(1, past_cur_len + 1, device=ids1.device)
                        out_sep = model(
                            torch.tensor([[sep_id]], device=ids1.device),
                            attn_mask,
                            past_key_values=combined,
                            use_cache=True,
                            return_dict=True,
                        )
                        combined = dc_from_legacy(out_sep.past_key_values)
                    except Exception:
                        pass

                # Free intermediate tensors to control memory (sequential doc processing)
                try:
                    del ids1, ids1_list, prefix_ids, special_ids, out_prefix, prefix_pkv, attn_mask_sp, out_sp, kv_sp_full
                except Exception:
                    pass
                try:
                    del out_sep, attn_mask
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            special_seq = doc_sep_token.join(special_chunks)
            gap = gap_max
            past_len = dc_get_kv(combined, 0)[0].shape[2] if USE_DYNAMIC_CACHE else combined[0][0].shape[2]
            pos_base = (past_len + gap) if args.pos_mode == "original" else past_len
            return combined, past_len, pos_base, special_seq, (question,)

    def _prefill_and_build_combined(question: str, document: str):
        if IS_MISTRAL:
            return _prefill_and_build_combined_mistral(question, document)
        return _prefill_and_build_combined_llama_like(question, document)

    # Phase2 prefix 생성
    def _build_phase2_prefix_mistral(question: str, special_seq: str):
        # Phase2: sys, user(specials, q), assistant("")
        prefix_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": special_seq + f"Question: {question}"}
        ]
        prefix_ids = tokenizer(tokenizer.apply_chat_template(prefix_msgs, tokenize=False),
                               add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        return prefix_ids, prefix_msgs

    def _build_phase2_prefix_llama_like(question: str, special_seq: str):
        if header_mode == "assistant":
            prefix_msgs = [
                {"role": "system",    "content": system_prompt},
                {"role": "assistant", "content": special_seq},
                {"role": "user",      "content": f"Question: {question}"},
                {"role": "assistant", "content": ""}  # 빈 답변 헤더
            ]
        else:
            prefix_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": special_seq + f"Question: {question}"}
            ]
        prefix_ids = tokenizer(tokenizer.apply_chat_template(prefix_msgs, tokenize=False),
                               add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        return prefix_ids, prefix_msgs

    def _build_phase2_prefix(question: str, special_seq: str):
        if IS_MISTRAL:
            return _build_phase2_prefix_mistral(question, special_seq)
        return _build_phase2_prefix_llama_like(question, special_seq)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────
    def forward_two_stage_original_batched_responses(question: str, document: str, responses: List[str]) -> torch.Tensor:
        resp_list = [str(r) for r in (responses or []) if isinstance(r, str) and len(str(r).strip()) > 0]
        if len(resp_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        combined, past_len, pos_base, special_seq, _ = _prefill_and_build_combined(question, document)
        if combined is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        prefix_ids, prefix_msgs = _build_phase2_prefix(question, special_seq)
        prefix_len = prefix_ids.size(0)

        ass_ids_list = []
        for resp in resp_list:
            # alternation 규칙 보호
            if not IS_MISTRAL and header_mode == "assistant":
                base_msgs = list(prefix_msgs[:-1])  # 마지막 빈 assistant 제거
            else:
                base_msgs = list(prefix_msgs)       # Mistral 또는 user 헤더는 그대로

            msgs2 = base_msgs + [{"role": "assistant", "content": resp}]
            full_ids2 = tokenizer(
                tokenizer.apply_chat_template(msgs2, tokenize=False),
                add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)

            assistant_ids = full_ids2[prefix_len:]
            if assistant_ids.numel() > 0:
                ass_ids_list.append(assistant_ids)

        if len(ass_ids_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        B = len(ass_ids_list)
        L_ass_max = max(t.size(0) for t in ass_ids_list)
        input_ids = torch.full((B, L_ass_max), tokenizer.pad_token_id, device=device, dtype=torch.long)
        labels    = torch.full((B, L_ass_max), -100, device=device, dtype=torch.long)
        pos_ids   = torch.zeros((B, L_ass_max), device=device, dtype=torch.long)

        for bi, a_ids in enumerate(ass_ids_list):
            La = a_ids.size(0)
            input_ids[bi, :La] = a_ids
            labels[bi,    :La] = a_ids
            pos_ids[bi,   :La] = torch.arange(pos_base, pos_base + La, device=device, dtype=torch.long)

        # Try batched path; if cache contains None layers, fallback to sequential for stability
        try:
            batch_cache = dc_repeat(combined, B)
            attn_mask = torch.cat([torch.ones(B, past_len, device=device, dtype=torch.long),
                                   (input_ids != tokenizer.pad_token_id).long()], dim=1)

            out = model(
                input_ids=input_ids,
                past_key_values=batch_cache,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                labels=labels,
                return_dict=True,
            )
            return out.loss
        except Exception:
            # Fallback: compute per response and average (keep graph)
            losses: List[torch.Tensor] = []
            valid = 0
            for a_ids in ass_ids_list:
                La = a_ids.size(0)
                attn_mask = torch.cat([
                    torch.ones(1, past_len, device=device, dtype=torch.long),
                    torch.ones(1, La,       device=device, dtype=torch.long)
                ], dim=1)
                pos_single = torch.arange(pos_base, pos_base + La, device=device, dtype=torch.long).unsqueeze(0)
                out = model(
                    input_ids=a_ids.unsqueeze(0),
                    past_key_values=combined,
                    attention_mask=attn_mask,
                    position_ids=pos_single,
                    labels=a_ids.unsqueeze(0),
                    return_dict=True,
                )
                if torch.isfinite(out.loss):
                    losses.append(out.loss)
                    valid += 1
            if valid == 0:
                return (attn_mask.sum() * 0.0)
            return sum(losses) / valid

    def forward_two_stage_original_single_response(question: str, document: str, response_str: str) -> torch.Tensor:
        resp = (response_str or "").strip()
        if len(resp) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        combined, past_len, pos_base, special_seq, _ = _prefill_and_build_combined(question, document)
        if combined is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        prefix_ids, prefix_msgs = _build_phase2_prefix(question, special_seq)
        prefix_len = prefix_ids.size(0)

        if not IS_MISTRAL and header_mode == "assistant":
            base_msgs = list(prefix_msgs[:-1])
        else:
            base_msgs = list(prefix_msgs)

        msgs2 = base_msgs + [{"role":"assistant","content": resp}]

        full_ids2 = tokenizer(
            tokenizer.apply_chat_template(msgs2, tokenize=False),
            add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(device)

        assistant_ids = full_ids2[prefix_len:]
        if assistant_ids.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        B = 1
        La = assistant_ids.size(0)
        input_ids = assistant_ids.unsqueeze(0)
        labels    = assistant_ids.unsqueeze(0).clone()
        pos_ids   = torch.arange(pos_base, pos_base + La, device=device, dtype=torch.long).unsqueeze(0)

        batch_cache = dc_repeat(combined, B)
        attn_mask = torch.cat([torch.ones(B, past_len, device=device, dtype=torch.long),
                               torch.ones(B, La,       device=device, dtype=torch.long)], dim=1)

        out = model(
            input_ids=input_ids,
            past_key_values=batch_cache,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            labels=labels,
            return_dict=True,
        )
        return out.loss

    # 10) Training loop with validation (+ LR scheduler)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_example_count = 0
        train_nan_skipped = 0
        train_badgrad_skipped = 0

        for qs, ds, rlists, rstrs in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            for i in range(len(qs)):
                if use_batch:
                    loss = forward_two_stage_original_batched_responses(qs[i], ds[i], rlists[i])
                else:
                    loss = forward_two_stage_original_single_response(qs[i], ds[i], rstrs[i])

                if torch.isnan(loss) or torch.isinf(loss):
                    train_nan_skipped += 1
                    wandb.log({"train/nan_or_inf_loss_skipped": 1})
                    continue

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                # Extra safety: sanitize embedding grads to avoid non-finite checks failing
                if inp_emb.weight.grad is not None:
                    inp_emb.weight.grad.data = torch.nan_to_num(
                        inp_emb.weight.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                    )

                grad_ok = (inp_emb.weight.grad is not None) and torch.isfinite(inp_emb.weight.grad).all()
                if not grad_ok:
                    train_badgrad_skipped += 1
                    wandb.log({"train/nonfinite_grad_skipped": 1})
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if args.grad_clip and args.grad_clip > 0:
                    accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

                if scheduler is not None and sched_mode == "step":
                    scheduler.step()

                global_step += 1
                train_example_count += 1
                train_loss_sum += loss.item()

                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"step/train_loss": loss.item(), "lr": current_lr, "step": global_step})

            torch.cuda.empty_cache()
            gc.collect()

        train_avg = train_loss_sum / max(1, train_example_count)
        wandb.log({
            "epoch": epoch,
            "train/avg_loss": train_avg,
            "train/nan_or_inf_loss_skipped_total": train_nan_skipped,
            "train/nonfinite_grad_skipped_total": train_badgrad_skipped
        })

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        val_example_count = 0
        val_nan_skipped = 0

        with torch.no_grad():
            for qs, ds, rlists, rstrs in tqdm(val_loader, desc=f"Epoch {epoch} [valid]"):
                for i in range(len(qs)):
                    if use_batch:
                        loss = forward_two_stage_original_batched_responses(qs[i], ds[i], rlists[i])
                    else:
                        loss = forward_two_stage_original_single_response(qs[i], ds[i], rstrs[i])

                    if torch.isnan(loss) or torch.isinf(loss):
                        val_nan_skipped += 1
                        wandb.log({"valid/nan_or_inf_skipped": 1})
                        try:
                            doc_len = len(tokenizer(ds[i], add_special_tokens=False).input_ids)
                            with open("_nan_val_samples.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps({"epoch": epoch, "idx_hint": i, "doc_len": int(doc_len)}, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        continue

                    val_loss_sum += loss.item()
                    val_example_count += 1

        val_avg = float("inf") if val_example_count == 0 else (val_loss_sum / val_example_count)

        wandb.log({
            "epoch": epoch,
            "valid/avg_loss": val_avg if val_avg != float("inf") else None,
            "valid/nan_or_inf_skipped_total": val_nan_skipped
        })
        human_val = f"{val_avg:.4f}" if val_avg != float("inf") else "inf"

        print(f"[Epoch {epoch}] train_avg={train_avg:.4f} "
              f"| train_skipped(nan_loss={train_nan_skipped}, bad_grad={train_badgrad_skipped}) "
              f"|| valid_avg={human_val} | valid_skipped={val_nan_skipped}")

        if scheduler is not None and sched_mode == "epoch":
            scheduler.step(val_avg)

        if accelerator.is_main_process and val_avg < min_val_loss:
            min_val_loss = val_avg
            best_epoch = epoch
            for p in best_dir.glob("*"):
                if p.is_file(): p.unlink()
                elif p.is_dir(): shutil.rmtree(p)
            unwrapped = accelerator.unwrap_model(model)
            # Package in Hub-ready layout:
            #   best/
            #     config.json (remote-code auto_map)
            #     generation_config.json
            #     chat_template.jinja (if available)
            #     tokenizer.json, tokenizer_config.json, special_tokens_map.json
            #     modeling_partial_layer.py, partial_xgenerate.py
            #     base/ (vanilla weights)
            #     lora/ (adapter, if used)
            base_dir = best_dir / "base"
            base_dir.mkdir(parents=True, exist_ok=True)

            # 1) Save vanilla base weights under base/
            # LoRA 사용 시 base 모델 구조에 LoRA 래퍼가 남아 있을 수 있으므로,
            # 깨끗한 베이스를 로드한 뒤(원본 backbone), 학습된 special-token 임베딩만 복사하여 저장합니다.
            try:
                from transformers import AutoModelForCausalLM
            except Exception:
                AutoModelForCausalLM = None

            base_saved = False
            try:
                if lora_enabled and AutoModelForCausalLM is not None:
                    print("[Best] LoRA enabled → loading clean base and transplanting special-token embeddings …")
                    load_kwargs = {}
                    if args.cache_dir_model:
                        load_kwargs["cache_dir"] = args.cache_dir_model
                    load_kwargs["trust_remote_code"] = True
                    clean_base = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
                    # Ensure vocab matches tokenizer (to host added Latent specials)
                    try:
                        clean_base.resize_token_embeddings(len(tokenizer))
                    except Exception:
                        pass
                    # Copy only the rows for the additional_special_tokens from the trained model
                    try:
                        trained_base = unwrapped.get_base_model() if hasattr(unwrapped, "get_base_model") else unwrapped
                    except Exception:
                        trained_base = unwrapped
                    emb_src = trained_base.get_input_embeddings().weight.data
                    emb_dst = clean_base.get_input_embeddings().weight.data
                    # Build special id set from tokenizer
                    special_tokens_vocab = getattr(tokenizer, "additional_special_tokens", []) or []
                    special_id_set = set(tokenizer.convert_tokens_to_ids(t) for t in special_tokens_vocab)
                    H_src, H_dst = emb_src.shape[0], emb_dst.shape[0]
                    copied = 0
                    for sid in special_id_set:
                        if sid is None or sid < 0: continue
                        if sid < H_src and sid < H_dst:
                            emb_dst[sid].copy_(emb_src[sid])
                            copied += 1
                    print(f"[Best] Transplanted {copied} special-token embedding rows")
                    # Avoid leaking remote-code auto_map into base config
                    try:
                        setattr(clean_base.config, "auto_map", None)
                    except Exception:
                        pass
                    clean_base.save_pretrained(base_dir, safe_serialization=True)
                    try:
                        del clean_base
                    except Exception:
                        pass
                    base_saved = True
                else:
                    # No LoRA or AutoModel unavailable → save current base as-is (already includes resized embeddings)
                    base_to_save = unwrapped.get_base_model() if hasattr(unwrapped, "get_base_model") else unwrapped
                    try:
                        setattr(base_to_save.config, "auto_map", None)
                    except Exception:
                        pass
                    base_to_save.save_pretrained(base_dir, safe_serialization=True)
                    base_saved = True
            except Exception as e:
                print(f"[Warn] Clean base save/transplant failed ({e}); falling back to current model base")
                try:
                    base_to_save = unwrapped.get_base_model() if hasattr(unwrapped, "get_base_model") else unwrapped
                    try:
                        setattr(base_to_save.config, "auto_map", None)
                    except Exception:
                        pass
                    base_to_save.save_pretrained(base_dir, safe_serialization=True)
                    base_saved = True
                except Exception as ee:
                    raise RuntimeError(f"Failed to save base under base/: {ee}")

            if base_saved:
                print(f"[Best] Saved base weights to {base_dir}")

            # 2) Save LoRA adapter under lora/ (for reattachment/merge at runtime)
            if lora_enabled:
                try:
                    (best_dir / "lora").mkdir(parents=True, exist_ok=True)
                    unwrapped.save_pretrained(best_dir / "lora")
                    print("[Best] Saved LoRA adapter under ./lora")
                except Exception as e:
                    print(f"[Warn] Failed to save LoRA adapter: {e}")

            # 3) Tokenizer at root (for chat template + specials)
            try:
                tokenizer.save_pretrained(best_dir)
                print(f"[Best] Saved tokenizer to {best_dir}")
            except Exception as e:
                print(f"[Warn] Failed to save tokenizer: {e}")

            # 4) generation_config.json at root
            try:
                gen_cfg_path = best_dir / "generation_config.json"
                if hasattr(base_to_save, "generation_config") and base_to_save.generation_config is not None:
                    base_to_save.generation_config.to_json_file(str(gen_cfg_path))
                elif (base_dir / "generation_config.json").is_file():
                    shutil.copy2(base_dir / "generation_config.json", gen_cfg_path)
            except Exception:
                pass

            # 5) chat_template.jinja at root (from tokenizer.chat_template or base_dir)
            try:
                tmpl = getattr(tokenizer, "chat_template", None)
                if isinstance(tmpl, str) and tmpl.strip():
                    (best_dir / "chat_template.jinja").write_text(tmpl, encoding="utf-8")
                elif (base_dir / "chat_template.jinja").is_file():
                    shutil.copy2(base_dir / "chat_template.jinja", best_dir / "chat_template.jinja")
            except Exception:
                pass

            # 6) Remote code: copy modeling + pxgenerate helpers
            try:
                # Use repository-stable locations only (no dependency on jeongseokoh2 folder)
                if IS_MISTRAL:
                    candidates_modeling = [Path(__file__).parent / "modeling_mistral_partial.py"]
                    dst_modeling = best_dir / "modeling_mistral_partial.py"
                else:
                    candidates_modeling = [Path(__file__).parent / "modeling_partial_layer.py"]
                    dst_modeling = best_dir / "modeling_partial_layer.py"
                candidates_px = [
                    Path(__file__).parent / "evaluate" / "utils" / "partial_xgenerate.py",
                ]
                dst_px = best_dir / "partial_xgenerate.py"
                src_m = next((p for p in candidates_modeling if p.is_file()), None)
                src_p = next((p for p in candidates_px if p.is_file()), None)
                if src_m is not None:
                    shutil.copy2(src_m, dst_modeling)
                if src_p is not None:
                    shutil.copy2(src_p, dst_px)
                print(f"[Best] Injected remote-code helpers ({dst_modeling.name}, partial_xgenerate.py)")
            except Exception as e:
                print(f"[Warn] Failed to inject remote-code helpers: {e}")

            # 7) Root config.json with remote-code auto_map
            try:
                # Start from base config
                cfg_dict = {}
                try:
                    cfg_dict = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
                except Exception:
                    try:
                        from transformers import PretrainedConfig
                        cfg_dict, _ = PretrainedConfig.get_config_dict(args.model_name)
                    except Exception:
                        cfg_dict = getattr(base_to_save, "config", object()).to_dict() if hasattr(base_to_save, "config") else {}
                # Preserve existing model_type from base if present; otherwise default by family
                if "model_type" not in cfg_dict or not cfg_dict["model_type"]:
                    cfg_dict["model_type"] = "mistral" if IS_MISTRAL else "llama"
                auto_map = cfg_dict.get("auto_map", {}) or {}
                auto_map_key = "AutoModelForCausalLM"
                auto_map_val = (
                    "modeling_mistral_partial.MistralForCausalLM" if IS_MISTRAL
                    else "modeling_partial_layer.LlamaForCausalLM"
                )
                auto_map[auto_map_key] = auto_map_val
                cfg_dict["auto_map"] = auto_map
                # Prefer bfloat16
                try:
                    cfg_dict["torch_dtype"] = "bfloat16"
                except Exception:
                    pass
                (best_dir / "config.json").write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[Best] Wrote remote-code root config.json with auto_map → {auto_map_val}")
            except Exception as e:
                print(f"[Warn] Failed to write remote-code config.json: {e}")

            print(f"[Best] epoch {epoch} (val={human_val}) → {best_dir} packaged (Hub-ready)")

    accelerator.wait_for_everyone()

    # 11) (Best-only) Push to Hugging Face Hub
    if accelerator.is_main_process and args.push_to_hub and args.hf_repo_id:
        if best_epoch < 0:
            raise RuntimeError("best_epoch가 기록되지 않았습니다. 유효한 validation이 있었는지 확인하세요.")
        # 보안을 위해 토큰은 환경변수로 읽습니다.
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
        if not HF_TOKEN:
            raise ValueError("환경변수 HF_TOKEN 이 설정되어 있지 않습니다. export HF_TOKEN=...")

        # Root already packaged with remote-code; just push

        repo_id = HF_REPO_ID or args.hf_repo_id
        create_repo(repo_id, exist_ok=True, private=args.private_repo, token=HF_TOKEN)
        upload_folder(
            repo_id=repo_id, folder_path=str(best_dir), token=HF_TOKEN,
            commit_message=f"Best epoch {best_epoch} (val_loss={min_val_loss:.4f})", allow_patterns=["*"],
        )
        print(f"✅ Hub 업로드 완료: {repo_id} (best epoch={best_epoch}, val={min_val_loss:.4f})")

if __name__ == "__main__":
    main()

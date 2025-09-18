#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py — TRI(balanced) 검증 도구
- quick_probe: trainer loss vs manual NLL 대조
- corrupt_and_loss: 10% 라벨 교란 시 손실 증가 확인

전제:
- lopa_llama_modeling.py 가 TRI(use_cache 준수)로 패치되어 있어야 함.
- 데이터셋은 {"question","document","responses"/"response"} 필드를 갖는 jsonl.

예시:
python test.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_file /path/to/data.jsonl \
  --lopa_modeling_path ./lopa_llama_modeling.py \
  --prefill_layers 16 \
  --samples 16 \
  --corruption 0.10
"""

from __future__ import annotations
import argparse
import json
import os
import random
from copy import copy
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------
# Modeling patch loader (TRI)
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

    need_LM = ["LlamaModel", "LlamaForCausalLM"]
    for n in need_LM:
        if not hasattr(module, n):
            raise RuntimeError(f"Patched module missing `{n}`")

    # TRI API 확인
    LM, TOP = module.LlamaModel, module.LlamaForCausalLM
    for n in ["tri_prefill_system_all", "tri_prefill_user_lower", "tri_build_caches", "tri_forward_assistant"]:
        if not hasattr(LM, n):
            raise RuntimeError(f"Patched LlamaModel lacks `{n}`")
    for n in ["tri_build_caches", "tri_forward_assistant", "tri_step_logits"]:
        if not hasattr(TOP, n):
            raise RuntimeError(f"Patched LlamaForCausalLM lacks `{n}`")

    print("[DEBUG] TRI patch loaded:", modeling_path)
    return module


# -----------------------
# Dataset
# -----------------------
class QADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_doc_tokens: int = 4096):
        self.recs = []
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

                # responses or single response
                rs = []
                rr = rec.get("responses")
                if isinstance(rr, list):
                    rs = [s.strip() for s in rr if isinstance(s, str) and s.strip()]
                if not rs:
                    r_single = rec.get("response")
                    if isinstance(r_single, str) and r_single.strip():
                        rs = [r_single.strip()]
                if not rs:
                    continue

                # 길이 제한
                try:
                    n_tok = len(tokenizer(d, add_special_tokens=False).input_ids)
                except Exception:
                    n_tok = 0
                if n_tok > max_doc_tokens:
                    continue

                self.recs.append((q, d, rs))

    def __len__(self): return len(self.recs)
    def __getitem__(self, idx): return self.recs[idx]


# -----------------------
# Chat template helpers
# -----------------------
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

def lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
    L = min(a.size(1), b.size(1))
    eq = (a[0, :L] == b[0, :L])
    nz = (~eq).nonzero(as_tuple=False)
    return int(nz[0, 0]) if nz.numel() else L

def _assistant_content_delta(tokenizer, msgs, resp: str, SU_gen_ids, device):
    msgs_ass = msgs + [{"role": "assistant", "content": resp}]
    full_ids = tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
    return full_ids[:, SU_gen_ids.size(1):]


# -----------------------
# Cache helpers
# -----------------------
def _safe_len_seqlen(x):
    if x is None: return 0
    if hasattr(x, "shape") and x.dim() >= 3:
        return int(x.shape[2])
    return 0

def _layer_past_len(pkv, li: int) -> int:
    if pkv is None: return 0
    if hasattr(pkv, "key_cache") and isinstance(pkv.key_cache, (list, tuple)) and li < len(pkv.key_cache):
        return _safe_len_seqlen(pkv.key_cache[li])
    if hasattr(pkv, "layers") and isinstance(pkv.layers, (list, tuple)) and li < len(pkv.layers):
        lyr = pkv.layers[li]
        k = None if lyr is None else getattr(lyr, "keys", None)
        return _safe_len_seqlen(k)
    try:
        kv = pkv[li]
        if isinstance(kv, (list, tuple)) and len(kv) >= 1:
            return _safe_len_seqlen(kv[0])
    except Exception:
        pass
    return 0

def _fork_cache_view(pkv):
    new = copy(pkv)
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        new.key_cache  = list(pkv.key_cache)
        new.value_cache = list(pkv.value_cache)
    elif hasattr(pkv, "layers"):
        new.layers = list(getattr(pkv, "layers") or [])
    return new

def debug_check_layout(pkv, S, U, lower_k, tag=""):
    try:
        L = len(pkv.key_cache) if hasattr(pkv, "key_cache") else len(pkv.layers)
        lens = [ _layer_past_len(pkv, li) for li in range(L) ]
        print(f"[DEBUG:{tag}] per-layer past: {lens} (lower_k={lower_k}, S={S}, U={U})")
    except Exception as e:
        print("[DEBUG layout] skip:", e)


# -----------------------
# Probes
# -----------------------
def quick_probe(model, tokenizer, dataset, K: int, samples: int, device, system_prompt="You are a helpful assistant."):
    model.eval()
    N = min(samples, len(dataset))
    idxs = random.sample(range(len(dataset)), N)

    trainer_like_losses, manual_nlls, masked_rates, lengths = [], [], [], []

    for i in idxs:
        q, d, rs = dataset[i]
        resp = next((r for r in rs if isinstance(r, str) and r.strip()), None)
        if not resp: 
            continue

        # --- segments ---
        msgs = build_messages(system_prompt, d, q, include_query=True)
        S_ids  = tokens_from_messages(tokenizer, msgs[:1], device, add_generation_prompt=False)
        SU_ids = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
        SU_gen = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)

        l_su = lcp_len(S_ids, SU_ids)
        user_delta = SU_ids[:, l_su:SU_ids.size(1)]
        header_delta = SU_gen[:, SU_ids.size(1):]

        # --- caches ---
        with torch.no_grad():
            pkv_su, S_len, U_len = model.tri_build_caches(S_ids, user_delta, lower_k=K)
            pkv = _fork_cache_view(pkv_su)
            # header: no-grad write
            _ = model.tri_step_logits(header_delta, K, pkv, S_len, U_len, logits_to_keep=0, labels=None, write_cache=True)

        # --- assistant content ---
        ad = _assistant_content_delta(tokenizer, msgs, resp, SU_gen, device)
        if ad.numel() == 0:
            continue

        # trainer-like loss (HF 내부 loss_function)
        out = model.tri_step_logits(ad, K, pkv, S_len, U_len,
                                    logits_to_keep=ad.size(1), labels=ad.clone(), write_cache=False)
        trainer_like = float(out.loss.detach().item())
        trainer_like_losses.append(trainer_like)

        # manual NLL (shift)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ad[..., 1:].contiguous()
        manual = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1),
                                 ignore_index=-100, reduction="mean")
        manual_nlls.append(float(manual.detach().item()))

        # mask rate / len
        lengths.append(int(ad.size(1)))
        masked_rates.append(float((ad == -100).float().mean().item()))

    def _mean(x): return sum(x)/max(1,len(x))

    print("\n[quick_probe]")
    print("  trainer_like_loss_avg :", _mean(trainer_like_losses))
    print("  manual_NLL_avg        :", _mean(manual_nlls))
    print("  avg_len_tokens        :", _mean(lengths))
    print("  masked_rate_avg       :", _mean(masked_rates))
    print("  n_samples_used        :", len(trainer_like_losses))

    # 간단 비교 경고
    if abs(_mean(trainer_like_losses) - _mean(manual_nlls)) > 0.02:
        print("  [WARN] trainer-like vs manual NLL 차이가 큽니다. 라벨 시프트/마스킹/경계 재확인 요망.")
    if _mean(masked_rates) > 0.0:
        print("  [WARN] assistant_delta에 -100 마스크가 포함되어 있습니다(balanced에서는 보통 0이어야 함).")


def corrupt_and_loss(model, tokenizer, dataset, K: int, samples: int, corruption: float, device, system_prompt="You are a helpful assistant."):
    model.eval()
    N = min(samples, len(dataset))
    idxs = random.sample(range(len(dataset)), N)

    diffs = []
    for i in idxs:
        q, d, rs = dataset[i]
        resp = next((r for r in rs if isinstance(r, str) and r.strip()), None)
        if not resp:
            continue

        msgs = build_messages(system_prompt, d, q, include_query=True)
        S_ids  = tokens_from_messages(tokenizer, msgs[:1], device, add_generation_prompt=False)
        SU_ids = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
        SU_gen = tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)

        l_su = lcp_len(S_ids, SU_ids)
        user_delta   = SU_ids[:, l_su:SU_ids.size(1)]
        header_delta = SU_gen[:, SU_ids.size(1):]

        with torch.no_grad():
            pkv_su, S_len, U_len = model.tri_build_caches(S_ids, user_delta, lower_k=K)
            pkv = _fork_cache_view(pkv_su)
            _ = model.tri_step_logits(header_delta, K, pkv, S_len, U_len, logits_to_keep=0, labels=None, write_cache=True)

        ad = _assistant_content_delta(tokenizer, msgs, resp, SU_gen, device)
        if ad.numel() == 0:
            continue

        # clean loss
        out_clean = model.tri_step_logits(ad, K, pkv, S_len, U_len,
                                          logits_to_keep=ad.size(1), labels=ad.clone(), write_cache=False)
        loss_clean = float(out_clean.loss.detach().item())

        # corrupt 10% (ignore_index 위치는 원래 없음이 정상이나, 안전하게 필터)
        vocab = model.config.vocab_size
        mask = (torch.rand_like(ad.float()) < float(corruption)) & (ad != -100)
        ad_corrupt = ad.clone()
        rand_tok = torch.randint(low=0, high=vocab, size=ad.shape, device=ad.device, dtype=ad.dtype)
        ad_corrupt = torch.where(mask, rand_tok, ad_corrupt)

        out_corr = model.tri_step_logits(ad_corrupt, K, pkv, S_len, U_len,
                                         logits_to_keep=ad_corrupt.size(1), labels=ad_corrupt.clone(), write_cache=False)
        loss_corr = float(out_corr.loss.detach().item())

        diffs.append((loss_clean, loss_corr, float(mask.float().mean().item())))

    if not diffs:
        print("\n[corrupt_and_loss] 유효 샘플이 없습니다.")
        return

    print("\n[corrupt_and_loss]")
    for j, (lc, lr, rate) in enumerate(diffs[:10], 1):
        print(f"  ex{j:02d}: clean={lc:.4f}  corrupt={lr:.4f}  (flip_rate≈{rate*100:.1f}%)  Δ={lr-lc:.4f}")
    avg_delta = sum((b - a) for a, b, _ in diffs) / len(diffs)
    print(f"  Δ(corrupt-clean) avg = {avg_delta:.4f}  over {len(diffs)} samples")
    if avg_delta < 0.05:
        print("  [WARN] 교란 후 손실 증가가 미미합니다. 손실/라벨 파이프라인 또는 캐시 경로를 재점검하세요.")


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_file", type=str, required=True)
    ap.add_argument("--lopa_modeling_path", type=str, required=True)
    ap.add_argument("--prefill_layers", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--corruption", type=float, default=0.10)
    ap.add_argument("--cache_dir_model", type=str, default=None)
    ap.add_argument("--cache_dir_tokenizer", type=str, default=None)
    args = ap.parse_args()

    # Patch TRI modeling
    modeling = load_custom_llama_modeling_TRI(Path(args.lopa_modeling_path))
    LlamaForCausalLM = modeling.LlamaForCausalLM

    # Device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
             else (torch.float16 if torch.cuda.is_available() else torch.float32))

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model
    model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, cache_dir="/data2/jeongseokoh/hub/model")
    model = model.to(device)
    try:
        model.config._attn_implementation = "eager"
        modeling.LlamaModel(model.config)._supports_attention_backend = True
    except Exception:
        pass

    # Data
    ds = QADataset(args.data_file, tok)

    print(f"[INFO] device={device}, dtype={dtype}, |data|={len(ds)}")
    print(f"[INFO] lower_k(prefill_layers) = {args.prefill_layers}")

    # Probes
    quick_probe(model, tok, ds, K=int(args.prefill_layers), samples=int(args.samples), device=device)
    corrupt_and_loss(model, tok, ds, K=int(args.prefill_layers), samples=int(args.samples),
                     corruption=float(args.corruption), device=device)


if __name__ == "__main__":
    main()

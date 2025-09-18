#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSM8K(train) → 수학 추론 데이터 생성기
- 문서 없이 문제만 주고, 답변은 반드시 마지막에 \boxed{...} 로 숫자만 출력
- 각 문제당 5개 응답 생성(매칭된 응답만 저장)
- question_id: gsm8k_{index}
- 출력 포맷(jsonl):
  {
    "sample_idx": int,
    "question_id": "gsm8k_{i}",
    "document": "",               # 비어 있음
    "question": <raw question>,   # 모델 입력으로 쓸 순수 질문
    "responses": [matched_response, ...],
    "matched_boxed": ["24", ...], # \boxed{...} 안의 숫자
    "ground_truth_answer_raw": "<gold_str>",
    "step_by_step": bool,
    "num_responses": int
  }
"""

import os
import json
import random
import gc
import re
import string
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset
from tqdm import tqdm
from decimal import Decimal, InvalidOperation
from fractions import Fraction

# --------------------------
# 1) 설정
# --------------------------
SEED = 42
NUM_SAMPLES = 4000            # GSM8K train 전체가 7k+ 이므로, 필요 개수만
NUM_RETURN_SEQUENCES = 5
MAX_TOKENS = 512
TEMPERATURE = 0.9

CACHE_DIR_DATA = "/data/jeongseokoh/data"
CACHE_DIR_TOKENIZER = "/data/jeongseokoh/hub/tokenizer"
CACHE_DIR_MODEL = "/data/jeongseokoh/hub/model"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# 메모리/배치
BATCH_SIZE = 50
MEMORY_CLEANUP_INTERVAL = 100

# \boxed 강제 템플릿(숫자만)
FORCED_CONCLUDE_TEMPLATE = (
    "Ensure your final answer is presented within the format '#### {numeric answer}'."
)

# --------------------------
# 2) 시드/디바이스
# --------------------------
set_seed(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    torch.cuda.empty_cache()

# --------------------------
# 3) 데이터 로드(GSM8K)
# --------------------------
print("GSM8K(train) 로드 중...")
ds = load_dataset("openai/gsm8k", "main", cache_dir=CACHE_DIR_DATA)
train_data = list(ds["train"])
random.shuffle(train_data)
selected_samples = train_data[:NUM_SAMPLES]
print(f"총 {len(train_data)}개 중 {len(selected_samples)}개 샘플 사용 (seed={SEED})")

# --------------------------
# 4) 모델/토크나이저
# --------------------------
print("모델 및 토크나이저 로드 중...")
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "최소 1개의 GPU가 필요합니다."
# 각 GPU 메모리의 90%를 max_memory로 설정
def _mem_str(i):
    cap = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    return f"{int(cap * 0.9)}GiB"
max_mem = {i: _mem_str(i) for i in range(n_gpu)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR_TOKENIZER)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory=max_mem,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"모델 로드 완료: {MODEL_NAME}")

# --------------------------
# 5) 메모리 유틸
# --------------------------
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB"
    return "CUDA 불가"

# --------------------------
# 6) 정규화/추출/정답 비교
# --------------------------
_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})
_BOXED_LAST_RE = re.compile(r"\\(?:boxed|fbox)\{([^{}]*)\}")

# #### 숫자 추출용 정규식
_GSM_FINAL_RE = re.compile(r"####\s*([^\n]+)")

def last_hash_answer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "No Match"
    matches = list(_GSM_FINAL_RE.finditer(text))
    if not matches:
        return "No Match"
    content = matches[-1].group(1).strip()
    return content if content else "No Match"

def normalize_simple(s: str) -> str:
    """간단 텍스트 정규화(비수학적)."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_GSM_FINAL_RE = re.compile(r"####\s*(.+)")

def extract_gsm8k_gold(answer_field: str) -> str:
    """
    GSM8K의 gold answer는 '#### 24' 형태가 보통 마지막 줄에 존재.
    해당 부분만 추출(좌우 공백/마침표 제거, 쉼표 제거).
    """
    if not isinstance(answer_field, str): 
        return ""
    m = _GSM_FINAL_RE.search(answer_field)
    if not m:
        return ""
    s = m.group(1).strip()
    s = s.rstrip(".")
    s = s.replace(",", "")
    return s

def parse_number_like(s: str):
    """
    '24', '-3.5', '1/2', '50%' → 수치로 파싱 시도
    - 성공: (True, Fraction or Decimal)
    - 실패: (False, None)
    """
    if not isinstance(s, str):
        return (False, None)
    t = s.strip()
    if not t:
        return (False, None)
    # 퍼센트
    if t.endswith("%"):
        try:
            val = Decimal(t[:-1].strip())
            return (True, Decimal(val) / Decimal(100))
        except InvalidOperation:
            return (False, None)
    # 분수
    if "/" in t:
        num, *rest = t.split("/")
        if len(rest) == 1:
            den = rest[0]
            if num.strip().lstrip("-").isdigit() and den.strip().isdigit():
                try:
                    return (True, Fraction(int(num), int(den)))
                except ZeroDivisionError:
                    return (False, None)
    # 일반 숫자
    try:
        return (True, Decimal(t))
    except InvalidOperation:
        return (False, None)

def numeric_equiv(a: str, b: str, tol: Decimal = Decimal("1e-9")) -> bool:
    """
    숫자/분수/퍼센트 등 수치 동등성 허용 비교 (허용오차 tol).
    """
    ok_a, va = parse_number_like(a.replace(",", ""))
    ok_b, vb = parse_number_like(b.replace(",", ""))
    if ok_a and ok_b:
        # Fraction vs Decimal 비교 → Decimal로 통일
        if isinstance(va, Fraction):
            va = Decimal(va.numerator) / Decimal(va.denominator)
        if isinstance(vb, Fraction):
            vb = Decimal(vb.numerator) / Decimal(vb.denominator)
        diff = (va - vb).copy_abs()
        return diff <= tol
    # 숫자로 해석 안되면 문자열 동치(정규화)로 fallback
    return normalize_simple(a) == normalize_simple(b)

def exact_or_numeric_match(pred_extracted: str, gold_str: str) -> bool:
    if pred_extracted == "No Match" or not gold_str:
        return False
    # 1) 숫자 동등성 우선
    if numeric_equiv(pred_extracted, gold_str):
        return True
    # 2) 보수적으로 텍스트 정규화 비교
    return normalize_simple(pred_extracted) == normalize_simple(gold_str)

# --------------------------
# 7) 프롬프트 생성
# --------------------------
def create_prompt_math(question: str, step_by_step: bool = False):
    system_prompt = (
        "You are a methodical mathematician, adept at solving complex mathematical problems. Conclude your explanation with the answer in a '#### {numeric answer}' format, where the answer is solely a number."
    )
    user_prompt = f"Question: {question}"
    if step_by_step:
        user_prompt += "\nThink step by step."
    user_prompt += f"\n{FORCED_CONCLUDE_TEMPLATE}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, user_prompt

# --------------------------
# 8) 안전 생성
# --------------------------
@torch.inference_mode()
def generate_responses_safe(messages, num_return_sequences=NUM_RETURN_SEQUENCES, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True
        ).to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        prompt_len = inputs.shape[-1]
        responses = []
        for output in outputs:
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            responses.append(response.strip())

        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return responses, None

    except Exception as e:
        cleanup_memory()
        error_msg = str(e)
        print(f"    ⚠️ 생성 에러: {error_msg}")
        if "CUDA" in error_msg or "out of memory" in error_msg:
            print("    🔄 CUDA 에러 감지, 메모리 정리 및 재시도(축소)...")
            cleanup_memory()
            try:
                return generate_responses_safe(messages, num_return_sequences=1, max_tokens=max_tokens//2, temperature=temperature)
            except Exception as e2:
                return [], f"재시도 실패: {error_msg} / {e2}"
        return [], error_msg

# --------------------------
# 9) 체크포인트
# --------------------------
def save_checkpoint(processed_count, output_file):
    checkpoint = {
        "processed_count": processed_count,
        "output_file": output_file
    }
    with open(f"{output_file}.checkpoint", 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint(output_file):
    checkpoint_file = f"{output_file}.checkpoint"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

# --------------------------
# 10) 메인 루프
# --------------------------
def process_gsm8k_train():
    output_file = f"gsm8k_train_{NUM_RETURN_SEQUENCES}resp_seed{SEED}_samples{NUM_SAMPLES}_boxed_numeric_exact.jsonl"

    # 체크포인트
    checkpoint = load_checkpoint(output_file)
    start_idx = checkpoint["processed_count"] if checkpoint else 0
    if start_idx > 0:
        print(f"📁 체크포인트에서 재시작: {start_idx}개 처리 완료")
    else:
        if os.path.exists(output_file):
            os.remove(output_file)

    # step-by-step 절반 유지
    midpoint = len(selected_samples) // 2
    normal_samples = [(i, s) for i, s in enumerate(selected_samples[:midpoint])]
    step_samples   = [(midpoint + i, s) for i, s in enumerate(selected_samples[midpoint:])]
    random.shuffle(normal_samples)
    random.shuffle(step_samples)
    tasks = normal_samples + step_samples

    current_batch = []
    batch_num = 0
    processed_samples = 0
    total_samples_with_match = 0
    total_records_saved = 0
    total_matched_responses = 0

    def handle_one(i_local: int, sample: Dict[str, Any], step_by_step: bool):
        nonlocal processed_samples, current_batch, batch_num
        nonlocal total_samples_with_match, total_records_saved, total_matched_responses

        if processed_samples < start_idx:
            processed_samples += 1
            return

        q = sample.get("question", "").strip()
        gold_raw = extract_gsm8k_gold(sample.get("answer", ""))  # "#### ..."에서 추출
        qid = f"gsm8k_{i_local}"

        if not q or not gold_raw:
            print(f"⚠️ 샘플 {i_local}: 유효하지 않은 question/gold → 스킵")
            processed_samples += 1
            return

        print(f"\n--- GSM8K {processed_samples+1}/{NUM_SAMPLES} (메모리: {check_gpu_memory()}) ---")
        print(f"    step_by_step={step_by_step} | qid={qid} | gold='{gold_raw}'")

        # 메시지 구성(문서 없음)
        messages, user_prompt = create_prompt_math(q, step_by_step=step_by_step)
        responses, error = generate_responses_safe(messages, num_return_sequences=NUM_RETURN_SEQUENCES)

        if error:
            print(f"  생성 실패 - {error}")
            processed_samples += 1
            return

        matched_responses = []
        matched_boxed = []
        for r in responses:
            bx = last_hash_answer(r)
            if exact_or_numeric_match(bx, gold_raw):
                matched_responses.append(r)
                matched_boxed.append(bx)

        if matched_responses:
            total_samples_with_match += 1
            rec = {
                "sample_idx": i_local,
                "question_id": qid,
                "document": "",          # ← 문서 없음
                "question": q,           # ← 순수 질문만 저장(훈련시 'Question: {q}')
                "responses": matched_responses,
                "matched_boxed": matched_boxed,
                "ground_truth_answer_raw": gold_raw,
                "step_by_step": step_by_step,
                "num_responses": len(matched_responses),
            }
            current_batch.append(rec)
            total_records_saved += 1
            total_matched_responses += len(matched_responses)
            print(f"  ✅ 매칭 응답 {len(matched_responses)}개 저장")

            if len(current_batch) >= BATCH_SIZE:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for r in current_batch:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"✅ 배치 {batch_num} 저장 (레코드 {len(current_batch)}개, 누적={total_records_saved}, 응답누적={total_matched_responses})")
                current_batch.clear()
                batch_num += 1
        else:
            print(f"  ❌ 매칭된 응답 없음 → 저장 안함")

        processed_samples += 1

        if processed_samples % MEMORY_CLEANUP_INTERVAL == 0:
            cleanup_memory()
            save_checkpoint(processed_samples, output_file)
            print(f"🔄 메모리 정리 및 체크포인트 저장 (진행률: {processed_samples}/{NUM_SAMPLES})")

    # 실행
    try:
        for idx, sample in tqdm(tasks, desc="GSM8K 처리", initial=start_idx):
            step_flag = (idx >= midpoint)  # 후반부는 step-by-step True
            handle_one(idx, sample, step_flag)
    except KeyboardInterrupt:
        print("\n⏹️ 중단 - 현재까지 결과 저장")
        save_checkpoint(processed_samples, output_file)
    except Exception as e:
        print(f"\n❌ 예외: {e}")
        save_checkpoint(processed_samples, output_file)
        raise

    # 남은 배치 저장
    if current_batch:
        with open(output_file, 'a', encoding='utf-8') as f:
            for r in current_batch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ 마지막 배치 저장 (레코드 {len(current_batch)}개)")

    cleanup_memory()
    print("\n🎉 완료!")
    print(f"   - 매칭된 샘플 수: {total_samples_with_match}")
    print(f"   - 저장된 레코드 수: {total_records_saved}")
    print(f"   - 총 저장 응답 수: {total_matched_responses}")
    print(f"📄 출력 파일: {output_file}")
    return output_file

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    out = process_gsm8k_train()
    print("\n📊 파일 검증:")
    try:
        n_lines = 0
        n_resps = 0
        with open(out, 'r', encoding='utf-8') as f:
            for line in f:
                n_lines += 1
                n_resps += len(json.loads(line).get("responses", []))
        print(f"   - 파일 레코드 수: {n_lines}")
        print(f"   - 파일 내 저장된 응답 총 수: {n_resps}")
    except FileNotFoundError:
        print("   - 출력 파일이 존재하지 않습니다.")

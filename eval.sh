#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Hugging Face token (optional)
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

## Example runs (uncomment and tweak as needed):
# DATASETS="gsm8k" MAX_SAMPLES=200 VERBOSE=1 ./eval.sh
# DATASETS="hotpotqa" DATASET_CONFIG="hotpot_variant=fullwiki hotpot_max_paras=3" ./eval.sh
# jeongseokoh/LoPA_Llama3.1_8B_8_Lowers
REPO_ID="${REPO_ID:-jeongseokoh/LoPA_Llama3.1_8B_8_Lowers}"
DATASETS="${DATASETS:-hotpotqa}"
BASE_SUBFOLDER="${BASE_SUBFOLDER:-base}"
LORA_SUBFOLDER="${LORA_SUBFOLDER:-lora}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/lopa_llama_modeling.py}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant that answers questions based on the given document.\nAt the end of your explanation, wrap the answer in '\\boxed{answer}'.}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/evaluation_outputs}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
LOWER_K="${LOWER_K:-8}"  # optional override
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"  # optional limit per dataset
DATASET_CONFIG="${DATASET_CONFIG:-hotpot_variant=distractor hotpot_max_paras=10 hotpot_only_gold=True}" # space separated key=value pairs
INCLUDE_COMBINED_DOCS="${INCLUDE_COMBINED_DOCS:-0}"
VERBOSE="${VERBOSE:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
DTYPE="${DTYPE:-auto}"
DEVICE="${DEVICE:-}"  # optional e.g. cuda:0
VANILLA="${VANILLA:-0}"

if [[ "$REPO_ID" == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
  VANILLA="1"
  LORA_SUBFOLDER=""
fi

IFS=' ' read -r -a DATASET_ARRAY <<< "$DATASETS"

CMD=(
  python3 -m evaluation.evaluate
  --repo-id "$REPO_ID"
  --base-subfolder "$BASE_SUBFOLDER"
  --lora-subfolder "$LORA_SUBFOLDER"
  --modeling-path "$MODEL_PATH"
  --attn-impl "$ATTN_IMPL"
  --system-prompt "$SYSTEM_PROMPT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --repetition-penalty "$REPETITION_PENALTY"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --output-prefix "$OUTPUT_PREFIX"
  --log-interval "$LOG_INTERVAL"
  --dtype "$DTYPE"
  --datasets
)

for ds in "${DATASET_ARRAY[@]}"; do
  CMD+=( "$ds" )
done

if [[ -n "$DEVICE" ]]; then
  CMD+=( --device "$DEVICE" )
fi

if [[ "$VANILLA" == "1" ]]; then
  CMD+=( --vanilla )
fi

# dataset-level overrides
if [[ -n "$DATASET_CONFIG" ]]; then
  for kv in $DATASET_CONFIG; do
    CMD+=( --dataset-config "$kv" )
  done
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=( --max-samples "$MAX_SAMPLES" )
fi

if [[ -n "$LOWER_K" ]]; then
  CMD+=( --lower-k "$LOWER_K" )
fi

if [[ "$INCLUDE_COMBINED_DOCS" == "1" ]]; then
  CMD+=( --include-combined-docs )
fi

if [[ "$VERBOSE" == "1" ]]; then
  CMD+=( --verbose )
fi

LOG_DIR="${LOG_DIR:-$REPO_ROOT/evaluation/logs}"
mkdir -p "$LOG_DIR"
eval_ts=$(date +%Y%m%d_%H%M%S)
OUT_LOG="$LOG_DIR/eval_${eval_ts}.out"
ERR_LOG="$LOG_DIR/eval_${eval_ts}.err"

mkdir -p "$OUTPUT_DIR"

echo "[eval.sh] CWD: $(pwd)"
echo "[eval.sh] Command: ${CMD[*]}"
"${CMD[@]}" > >(tee "$OUT_LOG") 2> >(tee "$ERR_LOG" >&2)

#!/usr/bin/env python3
"""CLI for running LoPA TRI evaluation against supported datasets."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from evaluation.dataset_utils import (
    GSM8KIterator,
    HotpotQAIterator,
    NQIterator,
    TriviaQAIterator,
)
from evaluation.tri_infer import TRIModelConfig, TRIModelRunner
from evaluation.utils.utils import ScoredPrediction, aggregate_scores, score_prediction

DATASET_REGISTRY = {
    "gsm8k": GSM8KIterator,
    "hotpotqa": HotpotQAIterator,
    "nq": NQIterator,
    "triviaqa": TriviaQAIterator,
}


def _coerce_value(raw: str):
    text = raw.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower == "none":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_key_values(pairs: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Dataset config '{item}' must be key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Invalid dataset config key in '{item}'")
        result[key] = _coerce_value(value)
    return result


def _clip_text(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_oom_error(err: BaseException) -> bool:
    """Return True if the exception corresponds to an out-of-memory failure."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    if isinstance(err, MemoryError):
        return True
    message = str(err).lower()
    return "out of memory" in message


def evaluate_dataset(
    name: str,
    dataset_cfg: Dict[str, object],
    runner: TRIModelRunner,
    system_prompt: str,
    gen_kwargs: Dict[str, object],
    max_samples: int | None,
    log_interval: int,
    verbose: bool,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[ScoredPrediction]]:
    iterator_cls = DATASET_REGISTRY[name]
    dataset = iterator_cls(dataset_cfg)
    results: List[Dict[str, object]] = []
    scored_items: List[ScoredPrediction] = []
    latencies: List[float] = []
    em_count = 0
    contain_count = 0
    f1_total = 0.0

    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        question = sample.get("q_with_boxed") or sample.get("question", "")
        doc = sample.get("combined_docs")
        if not doc:
            docs = sample.get("documents") or []
            sep = dataset_cfg.get("doc_separator", "\n\n----[DOC SEP]----\n\n")
            doc = sep.join(docs)
        start_time = time.perf_counter()
        try:
            prediction = runner.generate(system_prompt, doc, question, **gen_kwargs)
        except Exception as err:
            if not is_oom_error(err):
                raise
            print(f"[{name}] #{idx + 1} | Out of memory encountered; skipping sample.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        latency = time.perf_counter() - start_time
        scored = score_prediction(prediction, sample.get("ground_truths", []))
        if scored.exact_match:
            em_count += 1
        if scored.contains_match:
            contain_count += 1
        f1_total += scored.f1
        record = {
            "dataset": name,
            "index": idx,
            "question": sample.get("question"),
            "question_with_boxed": question,
            "documents": sample.get("documents"),
            "combined_docs": doc,
            "ground_truths": sample.get("ground_truths", []),
            "prediction": prediction,
            "prediction_boxed": scored.prediction_boxed,
            "exact_match": scored.exact_match,
            "contains_match": scored.contains_match,
            "boxed_found": scored.boxed_found,
            "latency_sec": latency,
        }
        results.append(record)
        scored_items.append(scored)
        latencies.append(latency)
        cur_total = len(results)
        running_acc = em_count / cur_total if cur_total else 0.0
        q_preview = _clip_text(sample.get("question") or question, 160)
        gt_preview = _clip_text("; ".join(scored.ground_truths), 160)
        pred_preview = _clip_text(scored.prediction_boxed or prediction, 200)
        match_flag = "YES" if scored.exact_match else "NO"
        contain_flag = "YES" if scored.contains_match else "NO"
        running_f1 = f1_total / cur_total if cur_total else 0.0
        if verbose:
            print(
                f"[{name}] #{idx + 1} | EM={match_flag} | F1={scored.f1:.2f} | Contain={contain_flag} | "
                f"running EM={em_count}/{cur_total} ({running_acc:.2%}) | running F1={running_f1:.2f}\n"
                f"  Q: {q_preview}\n"
                f"  GT: {gt_preview}\n"
                f"  Pred: {pred_preview}"
            )
        elif log_interval > 0 and cur_total % log_interval == 0:
            contain_rate = contain_count / cur_total if cur_total else 0.0
            print(
                f"[{name}] Processed {cur_total} samples | EM {em_count}/{cur_total} ({running_acc:.2%}) | "
                f"F1 {running_f1:.2f} | Contain {contain_count}/{cur_total} ({contain_rate:.2%})\n"
                f"  Q: {q_preview}\n"
                f"  GT: {gt_preview}\n"
                f"  Pred: {pred_preview}\n"
                f"  Match: {match_flag} | Contain: {contain_flag} | F1: {scored.f1:.2f}"
            )

    summary = aggregate_scores(scored_items)
    summary.update({
        "dataset": name,
        "num_samples": len(results),
    })
    if latencies:
        summary["avg_latency_sec"] = sum(latencies) / len(latencies)
        summary["median_latency_sec"] = statistics.median(latencies)
    else:
        summary["avg_latency_sec"] = 0.0
        summary["median_latency_sec"] = 0.0
    return summary, results, scored_items


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate LoPA TRI models")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id (user/model)")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_REGISTRY.keys()),
                        help="Datasets to evaluate")
    parser.add_argument("--dataset-config", action="append", default=[], metavar="KEY=VALUE",
                        help="Override dataset iterator config (applied to all datasets)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional limit on samples per dataset")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful assistant that answers questions based on the given document.")
    parser.add_argument("--log-interval", type=int, default=20, help="Progress print frequency (0 to disable)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doc-separator", type=str, default="\n\n")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="If set, save predictions and summary under this directory")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Prefix for saved files inside output directory")
    parser.add_argument("--include-combined-docs", action="store_true", default=False,
                        help="Store combined document string in output (may be large)")
    parser.add_argument("--device", type=str, default=None, help="Explicit device (cpu/cuda)")
    parser.add_argument("--dtype", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")

    parser.add_argument("--base-subfolder", type=str, default="base")
    parser.add_argument("--lora-subfolder", type=str, default="lora")
    parser.add_argument("--modeling-path", type=str, default="lopa_llama_modeling.py")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--lower-k", type=int, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed-text", type=str, default="")
    parser.add_argument("--vanilla", action="store_true", help="Disable TRI path and use standard HF generation.")
    return parser


def resolve_dtype(device: str | None, dtype_arg: str) -> torch.dtype | None:
    if dtype_arg == "auto":
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype arg: {dtype_arg}")
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg in ("bf16", "fp16") and (device or "cuda").startswith("cuda"):
        return mapping[dtype_arg]
    return mapping[dtype_arg]


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    dataset_cfg = parse_key_values(args.dataset_config)
    dataset_cfg.setdefault("doc_separator", args.doc_separator)
    dataset_cfg.setdefault("seed", args.seed)

    lora_subfolder = args.lora_subfolder.strip() or None
    dtype = resolve_dtype(args.device, args.dtype)

    use_tri = not args.vanilla
    if args.repo_id == "meta-llama/Llama-3.1-8B-Instruct":
        use_tri = False

    model_cfg = TRIModelConfig(
        repo_id=args.repo_id,
        base_subfolder=args.base_subfolder,
        lora_subfolder=lora_subfolder,
        modeling_path=Path(args.modeling_path),
        attn_impl=args.attn_impl,
        lower_k=args.lower_k if use_tri else None,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed_text=args.seed_text,
        use_tri=use_tri,
    )

    setup_seed(args.seed)

    runner = TRIModelRunner(model_cfg)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed_text,
    }
    if use_tri and args.lower_k is not None:
        gen_kwargs["lower_k"] = args.lower_k

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    model_output_dir = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_repo = args.repo_id.replace("/", "__")
        model_output_dir = output_dir / sanitized_repo
        model_output_dir.mkdir(parents=True, exist_ok=True)

    cfg_suffix_parts: List[str] = []
    if args.dataset_config:
        for item in args.dataset_config:
            sanitized = item.replace("/", "-").replace(" ", "").replace("=", "-")
            cfg_suffix_parts.append(sanitized)
    cfg_suffix = ("_" + "_".join(cfg_suffix_parts)) if cfg_suffix_parts else ""

    summaries: List[Dict[str, float]] = []
    all_scores: List[ScoredPrediction] = []

    for name in args.datasets:
        print(f"=== Evaluating {name} ===")
        summary, records, scores = evaluate_dataset(
            name=name,
            dataset_cfg=dataset_cfg,
            runner=runner,
            system_prompt=args.system_prompt,
            gen_kwargs=gen_kwargs,
            max_samples=args.max_samples,
            log_interval=args.log_interval,
            verbose=args.verbose,
        )
        summaries.append(summary)
        all_scores.extend(scores)
        print(json.dumps(summary, indent=2))
        target_dir = model_output_dir or output_dir
        if target_dir:
            prefix = args.output_prefix or "eval"
            pred_path = target_dir / f"{prefix}_{name}{cfg_suffix}_predictions.jsonl"
            with pred_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    if not args.include_combined_docs:
                        rec = dict(rec)
                        rec.pop("combined_docs", None)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    overall = aggregate_scores(all_scores)
    overall.update({"dataset": "overall", "num_samples": sum(s["num_samples"] for s in summaries)})
    print("=== Overall ===")
    print(json.dumps(overall, indent=2))

    target_dir = model_output_dir or output_dir
    if target_dir:
        prefix = args.output_prefix or "eval"
        summary_path = target_dir / f"{prefix}{cfg_suffix}_summary.json"
        payload = {
            "summaries": summaries,
            "overall": overall,
            "config": {
                "repo_id": args.repo_id,
                "datasets": args.datasets,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "lower_k": runner.lower_k,
                "mode": "tri" if use_tri else "vanilla",
                "seed": args.seed,
            },
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Utilities shared across evaluation scripts."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, List, Sequence

_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})
_BOXED_REGEX = re.compile(r"\\(?:boxed|fbox)\{([^{}]*)\}")
_HASH_REGEX = re.compile(r"####\s*([^\n]+)")


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    s = text.strip().lower()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return "No Match"


def last_boxed_only_string(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "No Match"
    matches = list(_BOXED_REGEX.finditer(text))
    if matches:
        content = matches[-1].group(1).strip()
        if content:
            return content
    # back-up to #### marker used in GSM8K golds
    hash_matches = list(_HASH_REGEX.finditer(text))
    if hash_matches:
        content = hash_matches[-1].group(1).strip()
        if content:
            return content
    return "No Match"


def exact_match_score(pred_extracted: str, gold_norm: str) -> bool:
    if pred_extracted == "No Match":
        return False
    return normalize_answer(pred_extracted) == gold_norm


def contains_match(prediction: str, golds: Sequence[str]) -> bool:
    if not isinstance(prediction, str):
        return False
    pred_norm = normalize_answer(prediction)
    for gold in golds:
        gold_norm = normalize_answer(gold)
        if not gold_norm:
            continue
        if gold_norm in pred_norm:
            return True
    return False


def _f1_score(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    pred_overlap = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
    precision = pred_overlap / len(pred_tokens)
    recall = pred_overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class ScoredPrediction:
    prediction: str
    prediction_boxed: str
    prediction_norm: str
    ground_truths: List[str]
    ground_truths_norm: List[str]
    exact_match: bool
    contains_match: bool
    boxed_found: bool
    f1: float


def score_prediction(prediction: str, ground_truths: Sequence[str]) -> ScoredPrediction:
    gts = list(ground_truths)
    gts_norm = [normalize_answer(g) for g in gts]
    boxed = last_boxed_only_string(prediction)
    boxed_norm = normalize_answer(boxed) if boxed != "No Match" else ""
    em = boxed != "No Match" and any(boxed_norm == gold for gold in gts_norm)
    contains = contains_match(prediction, gts)
    pred_base = boxed_norm if boxed_norm else normalize_answer(prediction)
    pred_tokens = pred_base.split()
    f1 = 0.0
    if gts_norm:
        f1 = max(_f1_score(pred_tokens, gold.split()) for gold in gts_norm)
    return ScoredPrediction(
        prediction=prediction,
        prediction_boxed=boxed,
        prediction_norm=normalize_answer(prediction),
        ground_truths=gts,
        ground_truths_norm=gts_norm,
        exact_match=em,
        contains_match=contains,
        boxed_found=(boxed != "No Match"),
        f1=f1,
    )


def aggregate_scores(items: Iterable[ScoredPrediction]) -> dict:
    total = 0
    em = 0
    contains = 0
    boxed = 0
    f1_sum = 0.0
    for item in items:
        total += 1
        if item.exact_match:
            em += 1
        if item.contains_match:
            contains += 1
        if item.boxed_found:
            boxed += 1
        f1_sum += item.f1
    if total == 0:
        return {"total": 0, "exact_match": 0, "contains_match": 0, "boxed_found": 0,
                "exact_match_rate": 0.0, "contains_match_rate": 0.0, "boxed_found_rate": 0.0,
                "avg_f1": 0.0}
    return {
        "total": total,
        "exact_match": em,
        "contains_match": contains,
        "boxed_found": boxed,
        "exact_match_rate": em / total,
        "contains_match_rate": contains / total,
        "boxed_found_rate": boxed / total,
        "avg_f1": f1_sum / total,
    }


__all__ = [
    "normalize_answer",
    "remove_boxed",
    "last_boxed_only_string",
    "exact_match_score",
    "contains_match",
    "score_prediction",
    "aggregate_scores",
    "ScoredPrediction",
]

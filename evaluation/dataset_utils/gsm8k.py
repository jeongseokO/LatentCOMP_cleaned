# -*- coding: utf-8 -*-
import re
from typing import Any, Dict, Iterator, List, Optional
from datasets import load_dataset
from .base import BaseDataset
from evaluation.utils.utils import normalize_answer, last_boxed_only_string  # 외부 제공

def _extract_final(answer: str) -> Optional[str]:
    if not isinstance(answer, str):
        return None
    m = re.search(r"####\s*(.+)$", answer.strip())
    if m:
        return m.group(1).strip()
    # 백업: 마지막 \boxed{...}
    b = last_boxed_only_string(answer)
    return b if b != "No Match" else None

def _fewshot_examples(boxed: bool = True) -> List[str]:
    def ex(title_q, outline_lines, ans):
        lines = "\n".join(f"- {ln}" for ln in outline_lines)
        tail = f"\nTherefore, the answer is \\boxed{{{ans}}}." if boxed else f"\nAnswer: {ans}."
        return f"Example\nQ: {title_q}\nSolution Outline:\n{lines}{tail}"

    return [
        ex(
            "There are 15 trees in a grove. After planting, there are 21 trees. How many trees were planted?",
            ["Let initial = 15, final = 21.",
             "Compute planted = final − initial.",
             "planted = 21 − 15 = 6."],
            "6"
        ),
        ex(
            "A book costs $8 and a pen costs $2. If Jenny buys 4 books and 3 pens, how much does she pay?",
            ["Books cost = 4 × 8 = 32.",
             "Pens cost = 3 × 2 = 6.",
             "Total cost = 32 + 6 = 38."],
            "38"
        ),
        ex(
            "Sarah had 24 candies and gave 1/3 of them to a friend. How many candies does she have left?",
            ["Given away = (1/3) × 24 = 8.",
             "Remaining = 24 − 8 = 16."],
            "16"
        ),
        ex(
            "A recipe needs 3/4 cup of sugar per batch. If you have 6 cups of sugar, how many batches can you make?",
            ["Batches = total ÷ per_batch = 6 ÷ (3/4).",
             "Compute 6 ÷ (3/4) = 6 × 4/3 = 8."],
            "8"
        ),
    ]

class GSM8KIterator(BaseDataset):
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        print("[GSM8K] Loading dataset...")
        split = self.cfg.get("gsm8k_split", "test")
        ds = load_dataset("openai/gsm8k", split=split)

        few_docs = _fewshot_examples(boxed=self.cfg.get("boxed_format", True))
        combined_few = self.SEP.join(f"[Example {j+1}] {doc}" for j, doc in enumerate(few_docs))

        for rec in ds:
            q_plain = (rec.get("question", "") or "").strip()
            raw_ans = rec.get("answer", "")
            final = _extract_final(raw_ans)
            if not q_plain or not final:
                continue
            gts = [normalize_answer(final)]
            q = q_plain + ("\nAt the end of your explanation, wrap the answer in '\\boxed{answer}'."
                           if self.cfg.get("boxed_format", True) else "")
            yield {
                "question": q_plain,
                "q_with_boxed": q,
                "documents": few_docs,
                "combined_docs": combined_few,
                "ground_truths": gts,
            }

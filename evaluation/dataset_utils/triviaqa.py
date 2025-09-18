# -*- coding: utf-8 -*-
from typing import Any, Dict, Iterator, List
from datasets import load_dataset
from .base import BaseDataset
from evaluation.utils.utils import normalize_answer  # 외부 제공

class TriviaQAIterator(BaseDataset):
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        print("[TriviaQA] Loading dataset...")
        ds = load_dataset("mandarjoshi/trivia_qa",
                          self.cfg.get("triviaqa_config_name","rc"),
                          streaming=True, cache_dir="/data2/jeongseokoh/data")["validation"]
        ds = ds.shuffle(seed=self.cfg["seed"])
        N = int(self.cfg.get("triviaqa_max_docs", 1))
        print(f"Using up to {N} documents per question.")
        for sample in ds:
            docs: List[str] = sample.get("search_results", {}).get("search_context", []) or []
            if not docs:
                continue
            docs = docs[:N]
            combined_docs = self.SEP.join(f"[DOC {j+1}] {doc}" for j, doc in enumerate(docs))
            q_plain = sample["question"]
            q = q_plain + ("\nAt the end of your explanation, wrap the answer in '\\boxed{answer}'."
                           if self.cfg.get("boxed_format", True) else "")
            ans_obj = sample["answer"]
            gts = ans_obj.get("normalized_aliases") or [ans_obj.get("normalized_value", "")]
            gts = [normalize_answer(gt) for gt in gts]
            yield {"question": q_plain, "q_with_boxed": q, "documents": docs,
                   "combined_docs": combined_docs, "ground_truths": gts}

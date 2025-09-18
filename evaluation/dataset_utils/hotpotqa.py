# -*- coding: utf-8 -*-
import os, json, urllib.request, random
from typing import Any, Dict, Iterator, List, Tuple
from .base import BaseDataset
from evaluation.utils.utils import normalize_answer  # 외부 제공

def _download_hotpot_if_needed(variant: str, cache_dir: str = "./.cache_hotpot") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    url_map = {
        "distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "fullwiki":   "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    }
    if variant not in url_map:
        raise ValueError(f"hotpot_variant must be 'distractor' or 'fullwiki', got: {variant}")
    url = url_map[variant]
    local_path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(local_path):
        print(f"[HotpotQA] downloading → {local_path}")
        urllib.request.urlretrieve(url, local_path)
    return local_path

class HotpotQAIterator(BaseDataset):
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        print("[HotpotQA] Loading dataset...")
        variant = str(self.cfg.get("hotpot_variant", "distractor")).lower()
        print(f"HotpotQA variant: {variant}")
        gold_only = self.cfg.get("hotpot_only_gold", False)
        if gold_only:
            print("hotpot_only_gold=True → using only supporting_facts paragraphs.")
        use_titles = bool(self.cfg.get("hotpot_use_titles", True))
        K = int(self.cfg.get("hotpot_max_paras", 2))
        print(f"Using up to {K} paragraphs per question.")
        local_path = self.cfg.get("hotpot_local_path") or _download_hotpot_if_needed(
            variant, cache_dir=self.cfg.get("hotpot_cache_dir", "./.cache_hotpot"))
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rng = random.Random(self.cfg.get("seed", 42)); rng.shuffle(data)
        for sample in data:
            # Collect supporting_facts titles in order (unique) for gold-only mode
            supporting = sample.get("supporting_facts") or []
            gold_titles_ordered: List[str] = []
            if isinstance(supporting, list):
                for sf in supporting:
                    # sf is usually [title, sentence_idx]
                    if isinstance(sf, (list, tuple)) and len(sf) >= 1:
                        t = sf[0]
                        if isinstance(t, str) and t not in gold_titles_ordered:
                            gold_titles_ordered.append(t)
            ctx = sample.get("context", []) or []
            if not ctx:
                continue
            paras: List[str] = []
            title_to_para: Dict[str, str] = {}
            for pair in ctx:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                title, sents = pair
                text = " ".join(sents) if isinstance(sents, list) else str(sents)
                para = f"{title}: {text}" if (use_titles and isinstance(title, str)) else text
                if not para.strip():
                    continue
                # If gold_only is enabled and we're on the distractor split, keep only supporting titles
                if gold_only and variant == "distractor" and isinstance(title, str):
                    title_to_para[title] = para
                else:
                    paras.append(para)

            # If gold-only mode requested on distractor, reorder paras to match supporting_facts title order
            if gold_only and variant == "distractor":
                paras = [title_to_para[t] for t in gold_titles_ordered if t in title_to_para]
            if not paras:
                continue
            docs = paras[:K]
            combined_docs = self.SEP.join(f"[DOC {j+1}] {doc}" for j, doc in enumerate(docs))
            q_plain = sample.get("question", "")
            q = q_plain + ("\nAt the end of your explanation, wrap the answer in '\\boxed{answer}'."
                           if self.cfg.get("boxed_format", True) else "")
            gt = sample.get("answer", "")
            gts = [normalize_answer(gt)]
            yield {"question": q_plain, "q_with_boxed": q, "documents": docs,
                   "combined_docs": combined_docs, "ground_truths": gts}

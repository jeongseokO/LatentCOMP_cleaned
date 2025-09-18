# -*- coding: utf-8 -*-
from typing import Any, Dict, Iterator, List, Optional
from datasets import load_dataset
from .base import BaseDataset
from evaluation.utils.utils import normalize_answer  # 외부 제공
import warnings

def _maybe_seqdict_to_list(seq_or_list):
    if isinstance(seq_or_list, dict) and all(isinstance(v, (list, tuple)) for v in seq_or_list.values()):
        n = len(next(iter(seq_or_list.values()), []))
        out = []
        for i in range(n):
            out.append({k: (seq_or_list[k][i] if i < len(seq_or_list[k]) else None) for k in seq_or_list})
        return out
    return seq_or_list

def _tokens_span_to_text(doc_tokens: Dict[str, List[Any]], start_token: int, end_token: int) -> str:
    if not isinstance(doc_tokens, dict) or "token" not in doc_tokens:
        return ""
    toks = doc_tokens.get("token") or []
    is_html = doc_tokens.get("is_html") or [False] * len(toks)
    N = len(toks)
    s = max(0, int(start_token or 0))
    e = min(N, int(end_token or 0))
    words = [toks[i] for i in range(s, e) if i < N and not is_html[i]]
    return " ".join(words).strip()

def _tokens_all_plain(doc_tokens: Dict[str, List[Any]]) -> str:
    toks = doc_tokens.get("token") or []
    is_html = doc_tokens.get("is_html") or [False] * len(toks)
    words = [t for t, h in zip(toks, is_html) if not h]
    return " ".join(words).strip()

def _nq_yesno_to_text(v) -> Optional[str]:
    if v is None: return None
    if isinstance(v, str):
        vv = v.strip().upper()
        if vv in ("YES","NO"):
            return vv.lower()
        return None
    try:
        iv = int(v)
    except Exception:
        return None
    if iv == 1: return "yes"
    if iv == 0: return "no"
    return None

class NQIterator(BaseDataset):
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        print("[NQ] Loading dataset...")
        split = self.cfg.get("nq_split", "validation")
        hf_config = self.cfg.get("nq_hf_config", "default")
        streaming = bool(self.cfg.get("nq_streaming", True))

        ds = load_dataset("google-research-datasets/natural_questions",
                          hf_config, split=split, streaming=streaming)
        try:
            ds = ds.shuffle(seed=self.cfg.get("seed", 42), buffer_size=10000)
        except Exception:
            pass

        max_ctxs = int(self.cfg.get("nq_max_ctxs", 2))
        print(f"Using up to {max_ctxs} contexts per question.")
        max_win = int(self.cfg.get("nq_max_window_tokens", 512))
        include_long = bool(self.cfg.get("nq_include_long_answer", True))
        include_short = bool(self.cfg.get("nq_include_short_window", True))
        page_fallback = int(self.cfg.get("nq_page_fallback_tokens", 512))
        use_titles = bool(self.cfg.get("nq_use_titles", True))

        for rec in ds:
            try:
                q_plain = rec.get("question", {}).get("text", "") or rec.get("question", "") or ""
                if not isinstance(q_plain, str) or not q_plain.strip():
                    continue

                doc = rec.get("document", {}) or {}
                tokens = doc.get("tokens") or {}
                annotations = _maybe_seqdict_to_list(rec.get("annotations", [])) or []
                ann = annotations[0] if annotations else {}

                short_answers = _maybe_seqdict_to_list(ann.get("short_answers", [])) or []
                gts: List[str] = []
                for sa in short_answers:
                    txt = sa.get("text", None)
                    if isinstance(txt, str) and txt.strip():
                        gts.append(normalize_answer(txt))
                if not gts:
                    yn = _nq_yesno_to_text(ann.get("yes_no_answer", None))
                    if yn:
                        gts = [normalize_answer(yn)]

                la = ann.get("long_answer", {}) or {}
                la_s = la.get("start_token", -1)
                la_e = la.get("end_token", -1)

                docs: List[str] = []
                if include_long and isinstance(la_s, int) and isinstance(la_e, int) and la_s >= 0 and la_e > la_s:
                    la_text = _tokens_span_to_text(tokens, la_s, la_e)
                    if la_text:
                        docs.append(f"{doc.get('title','')}: {la_text}" if use_titles and doc.get('title') else la_text)

                if include_short and short_answers:
                    sa0 = short_answers[0]
                    if all(k in sa0 for k in ("start_token", "end_token")):
                        s0 = int(sa0["start_token"]); e0 = int(sa0["end_token"])
                        center = max(0, (s0 + e0) // 2)
                        half = max(1, max_win // 2)
                        ws = max(0, center - half)
                        we = center + half
                        win_text = _tokens_span_to_text(tokens, ws, we)
                        if win_text:
                            t = f"{doc.get('title','')}: {win_text}" if use_titles and doc.get('title') else win_text
                            if t not in docs:
                                docs.append(t)

                if len(docs) < max_ctxs:
                    page_text = _tokens_span_to_text(tokens, 0, page_fallback)
                    if not page_text:
                        page_text = _tokens_all_plain(tokens)
                        if page_fallback > 0:
                            page_text = " ".join(page_text.split()[:page_fallback])
                    if page_text:
                        t = f"{doc.get('title','')}: {page_text}" if use_titles and doc.get('title') else page_text
                        if t not in docs:
                            docs.append(t)

                if max_ctxs > 0:
                    docs = docs[:max_ctxs]

                if not gts and isinstance(la_s, int) and isinstance(la_e, int) and la_s >= 0 and la_e > la_s:
                    la_text_norm = normalize_answer(_tokens_span_to_text(tokens, la_s, la_e))
                    if la_text_norm:
                        gts = [la_text_norm]
                if not gts:
                    continue

                combined_docs = self.SEP.join(f"[DOC {j+1}] {d}" for j, d in enumerate(docs))
                q = q_plain + ("\nAt the end of your explanation, wrap the answer in '\\boxed{answer}'."
                               if self.cfg.get("boxed_format", True) else "")

                yield {
                    "question": q_plain,
                    "q_with_boxed": q,
                    "documents": docs,
                    "combined_docs": combined_docs,
                    "ground_truths": gts,
                }
            except Exception as e:
                warnings.warn(f"[NQ] sample skipped due to parse error: {e}")
                continue

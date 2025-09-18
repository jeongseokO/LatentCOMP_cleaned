# -*- coding: utf-8 -*-
from typing import Any, Dict, Iterator

class BaseDataset:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.SEP = cfg.get("doc_separator", "\n")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

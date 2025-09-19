from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


@dataclass
class TRIModelConfig:
    repo_id: str
    base_subfolder: str = "base"
    lora_subfolder: Optional[str] = "lora"
    modeling_path: Path = Path("lopa_llama_modeling.py")
    attn_impl: str = "flash_attention_2"
    lower_k: Optional[int] = None
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    seed_text: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    use_tri: bool = True
    num_specials: int = 0
    special_add_to: str = "none"


class TRIModelRunner:
    def __init__(self, cfg: TRIModelConfig):
        self.cfg = cfg
        self.use_tri = bool(cfg.use_tri)
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if cfg.dtype is not None:
            self.dtype = cfg.dtype
        else:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif self.device.type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        self.num_specials = max(0, int(cfg.num_specials))
        special_target = (cfg.special_add_to or "none").lower()
        self.special_add_to = special_target if special_target in {"user", "assistant"} else "none"
        self.special_tokens: List[str] = []
        self.specials_joined: str = ""
        self.assistant_prefix_ids: Optional[torch.Tensor] = None
        self.assistant_prefix_text: str = ""
        torch.set_grad_enabled(False)
        if self.use_tri:
            self._load_modeling(cfg.modeling_path)
        self._load_tokenizer()
        self._setup_special_tokens()
        self._load_model()
        self.default_gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "seed": cfg.seed_text,
        }
        self.lower_k = self._resolve_lower_k(cfg.lower_k) if self.use_tri else None

    def _load_modeling(self, modeling_path: Path) -> None:
        target = "transformers.models.llama.modeling_llama"
        abs_path = Path(modeling_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"TRI modeling file not found: {abs_path}")
        spec = importlib.util.spec_from_file_location(target, abs_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load TRI modeling spec from {abs_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.pop(target, None)
        sys.modules[target] = module
        spec.loader.exec_module(module)
        import transformers.models.llama as llama_pkg
        setattr(llama_pkg, "modeling_llama", module)
        missing = [name for name in ("tri_build_caches", "tri_forward_assistant", "tri_step_logits") if not hasattr(module.LlamaForCausalLM, name)]
        if missing:
            raise RuntimeError(f"TRI modeling file missing attributes: {missing}")
        self._tri_module = module

    def _load_tokenizer(self) -> None:
        cfg = self.cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.repo_id, use_fast=True, token=cfg.hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_special_tokens(self) -> None:
        if self.num_specials <= 0:
            return
        self.special_tokens = [f"<|Latent{i}|>" for i in range(1, self.num_specials + 1)]
        self.specials_joined = " ".join(self.special_tokens)
        vocab = self.tokenizer.get_vocab()
        missing = [tok for tok in self.special_tokens if tok not in vocab]
        if missing:
            print(f"[warn] Special tokens missing from tokenizer vocab: {missing}")
        if self.special_add_to == "assistant" and self.specials_joined:
            prefix_ids = self.tokenizer(self.specials_joined, add_special_tokens=False, return_tensors="pt").input_ids
            self.assistant_prefix_ids = prefix_ids.to(self.device)
            self.assistant_prefix_text = self.specials_joined

    def _load_model(self) -> None:
        cfg = self.cfg
        if self.use_tri:
            from transformers.models.llama.modeling_llama import LlamaForCausalLM  # type: ignore
            model = LlamaForCausalLM.from_pretrained(
                cfg.repo_id,
                subfolder=cfg.base_subfolder,
                torch_dtype=self.dtype,
                token=cfg.hf_token,
                cache_dir="/data2/jeongseokoh/hub"
            ).to(self.device)
            try:
                from peft import PeftModel
                if cfg.lora_subfolder:
                    peft_model = PeftModel.from_pretrained(
                        model,
                        cfg.repo_id,
                        subfolder=cfg.lora_subfolder,
                        token=cfg.hf_token,
                        cache_dir="/data2/jeongseokoh/hub"
                    )
                    try:
                        model = peft_model.merge_and_unload()
                    except Exception:
                        model = peft_model
                    model = model.to(self.device)
            except Exception:
                pass
            try:
                model.config._attn_implementation = cfg.attn_impl
            except Exception:
                pass
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                cfg.repo_id,
                torch_dtype=self.dtype,
                token=cfg.hf_token,
                cache_dir="/data2/jeongseokoh/hub"
            ).to(self.device)
            try:
                from peft import PeftModel
                if cfg.lora_subfolder:
                    peft_model = PeftModel.from_pretrained(
                        model,
                        cfg.repo_id,
                        subfolder=cfg.lora_subfolder,
                        token=cfg.hf_token,
                        cache_dir="/data2/jeongseokoh/hub"
                    )
                    try:
                        model = peft_model.merge_and_unload()
                    except Exception:
                        model = peft_model
                    model = model.to(self.device)
            except Exception:
                pass
        model.eval()
        self.model = model

    def _resolve_lower_k(self, override: Optional[int]) -> int:
        if override is not None:
            return int(override)
        cfg = self.cfg
        for sub in (None, cfg.base_subfolder):
            try:
                tri_path = hf_hub_download(cfg.repo_id, filename="tri_info.txt", subfolder=sub, token=cfg.hf_token)
            except Exception:
                continue
            txt = Path(tri_path).read_text(encoding="utf-8")
            m = re.search(r"lower_k\s*=\s*(\d+)", txt)
            if m:
                return int(m.group(1))
        return 8

    @staticmethod
    def build_messages(system: str, document: str, question: str, include_query: bool = True) -> List[Dict[str, str]]:
        user = f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def tokens_from_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> torch.LongTensor:
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False)
            template = getattr(self.tokenizer, "chat_template", "") or ""
            if add_generation_prompt and "<|start_header_id|>" in template:
                rendered += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

    @staticmethod
    def lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
        limit = min(a.size(1), b.size(1))
        eq = (a[0, :limit] == b[0, :limit])
        nz = (~eq).nonzero(as_tuple=False)
        return int(nz[0, 0]) if nz.numel() else limit

    @staticmethod
    def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float, repetition_penalty: float,
                     prev_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if repetition_penalty != 1.0 and prev_ids is not None and prev_ids.numel() > 0:
            logits = logits.clone()
            logits[:, prev_ids.unique()] /= repetition_penalty
        logits = logits.float() / max(1e-6, float(temperature))
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 0] = False
            keep = ~mask
            filtered = torch.zeros_like(sorted_probs).masked_scatter_(keep, sorted_probs[keep])
            probs = torch.zeros_like(probs).scatter(dim=-1, index=sorted_idx, src=filtered)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(self, system: str, document: str, question: str, **gen_kwargs) -> str:
        if not self.use_tri:
            return self._generate_vanilla(system, document, question, **gen_kwargs)
        opts = {**self.default_gen_kwargs, **gen_kwargs}
        lower_k = opts.pop("lower_k", self.lower_k)
        msgs = self.build_messages(system, document, question, include_query=True)
        if self.specials_joined and self.special_add_to == "user":
            user_content = msgs[-1]["content"]
            sep = "\n\n" if not user_content.endswith("\n") else ""
            msgs[-1]["content"] = f"{user_content}{sep}{self.specials_joined}"
        system_ids = self.tokens_from_messages(msgs[:1], add_generation_prompt=False)
        su_ids = self.tokens_from_messages(msgs, add_generation_prompt=False)
        su_gen = self.tokens_from_messages(msgs, add_generation_prompt=True)
        lcp = self.lcp_len(system_ids, su_ids)
        user_delta = su_ids[:, lcp:su_ids.size(1)]
        header_delta = su_gen[:, su_ids.size(1):]
        caches, sys_len, user_len = self.model.tri_build_caches(system_ids, user_delta, lower_k=lower_k)
        head = header_delta
        forced_prefix_ids: Optional[List[int]] = None
        if self.assistant_prefix_ids is not None and self.special_add_to == "assistant":
            prefix_ids = self.assistant_prefix_ids.to(self.device)
            head = torch.cat([head, prefix_ids], dim=1)
            forced_prefix_ids = prefix_ids[0].tolist()
        seed = opts.get("seed") or ""
        if seed:
            seed_ids = self.tokenizer(seed, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            head = torch.cat([head, seed_ids], dim=1)
        out = self.model.tri_step_logits(head, lower_k, caches, sys_len, user_len,
                                         logits_to_keep=1, labels=None, write_cache=True)
        logits = out.logits[:, -1, :]
        current = self.sample_top_p(logits, opts["temperature"], opts["top_p"], 1.0, None).unsqueeze(0).to(self.device)
        generated: List[int] = []
        for _ in range(int(opts["max_new_tokens"])):
            out = self.model.tri_step_logits(current, lower_k, caches, sys_len, user_len,
                                             logits_to_keep=1, labels=None, write_cache=True)
            token_id = int(current.item())
            generated.append(token_id)
            if self.tokenizer.eos_token_id is not None and token_id == int(self.tokenizer.eos_token_id):
                break
            logits = out.logits[:, -1, :]
            prev = torch.tensor(generated, device=logits.device, dtype=torch.long).unsqueeze(0)
            current = self.sample_top_p(logits, opts["temperature"], opts["top_p"], opts["repetition_penalty"], prev)
            current = current.unsqueeze(0).to(self.device)
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        if forced_prefix_ids and self.assistant_prefix_text:
            suffix = decoded
            if suffix and not suffix[0].isspace():
                suffix = " " + suffix
            return f"{self.assistant_prefix_text}{suffix}"
        return decoded

    def _generate_vanilla(self, system: str, document: str, question: str, **gen_kwargs) -> str:
        opts = {**self.default_gen_kwargs, **gen_kwargs}
        msgs = self.build_messages(system, document, question, include_query=True)
        if self.specials_joined and self.special_add_to == "user":
            user_content = msgs[-1]["content"]
            sep = "\n\n" if not user_content.endswith("\n") else ""
            msgs[-1]["content"] = f"{user_content}{sep}{self.specials_joined}"
        input_ids = self.tokens_from_messages(msgs, add_generation_prompt=True)
        base_seed = opts.get("seed") or ""
        seed = base_seed
        forced_prefix_text = ""
        if self.specials_joined and self.special_add_to == "assistant":
            forced_prefix_text = self.specials_joined
            seed = forced_prefix_text
            if base_seed:
                if not base_seed[0].isspace():
                    seed += " "
                seed += base_seed
            else:
                seed += " "
        if seed:
            seed_ids = self.tokenizer(seed, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            input_ids = torch.cat([input_ids, seed_ids], dim=1)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        generation = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(opts["max_new_tokens"]),
            temperature=float(opts["temperature"]),
            top_p=float(opts["top_p"]),
            repetition_penalty=float(opts["repetition_penalty"]),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = generation[0, input_ids.size(1):]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if forced_prefix_text:
            suffix = decoded
            if suffix and not suffix[0].isspace():
                suffix = " " + suffix
            return f"{forced_prefix_text}{suffix}"
        return decoded


__all__ = ["TRIModelConfig", "TRIModelRunner"]

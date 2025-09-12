"""
Unified wrapper around Llama partial-layer model to enforce generate() defaults:
- compress=False → use_lora=False by default (vanilla base only)
- compress=True  → use_lora=True by default (attach/merge adapter if present)

It reuses the full implementation from modeling_partial_layer.LlamaForCausalLM
and only overrides `generate` to adjust the `use_lora` default.
"""
from __future__ import annotations

from typing import Optional

from modeling_partial_layer import LlamaForCausalLM as _BaseLlamaForCausalLM


class LlamaForCausalLM(_BaseLlamaForCausalLM):
    def generate(
        self,
        *args,
        system: Optional[str] = None,
        query: Optional[str] = None,
        document: Optional[str] = None,
        compress: Optional[bool] = None,
        tokenizer=None,
        prefill_layers: int = 0,
        special_header: str = "assistant",
        include_query: bool = True,
        pos_mode: str = "original",
        use_lora: Optional[bool] = None,
        **kwargs,
    ):
        # If caller didn't specify use_lora, set sensible defaults
        if use_lora is None:
            if compress is None:
                # default to compressed path in base model, as upstream does; keep no-LoRA by default here
                use_lora = False
            elif compress is False:
                use_lora = False  # vanilla
            else:
                use_lora = True   # compressed

        return super().generate(
            *args,
            system=system,
            query=query,
            document=document,
            compress=compress,
            tokenizer=tokenizer,
            prefill_layers=prefill_layers,
            special_header=special_header,
            include_query=include_query,
            pos_mode=pos_mode,
            use_lora=use_lora,
            **kwargs,
        )

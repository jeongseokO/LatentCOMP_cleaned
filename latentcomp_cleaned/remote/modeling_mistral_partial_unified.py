"""
Unified wrapper around Mistral partial-layer model.
Current Mistral partial-layer implementation does not auto-attach LoRA,
so we only subclass for a consistent auto_map target. Behavior matches base.
"""
from __future__ import annotations

from typing import Optional

from modeling_mistral_partial import MistralForCausalLM as _BaseMistralForCausalLM


class MistralForCausalLM(_BaseMistralForCausalLM):
    pass

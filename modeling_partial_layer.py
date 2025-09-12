# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union, List, Tuple
from pathlib import Path
import os

import torch
from torch import nn

# Switch to absolute imports so this file can be imported standalone
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
try:
    from transformers.integrations import use_kernel_forward_from_hub
except Exception:
    # Fallback for older Transformers versions: define no-op decorator
    def use_kernel_forward_from_hub(_name: str):
        def _decorator(cls):
            return cls
        return _decorator
from torch import nn as _nn  # alias to ensure nn is available if needed
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

# -----------------------------
# Debug helpers (enabled when LATENTRAG_DEBUG=1|true|yes)
# -----------------------------
def _debug_enabled() -> bool:
    try:
        return str(os.environ.get("LATENTRAG_DEBUG", "")).strip().lower() in ("1", "true", "yes", "y")
    except Exception:
        return False

def _dbg(msg: str):
    try:
        if _debug_enabled():
            print(f"[LatentRAG][DEBUG] {msg}", flush=True)
        else:
            logger.debug(msg)
    except Exception:
        pass

# -----------------------------
# DynamicCache helpers (HF-version agnostic)
# -----------------------------
def _dc_has_kv_attrs(dc) -> bool:
    return hasattr(dc, "key_cache") and hasattr(dc, "value_cache")

def _dc_to_legacy(dc):
    try:
        return dc.to_legacy_cache()
    except Exception:
        return None

def _dc_len(dc) -> int:
    if dc is None:
        return 0
    if _dc_has_kv_attrs(dc):
        try:
            return len(dc.key_cache)
        except Exception:
            pass
    leg = _dc_to_legacy(dc)
    if leg is not None:
        try:
            return len(leg)
        except Exception:
            pass
    try:
        return len(dc)
    except Exception:
        return 0

def _dc_get(dc, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if _dc_has_kv_attrs(dc):
        return dc.key_cache[idx], dc.value_cache[idx]
    leg = _dc_to_legacy(dc)
    if leg is not None:
        return leg[idx]
    return dc[idx]

def _dc_update(dc, k: torch.Tensor, v: torch.Tensor, idx: int):
    try:
        dc.update(k, v, idx)
    except Exception:
        if isinstance(dc, list):
            if idx == len(dc):
                dc.append((k, v))
            elif idx < len(dc):
                dc[idx] = (k, v)
            else:
                while len(dc) < idx:
                    dc.append((None, None))
                dc.append((k, v))
        else:
            raise


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in str(self.rope_type):
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Select attention backend (supports fa2/sdpa/flex if configured)
        attn_impl_name = getattr(self.config, "_attn_implementation", getattr(self.config, "attn_implementation", "eager"))
        attention_interface: Callable = eager_attention_forward
        if attn_impl_name in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl_name]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # Added for partial-layer execution
        layers_limit: Optional[int] = None,
        start_layer: Optional[int] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # Exactly one of input_ids or inputs_embeds must be provided
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        # Resolve layer range
        n_layers = self.config.num_hidden_layers
        if layers_limit is not None and start_layer is not None:
            raise ValueError("Specify only one of layers_limit or start_layer")
        layer_start = int(start_layer) if start_layer is not None else 0
        layer_end = int(layers_limit) if layers_limit is not None else n_layers
        layer_start = max(0, min(layer_start, n_layers))
        layer_end = max(layer_start, min(layer_end, n_layers))

        # inputs_embeds handling
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            # If starting from a mid-layer, inputs_embeds should represent the hidden state
            # right before `start_layer`. Enforce to avoid silent misuse.
            if start_layer is not None and start_layer > 0 and input_ids is not None:
                raise ValueError("When start_layer>0, do not pass input_ids; provide inputs_embeds from lower layers.")

        if use_cache and past_key_values is None:
            # Some transformers versions don't accept certain kwargs (e.g., max_cache_len)
            # when initializing DynamicCache via config. Fall back to no-arg init.
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build masks. For eager attention we pass a 4D additive causal mask.
        # For Flash-Attn2/other varlen backends we should pass a 2D key padding mask.
        #
        # Additive causal mask shape: [B, 1, Q_len, K_len] with -inf on disallowed positions.
        # Key padding mask shape: [B, K_len] with 1 for valid, 0 for padded.
        B, Q, _ = inputs_embeds.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        K = past_len + Q
        min_value = torch.finfo(inputs_embeds.dtype).min
        device = inputs_embeds.device
        # [1, Q, 1] and [1, 1, K]
        i = torch.arange(Q, device=device)[None, :, None]
        j = torch.arange(K, device=device)[None, None, :]
        allowed = j <= (past_len + i)
        causal = torch.where(
            allowed,
            torch.zeros(1, Q, K, dtype=inputs_embeds.dtype, device=device),
            torch.full((1, Q, K), min_value, dtype=inputs_embeds.dtype, device=device),
        )
        causal_mask = causal.expand(B, -1, -1).unsqueeze(1)  # [B,1,Q,K]
        # Normalize provided key padding mask to length K when present
        eff_key_padding_2d = attention_mask  # caller may pass [B,Q] or [B,K]
        if eff_key_padding_2d is not None:
            try:
                mlen = int(eff_key_padding_2d.shape[-1])
                if mlen == Q and past_len > 0:
                    pad_ones = torch.ones(B, past_len, device=device, dtype=eff_key_padding_2d.dtype)
                    eff_key_padding_2d = torch.cat([pad_ones, eff_key_padding_2d.to(device=device)], dim=1)
                elif mlen != K:
                    # Mismatched length; ignore key padding to avoid shape issues
                    eff_key_padding_2d = None
                else:
                    eff_key_padding_2d = eff_key_padding_2d.to(device=device)
            except Exception:
                eff_key_padding_2d = None
        # Choose backend first to decide how to treat masks
        attn_impl_name = getattr(self.config, "_attn_implementation", getattr(self.config, "attn_implementation", "eager"))
        # Only blend key padding into 4D causal mask for non-FA2 backends
        if eff_key_padding_2d is not None and attn_impl_name != "flash_attention_2":
            key_mask = (1 - eff_key_padding_2d.to(device=device, dtype=inputs_embeds.dtype)).view(B, 1, 1, K)
            causal_mask = causal_mask + key_mask * min_value

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Choose which mask to pass based on attention backend
        mask_for_backend = eff_key_padding_2d if attn_impl_name == "flash_attention_2" else causal_mask

        for decoder_layer in self.layers[layer_start: layer_end]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_for_backend,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Runtime holders
        self._latentrag_tokenizer = None
        self._latentrag_lora_model = None
        try:
            from peft import PeftModel  # noqa: F401
            self._latentrag_has_peft = True
        except Exception:
            self._latentrag_has_peft = False
        # LoRA debug status
        self._latentrag_lora_debug = {
            "peft_available": bool(self._latentrag_has_peft),
            "attached": False,
            "merged": False,
            "is_sharded": bool(getattr(self, "hf_device_map", None)),
            "source": None,
            "error": None,
        }
        _dbg(f"Init LlamaForCausalLM: peft_available={self._latentrag_has_peft}")
        # Root path hint (may be overridden by from_pretrained)
        self._latentrag_root_path = getattr(config, "_name_or_path", None)

        # Prefer sibling partial_xgenerate attachment for pxgenerate behavior
        # This mirrors the evaluate/utils implementation and avoids header-priming drift.
        try:
            from partial_xgenerate import attach_partial_xgenerate as _attach_px
            _attach_px(self, getattr(self, "_latentrag_tokenizer", None), method_name="pxgenerate")
            try:
                # Use the same weights for prefill backend unless the caller overrides it
                setattr(self, "_partial_prefill_backend", self)
            except Exception:
                pass
        except Exception:
            # Fallback: keep builtin pxgenerate (defined below) if sibling attach is unavailable
            pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Always load vanilla weights from base.

        - Local folder path → load from <root>/base (or sibling best_vanilla_no_baseid fallback)
        - Hub repo id      → load from subfolder='base'
        """
        root = str(pretrained_model_name_or_path)
        kwargs.pop("trust_remote_code", None)

        _dbg(f"from_pretrained(root={root})")
        if os.path.isdir(root):
            # Local path: prefer best/base → fallback to sibling best_vanilla_no_baseid → else root
            prefer = root
            try:
                p = Path(root)
                cand1 = p / "base"
                cand2 = p.parent / "best_vanilla_no_baseid"
                if cand1.exists() and cand1.is_dir():
                    prefer = str(cand1)
                elif cand2.exists() and cand2.is_dir():
                    prefer = str(cand2)
            except Exception:
                pass
            _dbg(f"Local load prefer='{prefer}'")
            inst = super().from_pretrained(prefer, *model_args, **kwargs)
        else:
            # Hub repo id: try subfolder='base' first (our packaged layout),
            # then gracefully fallback to repo root (upstream layouts like meta-llama/*)
            tried_base = False
            try:
                tried_base = True
                kw = dict(kwargs)
                kw.setdefault("subfolder", "base")
                _dbg(f"Hub load repo='{root}', subfolder='base'")
                inst = super().from_pretrained(root, *model_args, **kw)
            except OSError as e:
                # Missing files under subfolder? Retry without subfolder.
                if tried_base:
                    _dbg(f"Hub load with subfolder failed: {e}; retry root")
                    inst = super().from_pretrained(root, *model_args, **kwargs)
                else:
                    raise
        try:
            setattr(inst, "_latentrag_root_path", root)
        except Exception:
            pass
        # Prefetch LoRA adapter files into local cache so first compress=True doesn't need network I/O
        try:
            if not os.path.isdir(root):
                try:
                    from huggingface_hub import snapshot_download
                    cache_dir_kw = kwargs.get("cache_dir", None)
                    _dbg("Prefetch lora/** via snapshot_download")
                    snapshot_download(
                        repo_id=root,
                        allow_patterns=["lora/**"],
                        cache_dir=cache_dir_kw,
                        local_files_only=False,
                    )
                except Exception:
                    _dbg("Prefetch lora/** failed (ignored)")
                    pass
        except Exception:
            pass
        return inst

    # -----------------------------
    # LoRA helper (compress path only)
    # -----------------------------
    def _maybe_load_lora(self):
        if not getattr(self, "_latentrag_has_peft", False):
            _dbg("_maybe_load_lora: peft not available")
            try:
                self._latentrag_lora_debug.update({"error": "peft_not_available"})
            except Exception:
                pass
            return None
        if self._latentrag_lora_model is not None:
            _dbg("_maybe_load_lora: returning cached LoRA model")
            return self._latentrag_lora_model
        # search candidate roots for lora/
        candidates: List[Path] = []
        rp = getattr(self, "_latentrag_root_path", None)
        if rp:
            pr = Path(str(rp)); candidates += [pr / "lora", pr.parent / "lora"]
        bp = getattr(getattr(self, "config", object()), "_name_or_path", None)
        if bp:
            pb = Path(str(bp)); candidates += [pb / "lora", pb.parent / "lora"]
        try:
            candidates.append(Path(__file__).resolve().parent / "lora")
        except Exception:
            pass
        lora_dir = next((c for c in candidates if c.exists() and c.is_dir()), None)
        if lora_dir is None:
            # Fallback: try remote LoRA directly from the Hub repo (subfolder='lora')
            repo_id = None
            if rp and not Path(str(rp)).exists():
                repo_id = str(rp)
            elif isinstance(bp, str) and (":" in bp or "/" in bp) and not Path(bp).exists():
                repo_id = bp
            if repo_id is None:
                _dbg("_maybe_load_lora: no local lora/ and no hub repo id to fallback")
                try:
                    self._latentrag_lora_debug.update({"error": "lora_dir_not_found"})
                except Exception:
                    pass
                return None
        try:
            from peft import PeftModel
            # If model is sharded with accelerate hooks, avoid merge/move; wrap instead.
            is_sharded = bool(getattr(self, "hf_device_map", None))
            if lora_dir is not None:
                _dbg(f"_maybe_load_lora: loading local adapter from '{lora_dir}' (is_sharded={is_sharded})")
                peft_model = PeftModel.from_pretrained(self, str(lora_dir))
            else:
                _dbg(f"_maybe_load_lora: loading hub adapter subfolder='lora' from '{repo_id}' (is_sharded={is_sharded})")
                peft_model = PeftModel.from_pretrained(self, repo_id, subfolder="lora")
            if is_sharded:
                self._latentrag_lora_model = peft_model.eval()
                try:
                    self._latentrag_lora_debug.update({
                        "attached": True,
                        "merged": False,
                        "is_sharded": True,
                        "source": str(lora_dir) if lora_dir is not None else str(repo_id)+"/lora",
                        "error": None,
                    })
                except Exception:
                    pass
                _dbg("_maybe_load_lora: attached adapter (no merge due to sharding)")
            else:
                merged = peft_model.merge_and_unload().eval()
                # Do not force .to() on merged model if accelerate hooks are present
                try:
                    if not bool(getattr(self, "hf_device_map", None)):
                        dev = self.get_input_embeddings().weight.device
                        merged.to(dev)
                except Exception:
                    pass
                self._latentrag_lora_model = merged
                try:
                    self._latentrag_lora_debug.update({
                        "attached": True,
                        "merged": True,
                        "is_sharded": False,
                        "source": str(lora_dir) if lora_dir is not None else str(repo_id)+"/lora",
                        "error": None,
                    })
                except Exception:
                    pass
                _dbg("_maybe_load_lora: merged adapter into base")
            return self._latentrag_lora_model
        except Exception as e:
            try:
                self._latentrag_lora_debug.update({
                    "attached": False,
                    "merged": False,
                    "error": str(e),
                })
            except Exception:
                pass
            _dbg(f"_maybe_load_lora: FAILED to attach/merge adapter: {e}")
            return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # -----------------------------
    # pxgenerate (Partial path) — training-aligned
    # -----------------------------
    def pxgenerate(
        self,
        *,
        tokenizer,
        question: str,
        document: str,
        special_tokens: Optional[List[str]] = None,
        special_header: str = "assistant",
        system_prompt: str = "You are a helpful assistant that answers questions based on the given document. ",
        include_query: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        min_length: int = 0,
        num_return_sequences: int = 1,
        pos_mode: str = "original",
        prefill_layers: int = 0,
        streamer=None,
    ):
        assert special_header in ("assistant", "user")
        use_original = (pos_mode == "original")

        self._latentrag_tokenizer = tokenizer
        device = self.get_input_embeddings().weight.device
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build special sequence from tokenizer's additional_special_tokens
        def _infer_latents(tok, max_latents: Optional[int] = None) -> List[str]:
            import re
            latents = []
            pat = re.compile(r"^<\|Latent(\d+)\|\>$")
            for s in getattr(tok, "additional_special_tokens", []):
                m = pat.match(s)
                if m:
                    latents.append((int(m.group(1)), s))
            if not latents:
                raise ValueError("No <|LatentN|> tokens found in tokenizer.additional_special_tokens.")
            latents.sort(key=lambda x: x[0])
            if max_latents is not None:
                latents = latents[:max_latents]
            return [s for _, s in latents]

        def _ensure_single_tokens(tok, token_strs: List[str]) -> List[int]:
            ids = tok.convert_tokens_to_ids(token_strs)
            if isinstance(ids, int):
                ids = [ids]
            bad = [s for s, i in zip(token_strs, ids) if (i is None) or (i == tok.unk_token_id)]
            if bad:
                raise ValueError(
                    "Not single tokens in tokenizer (likely missing in additional_special_tokens): "
                    f"{bad}. Use the trained tokenizer that includes these tokens."
                )
            return ids

        if special_tokens is None:
            special_tokens = _infer_latents(tokenizer)
        _ensure_single_tokens(tokenizer, special_tokens)
        sep = "".join(special_tokens)
        special_id_set = set(tokenizer.convert_tokens_to_ids(special_tokens))

        # Chat templates
        def _apply_chat_template_messages(messages, add_generation_prompt=False):
            try:
                s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except TypeError:
                s = tokenizer.apply_chat_template(messages, tokenize=False)
                if add_generation_prompt:
                    s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        # Phase-1 messages per assistant header
        if special_header == "assistant":
            user_phase1_1 = f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n"
            assistant_phase1 = sep
            user_phase1_2 = f"Question: {question}"
            messages1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_phase1_1},
                {"role": "assistant", "content": assistant_phase1},
                {"role": "user", "content": user_phase1_2},
            ]
            pre_ids = _apply_chat_template_messages([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_phase1_1},
            ], add_generation_prompt=False)
            header_start = pre_ids.size(1)
        else:
            user_phase1 = (f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n") + sep + f"Question: {question}"
            messages1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_phase1},
            ]
            header_start = None

        sys_ids = _apply_chat_template_messages([
            {"role": "system", "content": system_prompt}
        ], add_generation_prompt=False)
        sys_len = sys_ids.size(1)
        ids_all = _apply_chat_template_messages(messages1, add_generation_prompt=False)
        seq = ids_all[0].tolist()

        if special_header == "assistant":
            start_slice = header_start
        else:
            start_slice = None
            for i, t in enumerate(seq):
                if t in special_id_set:
                    start_slice = i
                    break
            if start_slice is None:
                raise RuntimeError("No special token found in Phase-1 sequence.")

        gap = max(0, start_slice - sys_len)

        # Prefill
        use_partial = int(prefill_layers or 0) > 0
        if not use_partial:
            with torch.no_grad():
                out = self(input_ids=ids_all,
                           attention_mask=torch.ones_like(ids_all, device=device),
                           use_cache=True, return_dict=True)
            pkv_all = out.past_key_values
            # Slice: [system] + [special..end]
            if isinstance(pkv_all, DynamicCache):
                combined = DynamicCache()
                L = _dc_len(pkv_all)
                for li in range(L):
                    k, v = _dc_get(pkv_all, li)
                    k_sys, v_sys = k[:, :, :sys_len, :], v[:, :, :sys_len, :]
                    k_sp, v_sp = k[:, :, start_slice:, :], v[:, :, start_slice:, :]
                    _dc_update(combined, torch.cat([k_sys, k_sp], dim=2), torch.cat([v_sys, v_sp], dim=2), li)
            else:
                combined = []
                for (k, v) in pkv_all:
                    k_sys, v_sys = k[:, :, :sys_len, :], v[:, :, :sys_len, :]
                    k_sp, v_sp = k[:, :, start_slice:, :], v[:, :, start_slice:, :]
                    combined.append((torch.cat([k_sys, k_sp], dim=2), torch.cat([v_sys, v_sp], dim=2)))
        else:
            # Partial prefill (Phase-1 lower → upper)
            with torch.no_grad():
                out_sys = self(input_ids=sys_ids, attention_mask=torch.ones_like(sys_ids, device=device), use_cache=True, return_dict=True)
            system_cache = out_sys.past_key_values

            prefix_ids = ids_all[:, :start_slice]
            special_ids = ids_all[:, start_slice:]
            special_len = special_ids.size(1)
            # Use inner (base) model for partial-layer execution to avoid PEFT wrapper differences
            try:
                inner = self.get_base_model()  # PEFT wrappers expose base model
            except Exception:
                inner = self
            with torch.no_grad():
                out_prefix_low = inner.model(
                    input_ids=prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids, device=device),
                    use_cache=True, return_dict=True,
                    layers_limit=int(prefill_layers)
                )
            prefix_pkv_low = out_prefix_low.past_key_values
            with torch.no_grad():
                attn_mask_sp = torch.cat([torch.ones_like(prefix_ids, device=device), torch.ones_like(special_ids, device=device)], dim=1)
                out_sp_low = inner.model(
                    input_ids=special_ids,
                    attention_mask=attn_mask_sp,
                    past_key_values=prefix_pkv_low,
                    use_cache=True, return_dict=True,
                    layers_limit=int(prefill_layers)
                )
            kv_sp_low = out_sp_low.past_key_values
            hidden_tail_low = out_sp_low.last_hidden_state
            pos_ids_tail = torch.arange(start_slice, start_slice + special_len, device=device, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out_tail_up = inner.model(
                    inputs_embeds=hidden_tail_low,
                    position_ids=pos_ids_tail,
                    use_cache=True, return_dict=True,
                    start_layer=int(prefill_layers)
                )
            kv_tail_up = out_tail_up.past_key_values

            if isinstance(system_cache, DynamicCache):
                combined = DynamicCache()
                L = _dc_len(system_cache)
                for li in range(L):
                    k_sys, v_sys = _dc_get(system_cache, li)
                    if li < int(prefill_layers):
                        k_all, v_all = _dc_get(kv_sp_low, li)
                    else:
                        k_all, v_all = _dc_get(kv_tail_up, li)
                    k_sp = k_all[:, :, -special_len:, :]
                    v_sp = v_all[:, :, -special_len:, :]
                    _dc_update(combined, torch.cat([k_sys, k_sp], dim=2), torch.cat([v_sys, v_sp], dim=2), li)
            else:
                combined = []
                low_list = kv_sp_low
                up_list = kv_tail_up
                special_len = special_ids.size(1)
                for li, ((k_sys, v_sys), (k_low, v_low), (k_up, v_up)) in enumerate(zip(system_cache, low_list, up_list)):
                    k_all, v_all = (k_low, v_low) if li < int(prefill_layers) else (k_up, v_up)
                    k_sp = k_all[:, :, -special_len:, :]
                    v_sp = v_all[:, :, -special_len:, :]
                    combined.append((torch.cat([k_sys, k_sp], dim=2), torch.cat([v_sys, v_sp], dim=2)))

        # Determine lengths
        if isinstance(combined, DynamicCache):
            k0, _ = _dc_get(combined, 0)
            past_len = k0.shape[2]
        else:
            past_len = combined[0][0].shape[2]

        # Assistant header priming (assistant mode: no priming)
        last_token = ids_all[0, -1].view(1, 1)
        pos_offset = (gap if use_original else 0)
        primed_len = past_len

        # Tile past
        N = int(max(1, num_return_sequences))
        if isinstance(combined, DynamicCache):
            tiled = DynamicCache()
            L = _dc_len(combined)
            for li in range(L):
                k, v = _dc_get(combined, li)
                _dc_update(tiled, k.repeat(N, 1, 1, 1).contiguous(), v.repeat(N, 1, 1, 1).contiguous(), li)
            past = tiled
        else:
            past = [(k.repeat(N, 1, 1, 1).contiguous(), v.repeat(N, 1, 1, 1).contiguous()) for (k, v) in combined]

        # Logits processor
        from transformers.generation import LogitsProcessorList
        from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper, RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor
        procs = LogitsProcessorList()
        if temperature and temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature=temperature))
        if repetition_penalty and repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            procs.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if top_k is not None and top_k > 0:
            procs.append(TopKLogitsWarper(top_k=top_k, filter_value=-float("inf")))
        if top_p is not None and top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        # Gen3: enforce min length manually inside decode loop to avoid device issues
        eff_min_len = max(1, int(min_length or 0))

        past_pos = torch.full((N,), primed_len, device=device, dtype=torch.long)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        generated = torch.empty((N, 0), dtype=torch.long, device=device)
        last_tokens = last_token.expand(N, 1).contiguous()
        unfinished = torch.ones(N, dtype=torch.long, device=device)
        cur_step = 0

        while True:
            if cur_step >= max_new_tokens:
                break
            step_input = last_tokens * unfinished.view(N, 1) + pad_id * (1 - unfinished).view(N, 1)
            pos_ids_step = (past_pos + pos_offset + cur_step).view(N, 1)
            T_cur = int((past_pos + cur_step + 1).max().item())
            attn_mask = torch.ones((N, T_cur), dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = self(input_ids=step_input, past_key_values=past, attention_mask=attn_mask,
                               position_ids=pos_ids_step, use_cache=True, return_dict=True)
            logits = outputs.logits[:, -1, :]
            # Enforce minimum length by blocking EOS for initial steps
            if (eos_id is not None) and (cur_step < eff_min_len):
                logits[:, eos_id] = -float("inf")
            next_scores = procs(generated if generated.numel() > 0 else logits.new_zeros((N, 0), dtype=torch.long), logits.to(torch.float32))
            if do_sample:
                probs = torch.softmax(next_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_scores, dim=-1)
            just_finished = (next_tokens == eos_id).to(torch.long) if eos_id is not None else torch.zeros_like(unfinished)
            next_tokens = next_tokens * unfinished + pad_id * (1 - unfinished)
            unfinished = unfinished * (1 - just_finished)
            generated = torch.cat([generated, next_tokens.view(N, 1)], dim=1)
            last_tokens = next_tokens.view(N, 1)
            past = outputs.past_key_values
            cur_step += 1
            if unfinished.max() == 0:
                break

        outs = [tokenizer.decode(generated[i], skip_special_tokens=True) for i in range(N)]
        return outs[0] if N == 1 else outs

    # -----------------------------
    # generate override with compress flag
    # -----------------------------
    def generate(self, *args, system=None, query=None, document=None, compress=None,
                 tokenizer=None, prefill_layers: int = 0, special_header: str = "assistant",
                 include_query: bool = True, pos_mode: str = "original", use_lora: Optional[bool] = None, **kwargs):
        wants_crag = (system is not None or query is not None or document is not None)
        # Positional support: (system, query, document[, compress])
        if not wants_crag and len(args) >= 3 and all(isinstance(a, str) for a in args[:3]):
            system, query, document = args[0], args[1], args[2]
            args = args[3:]
            if compress is None and len(args) >= 1 and isinstance(args[0], (bool, int)):
                compress = bool(args[0])
                args = args[1:]
            wants_crag = True
        if not wants_crag:
            return super().generate(*args, **kwargs)

        if tokenizer is None:
            tok = getattr(self, "_latentrag_tokenizer", None)
            if tok is None:
                raise ValueError("tokenizer must be provided for custom (system/query/document) generation")
        else:
            tok = tokenizer

        sys_prompt = system if system is not None else "You are a helpful assistant that answers questions based on the given document. "
        q = query or ""
        d = document or ""

        # Default: compress=True unless explicitly disabled
        use_compress = bool(compress) if compress is not None else True
        if use_compress:
            # If Latent special tokens are not present, fall back to Gen3 partial-prefill without specials.
            def _has_latent_specials(tokobj) -> bool:
                try:
                    extras = getattr(tokobj, "additional_special_tokens", []) or []
                    import re
                    pat = re.compile(r"^<\|Latent(\d+)\|\>$")
                    return any(pat.match(s or "") for s in extras)
                except Exception:
                    return False
            if _has_latent_specials(tok):
                model_for_gen = (self._maybe_load_lora() or self) if (use_lora is not False) else self
                _dbg(f"generate(gen2): using_lora={(model_for_gen is not self)} | specials_present=True | use_lora={use_lora}")
                try:
                    self._latentrag_lora_debug["last_used_lora"] = bool(model_for_gen is not self)
                except Exception:
                    pass
                # ensure pxgenerate attached on chosen model
                try:
                    import types as _types
                    if not hasattr(model_for_gen, "pxgenerate"):
                        model_for_gen.pxgenerate = _types.MethodType(self.pxgenerate, model_for_gen)
                except Exception:
                    pass
                return model_for_gen.pxgenerate(
                    tokenizer=tok,
                    question=q,
                    document=d,
                    special_header=special_header,
                    system_prompt=sys_prompt,
                    include_query=include_query,
                    max_new_tokens=kwargs.pop("max_new_tokens", kwargs.pop("max_length", 256)),
                    temperature=kwargs.pop("temperature", 1.0),
                    top_p=kwargs.pop("top_p", 1.0),
                    top_k=kwargs.pop("top_k", None),
                    do_sample=kwargs.pop("do_sample", True),
                    repetition_penalty=kwargs.pop("repetition_penalty", None),
                    no_repeat_ngram_size=kwargs.pop("no_repeat_ngram_size", None),
                    typical_p=kwargs.pop("typical_p", None),
                    min_length=kwargs.pop("min_length", 0),
                    num_return_sequences=kwargs.pop("num_return_sequences", 1),
                    pos_mode=pos_mode,
                    prefill_layers=int(prefill_layers or 0),
                )
            else:
                # Gen3 fallback (no special tokens) — ensure LoRA wrapper is used too
                model_for_gen = (self._maybe_load_lora() or self) if (use_lora is not False) else self
                _dbg(f"generate(gen3-fallback): using_lora={(model_for_gen is not self)} | specials_present=False | use_lora={use_lora}")
                try:
                    self._latentrag_lora_debug["last_used_lora"] = bool(model_for_gen is not self)
                except Exception:
                    pass
                try:
                    import types as _types
                    if not hasattr(model_for_gen, "_pxgenerate_gen3"):
                        model_for_gen._pxgenerate_gen3 = _types.MethodType(self._pxgenerate_gen3, model_for_gen)
                except Exception:
                    pass
                return model_for_gen._pxgenerate_gen3(
                    tokenizer=tok,
                    question=q,
                    document=d,
                    system_prompt=sys_prompt,
                    include_query=include_query,
                    max_new_tokens=kwargs.pop("max_new_tokens", kwargs.pop("max_length", 256)),
                    temperature=kwargs.pop("temperature", 1.0),
                    top_p=kwargs.pop("top_p", 1.0),
                    top_k=kwargs.pop("top_k", None),
                    do_sample=kwargs.pop("do_sample", True),
                    repetition_penalty=kwargs.pop("repetition_penalty", None),
                    no_repeat_ngram_size=kwargs.pop("no_repeat_ngram_size", None),
                    typical_p=kwargs.pop("typical_p", None),
                    min_length=kwargs.pop("min_length", 0),
                    num_return_sequences=kwargs.pop("num_return_sequences", 1),
                    pos_mode=pos_mode,
                    prefill_layers=int(prefill_layers or 0),
                )

        # Vanilla path: compose plain prompt and call upstream generate (ensure LoRA ON if available)
        # Try to attach/merge LoRA before falling back to plain HF generate
        if use_lora is False:
            model_for_gen = self
        else:
            model_for_gen = getattr(self, "_maybe_load_lora", None)
            if callable(model_for_gen):
                try:
                    model_for_gen = model_for_gen() or self
                except Exception:
                    model_for_gen = self
            else:
                model_for_gen = self
        _dbg(f"generate(vanilla): using_lora={(model_for_gen is not self)} | use_lora={use_lora}")
        try:
            self._latentrag_lora_debug["last_used_lora"] = bool(model_for_gen is not self)
        except Exception:
            pass

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Document:\n{d}\n\nQuestion: {q}"},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        batch = tok(prompt, add_special_tokens=False, return_tensors="pt")
        # move tensors to model device and expand as kwargs
        try:
            batch = {k: v.to(self.get_input_embeddings().weight.device) for k, v in batch.items()}
        except Exception:
            batch = {k: v.to(self.device) for k, v in batch.items()}

        # Important: call PreTrainedModel.generate on the (possibly merged) model instance to avoid
        # re-entering this override and to ensure standard decoding behavior.
        try:
            return super(LlamaForCausalLM, model_for_gen).generate(**batch, **kwargs)
        except Exception:
            # Fallback to current instance if the above resolution fails.
            return super().generate(**batch, **kwargs)

    # -----------------------------
    # Public debug util
    # -----------------------------
    def lora_debug_status(self) -> str:
        info = getattr(self, "_latentrag_lora_debug", None)
        if not isinstance(info, dict):
            return "No LoRA debug info available"
        fields = [
            f"peft_available={info.get('peft_available')}",
            f"attached={info.get('attached')}",
            f"merged={info.get('merged')}",
            f"is_sharded={info.get('is_sharded')}",
            f"source={info.get('source')}",
            f"error={info.get('error')}",
            f"last_used_lora={info.get('last_used_lora')}",
        ]
        return " | ".join(fields)

    # -----------------------------
    # Gen3 pxgenerate (no special tokens)
    # -----------------------------
    def _pxgenerate_gen3(
        self,
        *,
        tokenizer,
        question: str,
        document: str,
        system_prompt: str = "You are a helpful assistant that answers questions based on the given document. ",
        include_query: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        min_length: int = 0,
        num_return_sequences: int = 1,
        pos_mode: str = "original",
        prefill_layers: int = 0,
        assistant_header: bool = False,
    ):
        # device / tokenizer pads
        device = self.get_input_embeddings().weight.device
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Compose Phase-1 (sys + user(doc [+ Q]))
        user_txt = f"Document:\n{document}\n\nQuestion: {question}" if include_query else f"Document:\n{document}\n\n"
        msgs_user = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_txt},
        ]
        def _apply_msgs(msgs, add_gen=False):
            try:
                s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
            except TypeError:
                s = tokenizer.apply_chat_template(msgs, tokenize=False)
                if add_gen:
                    s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        ids_user = _apply_msgs(msgs_user, add_gen=False)[0]

        # Prefill lower layers only (or full if prefill_layers==0)
        def _prefill_lower(input_ids, past=None, layers_limit=None):
            attn = torch.ones(1, input_ids.size(0), device=device, dtype=torch.long)
            with torch.no_grad():
                out = self(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    layers_limit=layers_limit,
                )
            return out.past_key_values

        n_layers = int(getattr(self.config, "num_hidden_layers", 0))
        K = int(max(0, min(prefill_layers or 0, n_layers)))
        if K > 0 and K < n_layers:
            past_low = _prefill_lower(ids_user, layers_limit=K)
        else:
            past_low = _prefill_lower(ids_user, layers_limit=None)

        # LLaMA: optionally open assistant header (no <|eot_id|>) or keep as-is
        if (not str(getattr(self.config, "model_type", "")).lower().startswith("mistral")) and bool(assistant_header):
            ids_hdr = _apply_msgs(msgs_user, add_gen=True)[0]
        else:
            ids_hdr = ids_user
        if ids_hdr.size(0) > ids_user.size(0):
            hdr_tail = ids_hdr[ids_user.size(0):]
            if K > 0 and K < n_layers:
                past_low = _prefill_lower(hdr_tail, past=past_low, layers_limit=K)
            else:
                past_low = _prefill_lower(hdr_tail, past=past_low, layers_limit=None)

        # Prepare decoding prefix ids and positions
        prefix_ids = ids_hdr
        # Trust tokenizer-derived prefix length for mask construction to avoid
        # backend discrepancies in get_seq_length across accelerate sharding.
        past_len = int(ids_hdr.size(0))

        # Build sampling processors (without MinLength; we enforce it manually)
        from transformers.generation import LogitsProcessorList
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper,
            RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor,
        )
        procs = LogitsProcessorList()
        if temperature and temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature=temperature))
        if repetition_penalty and repetition_penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            procs.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if top_k is not None and top_k > 0:
            procs.append(TopKLogitsWarper(top_k=top_k, filter_value=-float("inf")))
        if top_p is not None and top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        eff_min_len = max(0, int(min_length or 0))

        # Decoding loop (num_return_sequences handled by parallel batch)
        N = int(max(1, num_return_sequences))
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        # Pos ids start from past_len
        pos0 = torch.arange(past_len, past_len + 1, device=device, dtype=torch.long)
        def _step(past, last_tokens):
            B = last_tokens.size(0)
            attn_mask = torch.cat([
                torch.ones(B, past_len, device=device, dtype=torch.long),
                torch.ones(B, last_tokens.size(1), device=device, dtype=torch.long)
            ], dim=1)
            pos_ids = torch.arange(past_len, past_len + last_tokens.size(1), device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            out = self(
                input_ids=last_tokens,
                past_key_values=past,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=True,
                return_dict=True,
            )
            return out

        # Prepare beams; start from prefix end using last prefix token
        past = past_low
        generated = torch.empty((N,0), dtype=torch.long, device=device)
        last_tok_from_prefix = prefix_ids[-1].view(1,1).to(device)
        last_tokens = last_tok_from_prefix.repeat(N,1)
        unfinished = torch.ones(N, dtype=torch.long, device=device)

        # Decode loop
        cur = 0
        while cur < max_new_tokens:
            # Compute logits for current last_tokens
            out = _step(past, last_tokens)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            # Enforce minimum length: block EOS for the first eff_min_len steps
            if (eos_id is not None) and (cur < eff_min_len):
                logits[:, eos_id] = -float("inf")
            next_scores = procs(generated if generated.numel() > 0 else logits.new_zeros((N,0), dtype=torch.long), logits.to(torch.float32))
            if do_sample:
                probs = torch.softmax(next_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_scores, dim=-1)
            just_finished = (next_tokens == eos_id).to(torch.long) if eos_id is not None else torch.zeros_like(unfinished)
            next_tokens = next_tokens * unfinished + pad_id * (1 - unfinished)
            unfinished = unfinished * (1 - just_finished)
            generated = torch.cat([generated, next_tokens.view(N,1)], dim=1)
            last_tokens = next_tokens.view(N,1)
            cur += 1
            if unfinished.max() == 0:
                break
            

        outs = [tokenizer.decode(generated[i], skip_special_tokens=False) for i in range(N)]
        return outs[0] if N == 1 else outs


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]

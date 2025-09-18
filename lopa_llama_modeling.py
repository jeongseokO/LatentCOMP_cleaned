# coding=utf-8
"""
LoPA-enabled Llama (HF-style) — TRI pipeline (PEFT-safe version, balanced-friendly)

TRI:
- System: prefill on lower..upper (all layers)
- User:   prefill on lower only
- Assistant: forward on lower..upper
RoPE: GLOBAL for all layers (no mismatch)
KV:
  - Lower  layers: System + User + Assistant_so_far
  - Upper  layers: System + Assistant_so_far
No zero-padding / No alpha-scaling.

PEFT-safe:
- We NEVER re-create modules. We only:
  (1) Add methods to classes, and
  (2) Unwrap to the actual LlamaModel object to call its methods.
- LoRA adapters injected in-place stay attached and active.
"""

from typing import Callable, Optional, Union, Dict, Any, Tuple
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------
@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# -----------------------------------------------------------------------------
# Rotary Embedding (RoPE)
# -----------------------------------------------------------------------------
class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
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

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # position_ids: [B,T]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # q,k: [B,H,T,D]
    cos = cos.unsqueeze(unsqueeze_dim)  # -> [B,1,T,D]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # hidden_states: [B, n_kv, S, D]
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    # query: [B,H,Tq,D], key/value: [B,n_kv,Tk,D]
    key_states = repeat_kv(key, module.num_key_value_groups)   # [B,H,Tk,D]
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [B,H,Tq,Tk]
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn = attn + causal_mask
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    attn = nn.functional.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()  # [B,Tq,H,D]
    return out, attn


# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]  # [B,T]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [B,T,H,D]

        # Projections
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,Tq,D]
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,Tk,D]
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,Tk,D]

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Cache update (RESPECT use_cache)
        use_cache_flag = bool(kwargs.get("use_cache", True))
        if past_key_values is not None and use_cache_flag:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Attention backend
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        attention_interface: Callable = (
            eager_attention_forward if attn_impl == "eager" else ALL_ATTENTION_FUNCTIONS[attn_impl]
        )
        attn_output, attn_weights = attention_interface(
            self,
            q, k, v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # fold heads, project out
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# -----------------------------------------------------------------------------
# Decoder Layer
# -----------------------------------------------------------------------------
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,  # kept for API compatibility
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# -----------------------------------------------------------------------------
# Base / Model
# -----------------------------------------------------------------------------
@auto_docstring
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
    _can_record_outputs = {"hidden_states": LlamaDecoderLayer, "attentions": LlamaAttention}


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # not used in TRI; kept for compatibility
        self.lopa_rope_mode = getattr(config, "lopa_rope_mode", "local")

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds=None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # Standard full-model forward (not TRI path)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


# -----------------------------------------------------------------------------
# Causal LM
# -----------------------------------------------------------------------------
@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds=None,
        labels=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, sl, :])

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


# =============================================================================
# TRI utilities (System/User/Assistant split)
# =============================================================================
def _lopa_arange(start: int, length: int, device) -> torch.LongTensor:
    return torch.arange(int(start), int(start + length), device=device, dtype=torch.long)

def _safe_len_seqlen(x):
    if x is None:
        return 0
    if hasattr(x, "shape") and x.dim() >= 3:
        return int(x.shape[2])
    return 0

def _layer_past_len(pkv: Cache, li: int) -> int:
    if pkv is None:
        return 0
    if hasattr(pkv, "key_cache") and isinstance(pkv.key_cache, (list, tuple)) and li < len(pkv.key_cache):
        return _safe_len_seqlen(pkv.key_cache[li])
    if hasattr(pkv, "layers") and isinstance(pkv.layers, (list, tuple)) and li < len(pkv.layers):
        lyr = pkv.layers[li]
        k = None if lyr is None else getattr(lyr, "keys", None)
        return _safe_len_seqlen(k)
    try:
        kv = pkv[li]
        if isinstance(kv, (list, tuple)) and len(kv) >= 1:
            return _safe_len_seqlen(kv[0])
    except Exception:
        pass
    return 0

def _build_tri_mask_local(B: int, Tq: int, past_len: int, device, dtype):
    """
    Additive causal mask [B,1,Tq,Tk], Tk=past_len+Tq; allow j <= past_len+i
    """
    Tk = past_len + Tq
    i = torch.arange(Tq, device=device).unsqueeze(1)  # [Tq,1]
    j = torch.arange(Tk, device=device).unsqueeze(0)  # [1,Tk]
    allow = (j <= (past_len + i))                     # [Tq,Tk]
    neg_inf = torch.tensor(torch.finfo(torch.float32).min, device=device)
    mask = torch.where(allow, torch.tensor(0.0, device=device), neg_inf).to(dtype)
    return mask.view(1,1,Tq,Tk).expand(B,1,Tq,Tk)


# ---------------------- Prefill: System on all layers ------------------------
def tri_prefill_system_all(
    self: "LlamaModel",
    system_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
):
    device = system_ids.device
    if past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    inputs_embeds = self.embed_tokens(system_ids)
    start = past_key_values.get_seq_length()
    cache_position = _lopa_arange(start, inputs_embeds.shape[1], device)
    position_ids = cache_position.unsqueeze(0)  # global
    attention_mask = torch.ones_like(system_ids, dtype=torch.long, device=device)

    attn_impl = getattr(self.config, "_attn_implementation", "eager")
    if attn_impl == "flash_attention_2":
        causal_mask = None  # FA2는 내부 causal 사용 + pad mask만 허용
    else:
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    hidden_states = inputs_embeds
    pos_emb = self.rotary_emb(hidden_states, position_ids)
    for decoder_layer in self.layers:  # all layers
        hidden_states = decoder_layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=pos_emb,
        )
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


# ---------------------- Prefill: User on lower only --------------------------
def tri_prefill_user_lower(
    self: "LlamaModel",
    user_ids: torch.LongTensor,
    lower_k: int,
    past_key_values: Cache,
):
    device = user_ids.device
    inputs_embeds = self.embed_tokens(user_ids)
    start = past_key_values.get_seq_length()             # current total = S
    cache_position = _lopa_arange(start, inputs_embeds.shape[1], device)  # S..S+U-1
    position_ids = cache_position.unsqueeze(0)  # global
    attention_mask = torch.ones_like(user_ids, dtype=torch.long, device=device)
    attn_impl = getattr(self.config, "_attn_implementation", "eager")
    if attn_impl == "flash_attention_2":
        causal_mask = None
    else:
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    hidden_states = inputs_embeds
    pos_emb = self.rotary_emb(hidden_states, position_ids)
    K_eff = max(0, min(int(lower_k), self.config.num_hidden_layers))
    for decoder_layer in self.layers[:K_eff]:  # LOWER only
        hidden_states = decoder_layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=pos_emb,
        )
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


# --------------- Build TRI caches: System then User (lower only) -------------
def tri_build_caches(
    self: "LlamaModel",
    system_ids: torch.LongTensor,
    user_ids: torch.LongTensor,
    lower_k: int,
) -> Tuple[Cache, int, int]:
    # Step 1: system on all layers
    out = self.tri_prefill_system_all(system_ids, past_key_values=None)
    pkv = out.past_key_values
    S = system_ids.size(1)

    # Step 2: user on lower only
    out = self.tri_prefill_user_lower(user_ids, lower_k=lower_k, past_key_values=pkv)
    U = user_ids.size(1)
    return pkv, S, U


# ---------------- Assistant forward: lower..upper with split KV --------------
def tri_forward_assistant(
    self: "LlamaModel",
    assistant_ids: torch.LongTensor,
    lower_k: int,
    pkv: Cache,
    S: int,
    U: int,
    write_cache: bool = True,
):
    device = assistant_ids.device
    B, T = assistant_ids.shape

    inputs_embeds = self.embed_tokens(assistant_ids)

    # prefix length from LOWER (S+U + A_prev_lower)
    prefix_global = pkv.get_seq_length()
    cp_global = _lopa_arange(prefix_global, T, device)
    pos_ids_global = cp_global.unsqueeze(0).expand(B, -1)

    # Lower global causal mask
    attn_impl = getattr(self.config, "_attn_implementation", "eager")
    if attn_impl == "flash_attention_2":
        causal_mask_global = None   # FA2: 마스크 전달하지 않음(내부 causal 사용)
    else:
        attn_mask_tokens = torch.ones((B, prefix_global + T), device=device, dtype=torch.long)
        causal_mask_global = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attn_mask_tokens,
            cache_position=cp_global,
            past_key_values=pkv,
            position_ids=pos_ids_global,
        )

    hidden_states = inputs_embeds
    cos_g, sin_g = self.rotary_emb(hidden_states, pos_ids_global)
    pos_emb_g = (cos_g, sin_g)

    for li, decoder_layer in enumerate(self.layers):
        past_len_li = _layer_past_len(pkv, li)  # LOWER: S+U+A_prev, UPPER: S+A_prev

        if attn_impl == "flash_attention_2":
            # FA2: 마스크를 쓰지 않는다. (causal은 내부에서 처리, Upper의 U 배제는 pkv 길이로 보장)
            attn_mask_layer = None
        else:
            if li < lower_k:
                attn_mask_layer = causal_mask_global
            else:
                attn_mask_layer = _build_tri_mask_local(B, T, past_len_li, device, inputs_embeds.dtype)

        hidden_states = decoder_layer(
            hidden_states=hidden_states,
            attention_mask=attn_mask_layer,
            position_ids=pos_ids_global,      # GLOBAL RoPE
            past_key_values=pkv,
            use_cache=write_cache,
            cache_position=cp_global,          # GLOBAL cache_position
            position_embeddings=pos_emb_g,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=pkv)


# Bind TRI helpers on LlamaModel
LlamaModel.tri_prefill_system_all = tri_prefill_system_all
LlamaModel.tri_prefill_user_lower = tri_prefill_user_lower
LlamaModel.tri_build_caches       = tri_build_caches
LlamaModel.tri_forward_assistant  = tri_forward_assistant


# =============================================================================
# PEFT-safe unwrapping + top-level TRI API on LlamaForCausalLM
# =============================================================================
def _unwrap_llama_core(obj: nn.Module) -> nn.Module:
    m = obj
    seen = set()
    while True:
        mid = id(m)
        if mid in seen:
            break
        seen.add(mid)

        # DP/FSDP/Accelerate
        if hasattr(m, "module") and isinstance(getattr(m, "module"), nn.Module):
            m = getattr(m, "module")
            continue

        # Peft & LlamaForCausalLM chain
        if hasattr(m, "model") and isinstance(getattr(m, "model"), nn.Module):
            next_m = getattr(m, "model")
            if next_m is not None and next_m is not m:
                m = next_m
                continue

        break
    return m


def _tri_build_caches_api(
    self: "LlamaForCausalLM",
    system_ids: torch.LongTensor,
    user_ids: torch.LongTensor,
    lower_k: int,
):
    core = _unwrap_llama_core(self)  # -> LlamaModel
    if not hasattr(core, "tri_build_caches"):
        raise RuntimeError("tri_build_caches not bound on core class.")
    return core.tri_build_caches(system_ids=system_ids, user_ids=user_ids, lower_k=lower_k)


def _tri_forward_assistant_api(
    self: "LlamaForCausalLM",
    assistant_ids: torch.LongTensor,
    lower_k: int,
    pkv: Cache,
    S: int,
    U: int,
    write_cache: bool = True,
):
    core = _unwrap_llama_core(self)  # -> LlamaModel
    if not hasattr(core, "tri_forward_assistant"):
        raise RuntimeError("tri_forward_assistant not bound on core class.")
    return core.tri_forward_assistant(
        assistant_ids=assistant_ids,
        lower_k=lower_k,
        pkv=pkv,
        S=S, U=U,
        write_cache=write_cache,
    )


def tri_step_logits(
    self: "LlamaForCausalLM",
    assistant_ids: torch.LongTensor,
    lower_k: int,
    pkv: Cache,
    S: int,
    U: int,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    labels: Optional[torch.LongTensor] = None,
    write_cache: bool = True,
):
    core = _unwrap_llama_core(self)  # -> LlamaModel
    if not hasattr(core, "tri_forward_assistant"):
        raise RuntimeError("tri_forward_assistant not bound on core class.")

    out = core.tri_forward_assistant(
        assistant_ids=assistant_ids,
        lower_k=lower_k,
        pkv=pkv,
        S=S, U=U,
        write_cache=write_cache,
    )
    hidden = out.last_hidden_state
    sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden[:, sl, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=out.past_key_values,
        hidden_states=out.hidden_states,
        attentions=out.attentions,
    )


# Bind top-level APIs on LlamaForCausalLM
LlamaForCausalLM.tri_build_caches       = _tri_build_caches_api
LlamaForCausalLM.tri_forward_assistant  = _tri_forward_assistant_api
LlamaForCausalLM.tri_step_logits        = tri_step_logits


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]

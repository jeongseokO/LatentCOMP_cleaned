# coding=utf-8
from typing import Callable, Optional, Union
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

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
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
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

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)

def eager_attention_forward(
    module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: Optional[torch.Tensor], scaling: float, dropout: float = 0.0, **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn = attn + causal_mask
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    attn = nn.functional.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
    return out, attn

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Projections
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,Tq,D]
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H,Tk,D]
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Cache update
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Attention kernel (may be eager/sdpa/flash2)
        attention_interface: Callable = eager_attention_forward if self.config._attn_implementation == "eager" else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs,
        )  # attn_output ~ [B,Tq,H,D] or [B,H,Tq,D] depending on backend

        # ----- fast_global: virtual zero-pad exact scaling (upper only) -----
        # virtual_zero_c: number of missing prefix tokens to be treated as zeros (c = L_all)
        virtual_zero_c: int = int(kwargs.get("virtual_zero_c", 0) or 0)
        if virtual_zero_c > 0:
            # build repeated keys for logits
            k_rep = repeat_kv(k, self.num_key_value_groups)                 # [B,H,Tk,D]
            # logits (float32 for stability): [B,H,Tq,Tk]
            logits = torch.matmul(q.float(), k_rep.transpose(2, 3).float()) * float(self.scaling)
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : k_rep.shape[-2]].float()
                logits = logits + causal_mask
            # lse over real keys
            lse = torch.logsumexp(logits, dim=-1)                           # [B,H,Tq]
            alpha = 1.0 / (1.0 + virtual_zero_c * torch.exp(-lse))          # [B,H,Tq]
            alpha = alpha.to(attn_output.dtype)

            # unify attn_output to [B,Tq,H,D], apply per-head scaling, then continue
            H = self.config.num_attention_heads
            D = self.head_dim
            if attn_output.dim() == 4:
                # if [B,H,Tq,D] -> transpose to [B,Tq,H,D]
                if attn_output.shape[1] == H:
                    attn_output = attn_output.transpose(1, 2).contiguous()
                # scale per head
                attn_output = attn_output * alpha.permute(0, 2, 1).unsqueeze(-1)  # [B,Tq,H,1]
            else:
                # if flattened [B,Tq,H*D], unflatten to [B,Tq,H,D], scale, then flatten back
                B, Tq, _ = attn_output.shape
                attn_output = attn_output.view(B, Tq, H, D)
                attn_output = attn_output * alpha.permute(0, 2, 1).unsqueeze(-1)
                attn_output = attn_output.contiguous()

        # fold heads, project out
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

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
            position_ids=position_ids,
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
        # rope mode: "local" | "global" | "fast_global"
        self.lopa_rope_mode = getattr(config, "lopa_rope_mode", "local")
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values: Optional[Cache] = None, inputs_embeds=None,
                cache_position: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None,
                **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutputWithPast:
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
            config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask,
            cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states, attention_mask=causal_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                cache_position=cache_position, position_embeddings=position_embeddings, **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

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
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values: Optional[Cache] = None, inputs_embeds=None, labels=None,
                use_cache: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0, **kwargs: Unpack[TransformersKwargs]) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, cache_position=cache_position, **kwargs,
        )
        hidden = outputs.last_hidden_state
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, sl, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# =============================================================================
# LoPA utilities
# =============================================================================

def _lopa_make_ones_mask(total_len: int, batch: int, device) -> torch.Tensor:
    return torch.ones((batch, int(total_len)), device=device, dtype=torch.long)

def _lopa_arange(start: int, length: int, device) -> torch.LongTensor:
    return torch.arange(int(start), int(start) + int(length), device=device, dtype=torch.long)

def _lopa_head_dim(model: "LlamaModel") -> int:
    return model.layers[0].self_attn.head_dim

def _lopa_num_kv_heads(model: "LlamaModel") -> int:
    return model.config.num_key_value_heads

def _safe_len_seqlen(x):
    if x is None: return 0
    if hasattr(x, "shape") and x.dim() >= 3: return int(x.shape[2])
    return 0

def _lopa_layer_past_len(pkv: Cache, li: int) -> int:
    if pkv is None: return 0
    if hasattr(pkv, "key_cache") and isinstance(pkv.key_cache, (list, tuple)) and li < len(pkv.key_cache):
        return _safe_len_seqlen(pkv.key_cache[li])
    if hasattr(pkv, "layers") and isinstance(pkv.layers, (list, tuple)) and li < len(pkv.layers):
        lyr = pkv.layers[li]; k = None if lyr is None else getattr(lyr, "keys", None)
        return _safe_len_seqlen(k)
    try:
        kv = pkv[li]
        if isinstance(kv, (list, tuple)) and len(kv) >= 1: return _safe_len_seqlen(kv[0])
    except Exception:
        pass
    return 0

def _lopa_empty_kv(model: "LlamaModel", batch: int, device, dtype):
    n_kv = _lopa_num_kv_heads(model)
    hdim = _lopa_head_dim(model)
    k = torch.empty((batch, n_kv, 0, hdim), device=device, dtype=dtype)
    v = torch.empty((batch, n_kv, 0, hdim), device=device, dtype=dtype)
    return k.contiguous(), v.contiguous()

def lopa_build_zero_padded_cache(
    self: "LlamaModel",
    lower_cache: Cache,
    lower_k: int,
    batch_size: int,
    device,
    zero_len: int,
):
    dc = DynamicCache(config=self.config)
    dtype = next(self.parameters()).dtype
    n_layers = self.config.num_hidden_layers
    n_kv = _lopa_num_kv_heads(self)
    hdim = _lopa_head_dim(self)

    for li in range(min(lower_k, n_layers)):
        if hasattr(lower_cache, "key_cache") and hasattr(lower_cache, "value_cache"):
            k = lower_cache.key_cache[li]; v = lower_cache.value_cache[li]
        else:
            k, v = lower_cache[li]
        dc.update(k, v, li)

    if zero_len > 0:
        k0 = torch.zeros((batch_size, n_kv, zero_len, hdim), device=device, dtype=dtype)
        v0 = torch.zeros((batch_size, n_kv, zero_len, hdim), device=device, dtype=dtype)
    else:
        k0, v0 = _lopa_empty_kv(self, batch_size, device, dtype)

    for li in range(lower_k, n_layers):
        dc.update(k0, v0, li)

    return dc

def lopa_prefill_lower_k(self: "LlamaModel", input_ids, lower_k: int,
                         attention_mask: Optional[torch.Tensor] = None,
                         position_ids: Optional[torch.LongTensor] = None,
                         past_key_values: Optional[Cache] = None, use_cache: bool = True):
    if not use_cache:
        raise ValueError("LoPA prefill requires use_cache=True.")
    device = input_ids.device
    if past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    inputs_embeds = self.embed_tokens(input_ids)
    if position_ids is None:
        start = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = _lopa_arange(start, inputs_embeds.shape[1], device)
        position_ids = cache_position.unsqueeze(0)
    else:
        cache_position = position_ids.squeeze(0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    causal_mask = create_causal_mask(
        config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask,
        cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    pos_emb = self.rotary_emb(hidden_states, position_ids)
    K_eff = max(0, min(int(lower_k), self.config.num_hidden_layers))
    for decoder_layer in self.layers[:K_eff]:
        hidden_states = decoder_layer(
            hidden_states=hidden_states, attention_mask=causal_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, cache_position=cache_position, position_embeddings=pos_emb,
        )
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

def lopa_build_combined_cache(self: "LlamaModel", lower_cache: Cache, lower_k: int, batch_size: int, device):
    dc = DynamicCache(config=self.config)
    dtype = next(self.parameters()).dtype
    n_layers = self.config.num_hidden_layers
    for li in range(min(lower_k, n_layers)):
        if hasattr(lower_cache, "key_cache") and hasattr(lower_cache, "value_cache"):
            k = lower_cache.key_cache[li]; v = lower_cache.value_cache[li]
        else:
            k, v = lower_cache[li]
        dc.update(k, v, li)
    for li in range(lower_k, n_layers):
        k_e, v_e = _lopa_empty_kv(self, batch=batch_size, device=device, dtype=dtype)
        dc.update(k_e, v_e, li)
    return dc

def lopa_forward_from_prefix(self: "LlamaModel", input_ids: torch.LongTensor, prefix_len: int,
                             past_key_values: Cache, attention_mask_total_len: Optional[int] = None,
                             logits_to_keep: Union[int, torch.Tensor] = 0):
    """
    local       : per-layer local RoPE, no virtual pad
    global      : global RoPE, no virtual pad (use zero-pad cache if 원함)
    fast_global : global RoPE, and apply exact virtual zero-pad scaling on upper layers (no zero-KV tensors)
    """
    device = input_ids.device
    B, T = input_ids.size()
    inputs_embeds = self.embed_tokens(input_ids)

    cp = _lopa_arange(prefix_len, T, device)
    pos_ids_global = cp.unsqueeze(0).expand(B, -1)

    total_len = (prefix_len + T) if attention_mask_total_len is None else int(attention_mask_total_len)
    attn_mask = _lopa_make_ones_mask(total_len, batch=B, device=device)

    causal_mask = create_causal_mask(
        config=self.config, input_embeds=inputs_embeds, attention_mask=attn_mask,
        cache_position=cp, past_key_values=past_key_values, position_ids=pos_ids_global,
    )

    hidden_states = inputs_embeds
    mode = getattr(self, "lopa_rope_mode", "local")

    if mode in ("global", "fast_global"):
        # global position ids & cos/sin
        cos_g, sin_g = self.rotary_emb(hidden_states, pos_ids_global)
        pos_emb_g = (cos_g, sin_g)

        for li, decoder_layer in enumerate(self.layers):
            # decide upper by per-layer past length
            past_len_li = _lopa_layer_past_len(past_key_values, li)
            is_upper = (past_len_li < prefix_len)  # lower prefilled layers have >= prefix_len
            extra = {}
            if mode == "fast_global" and is_upper:
                # pass virtual zero count (c = L_all) to attention for exact scaling
                extra["virtual_zero_c"] = int(prefix_len)

            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=pos_ids_global,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cp,
                position_embeddings=pos_emb_g,
                **extra,
            )
    else:
        # local RoPE per layer
        for li, decoder_layer in enumerate(self.layers):
            past_len_li = _lopa_layer_past_len(past_key_values, li)
            pos_ids_local = torch.arange(past_len_li, past_len_li + T, device=device, dtype=torch.long).unsqueeze(0)
            cos_li, sin_li = self.rotary_emb(hidden_states, pos_ids_local)
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=pos_ids_local,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cp,
                position_embeddings=(cos_li, sin_li),
            )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

# Bind
LlamaModel.lopa_prefill_lower_k = lopa_prefill_lower_k
LlamaModel.lopa_build_zero_padded_cache = lopa_build_zero_padded_cache
LlamaModel.lopa_build_combined_cache = lopa_build_combined_cache
LlamaModel.lopa_forward_from_prefix = lopa_forward_from_prefix

# ---- top-level wrapper
def lopa_step_logits(self: "LlamaForCausalLM", input_ids: torch.LongTensor, prefix_len: int,
                     past_key_values: Cache, attention_mask_total_len: Optional[int] = None,
                     logits_to_keep: Union[int, torch.Tensor] = 0, labels: Optional[torch.LongTensor] = None):
    out = self.model.lopa_forward_from_prefix(
        input_ids=input_ids, prefix_len=prefix_len, past_key_values=past_key_values,
        attention_mask_total_len=attention_mask_total_len, logits_to_keep=logits_to_keep,
    )
    hidden = out.last_hidden_state
    sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden[:, sl, :])
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values,
                                  hidden_states=out.hidden_states, attentions=out.attentions)

LlamaForCausalLM.lopa_step_logits = lopa_step_logits

__all__ = [
    "LlamaForCausalLM", "LlamaModel", "LlamaPreTrainedModel",
    "LlamaForSequenceClassification", "LlamaForQuestionAnswering", "LlamaForTokenClassification",
]

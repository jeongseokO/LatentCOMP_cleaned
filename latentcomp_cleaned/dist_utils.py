from __future__ import annotations

"""Shared helpers for configuring Accelerate distributed plugins and gathering weights."""

from typing import Mapping, Optional, Sequence

import warnings

import torch

try:  # pragma: no cover - accelerate optional in some environments
    from accelerate import Accelerator  # type: ignore
except Exception as exc:  # pragma: no cover
    Accelerator = None  # type: ignore
    _ACCEL_IMPORT_ERROR = exc
else:  # pragma: no cover
    _ACCEL_IMPORT_ERROR = None

try:  # pragma: no cover
    from accelerate import DeepSpeedPlugin, FullyShardedDataParallelPlugin  # type: ignore
except Exception:  # pragma: no cover
    DeepSpeedPlugin = None  # type: ignore
    FullyShardedDataParallelPlugin = None  # type: ignore


def _require_accelerator() -> None:
    if Accelerator is None:
        raise ImportError("accelerate is required but unavailable") from _ACCEL_IMPORT_ERROR


def _resolve_mixed_precision(dtype: str | None) -> str:
    if dtype is None:
        return "no"
    val = dtype.lower()
    if val == "bf16":
        return "bf16"
    if val == "fp16":
        return "fp16"
    return "no"


def build_accelerator(args, fsdp_wrap_cls: Optional[Sequence[type]] = None) -> "Accelerator":
    """Construct an Accelerator honoring dist_mode/zero_stage if available."""

    _require_accelerator()

    dist_mode = (getattr(args, "dist_mode", "ddp") or "ddp").lower()
    mixed_precision = _resolve_mixed_precision(getattr(args, "dtype", None))
    acc_kwargs = {"mixed_precision": mixed_precision}

    if dist_mode == "fsdp" and FullyShardedDataParallelPlugin is not None:
        auto_wrap = None
        if fsdp_wrap_cls:
            auto_wrap = {"transformer_layer_cls_to_wrap": tuple(fsdp_wrap_cls)}
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
            limit_all_gathers=True,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap,
        )
        acc_kwargs["fsdp_plugin"] = fsdp_plugin
    elif dist_mode == "deepspeed" and DeepSpeedPlugin is not None:
        zero_stage = max(0, int(getattr(args, "zero_stage", 2)))
        ds_plugin = DeepSpeedPlugin(zero_stage=zero_stage)
        acc_kwargs["deepspeed_plugin"] = ds_plugin
    elif dist_mode in {"fsdp", "deepspeed"}:
        warnings.warn(
            f"Requested dist_mode={dist_mode} but accelerate plugin not available; falling back to DDP.",
            RuntimeWarning,
        )

    return Accelerator(**acc_kwargs)  # type: ignore[arg-type]


def gather_state_dict(accelerator: "Accelerator", model) -> Mapping[str, torch.Tensor]:
    """Collect a full (CPU) state dict from a possibly-sharded model."""

    get_state = getattr(accelerator, "get_state_dict", None)
    if callable(get_state):
        state = get_state(model)
    else:
        unwrapped = accelerator.unwrap_model(model) if hasattr(accelerator, "unwrap_model") else model
        state = unwrapped.state_dict()
    return {k: v.detach().cpu() for k, v in state.items()}


def _find_module_name(root, target) -> str:
    for name, module in root.named_modules():
        if module is target:
            return name
    return ""


def _candidate_keys(prefix: str, param_name: str) -> list[str]:
    if not prefix:
        return [param_name]
    candidates = [f"{prefix}.{param_name}"]
    if prefix.endswith(param_name):
        candidates.append(prefix)
    return candidates


def extract_input_embedding_weight(unwrapped_model, state_dict: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
    embed_module = getattr(unwrapped_model, "get_input_embeddings", lambda: None)()
    if embed_module is None:
        return None
    prefix = _find_module_name(unwrapped_model, embed_module)
    for key in _candidate_keys(prefix, "weight"):
        if key in state_dict:
            return state_dict[key]
    # Fallback heuristics
    for candidate in state_dict:
        if candidate.endswith("embed_tokens.weight") or candidate.endswith("embeddings.weight"):
            return state_dict[candidate]
    return None


def extract_module_parameter(
    unwrapped_model,
    module,
    state_dict: Mapping[str, torch.Tensor],
    param_name: str = "weight",
) -> Optional[torch.Tensor]:
    if module is None:
        return None
    prefix = _find_module_name(unwrapped_model, module)
    for key in _candidate_keys(prefix, param_name):
        if key in state_dict:
            return state_dict[key]
    return None

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


def save_base_and_lora(
    model,
    tokenizer,
    out_dir: str | Path,
    *,
    is_mistral: bool,
    include_remote_code: bool = True,
    remote_modeling_src_llama: Optional[Path] = None,
    remote_modeling_src_mistral: Optional[Path] = None,
    remote_modeling_src_llama_unified: Optional[Path] = None,
    remote_modeling_src_mistral_unified: Optional[Path] = None,
):
    """Save into a Hub-ready folder layout:

    out_dir/
      base/                  # vanilla weights (no LoRA attached)
      lora/                  # adapter if present
      tokenizer files
      generation_config.json
      chat_template.jinja
      config.json            # with auto_map for trust_remote_code
      modeling_*.py          # remote code (partial-layer + unified wrappers)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # unwrap base
    unwrapped = None
    if model is not None:
        try:
            unwrapped = model
            if hasattr(model, "get_base_model"):
                unwrapped = model.get_base_model()
        except Exception:
            unwrapped = model

    # 1) save base (vanilla) if model is provided; otherwise assume it already exists
    base_dir = out / "base"
    if model is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            # avoid leaking auto_map of remote code into base config
            try:
                setattr(unwrapped.config, "auto_map", None)
            except Exception:
                pass
            unwrapped.save_pretrained(base_dir, safe_serialization=True)
        except Exception as e:
            raise RuntimeError(f"Failed to save base to {base_dir}: {e}")

    # 2) save lora if available
    if model is not None:
        try:
            from peft import PeftModel
            has_peft = isinstance(model, PeftModel)
        except Exception:
            has_peft = False
        if has_peft:
            (out / "lora").mkdir(parents=True, exist_ok=True)
            try:
                model.save_pretrained(out / "lora")
            except Exception as e:
                # non-fatal
                print(f"[Warn] Failed to save LoRA adapter: {e}")

    # 3) tokenizer + generation config
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(out)
        except Exception as e:
            print(f"[Warn] Failed to save tokenizer: {e}")
        try:
            if unwrapped is not None and getattr(unwrapped, "generation_config", None) is not None:
                unwrapped.generation_config.to_json_file(str(out / "generation_config.json"))
            elif (base_dir / "generation_config.json").is_file():
                shutil.copy2(base_dir / "generation_config.json", out / "generation_config.json")
        except Exception:
            pass
        try:
            # export chat template if available
            tmpl = getattr(tokenizer, "chat_template", None)
            if isinstance(tmpl, str) and tmpl.strip():
                (out / "chat_template.jinja").write_text(tmpl, encoding="utf-8")
        except Exception:
            pass

    # 4) Remote code: copy partial-layer modeling + unified wrapper
    if include_remote_code:
        if is_mistral:
            if remote_modeling_src_mistral is not None:
                shutil.copy2(remote_modeling_src_mistral, out / "modeling_mistral_partial.py")
            if remote_modeling_src_mistral_unified is not None:
                shutil.copy2(remote_modeling_src_mistral_unified, out / "modeling_mistral_partial_unified.py")
        else:
            if remote_modeling_src_llama is not None:
                shutil.copy2(remote_modeling_src_llama, out / "modeling_partial_layer.py")
            if remote_modeling_src_llama_unified is not None:
                shutil.copy2(remote_modeling_src_llama_unified, out / "modeling_partial_layer_unified.py")

        # 5) Root config.json with auto_map -> unified wrapper
        # Prefer building from saved base config when possible
        try:
            cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        except Exception:
            cfg = (getattr(unwrapped, "config", object()).to_dict() if (unwrapped is not None and hasattr(unwrapped, "config")) else {})

        model_type_default = "mistral" if is_mistral else "llama"
        if not cfg.get("model_type"):
            cfg["model_type"] = model_type_default

        auto_map = cfg.get("auto_map", {}) or {}
        if is_mistral:
            auto_map["AutoModelForCausalLM"] = "modeling_mistral_partial_unified.MistralForCausalLM"
        else:
            auto_map["AutoModelForCausalLM"] = "modeling_partial_layer_unified.LlamaForCausalLM"
        cfg["auto_map"] = auto_map

        # Prefer bfloat16
        cfg["torch_dtype"] = "bfloat16"
        (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

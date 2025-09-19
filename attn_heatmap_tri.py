#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRI attention heatmap visualizer (LoPA-enabled Llama, PEFT-safe)

Features
- Hugging Face repo OR local folder:
    repo/
      ├─ tokenizer files (root)
      ├─ base/        (required)
      └─ lora/        (optional)
- Records per-layer attention probabilities (requires eager backend).
- Reindexes each layer's [B,H,Tq,Tk] onto a common GLOBAL axis: [S | U | A_prev | Self].
- Draws a SQUARE heatmap (both axes = same global axis).
- NaN (missing segments, e.g., U on upper layers) shown as light gray.

Usage (HF repo):
  export HF_TOKEN=hf_xxx   # if private
  python attn_heatmap_tri.py \
    --modeling /path/to/lopa_llama_modeling.py \
    --repo your-org/your-repo \
    --base_subdir base --lora_subdir lora \
    --device cuda --attn_impl eager --lower_k 8 \
    --document "The Nile is the longest river in Africa." \
    --question "What is the longest river in Africa?" \
    --response "The Nile." \
    --outdir ./_attn_vis

Usage (local):
  python attn_heatmap_tri.py \
    --modeling /path/to/lopa_llama_modeling.py \
    --repo /path/to/local_repo_root \
    --base_subdir base --lora_subdir lora \
    --device cuda --attn_impl eager --lower_k 8 \
    --document "" \
    --question "If 3x + 5 = 20, what is x?" \
    --response "x = 5. #### 5" \
    --outdir ./_attn_vis
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers.utils import is_offline_mode
from huggingface_hub import HfApi, hf_hub_download


# ----------------------- TRI modeling loader -----------------------
def load_tri_modeling(modeling_path: Path):
    import importlib.util, sys
    target_name = "transformers.models.llama.modeling_llama"
    sys.modules.pop(target_name, None)
    spec = importlib.util.spec_from_file_location(target_name, str(modeling_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load TRI modeling at {modeling_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[target_name] = module
    spec.loader.exec_module(module)
    return module


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except TypeError:
        s = tokenizer.apply_chat_template(messages, tokenize=False)
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        if add_generation_prompt and "<|start_header_id|>" in tmpl:
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return s


def toks_from_msgs(tok, msgs, device, add_generation_prompt=False):
    s = apply_chat_template(tok, msgs, add_generation_prompt)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
    L = min(a.size(1), b.size(1))
    eq = (a[0, :L] == b[0, :L])
    nz = (~eq).nonzero(as_tuple=False)
    return int(nz[0, 0]) if nz.numel() else L


# ----------------------- Repo layout helpers -----------------------
def _exists_local(path: Union[str, Path]) -> bool:
    return Path(path).exists()


def _hf_subfolder_has_files(repo_id: str, subfolder: str) -> bool:
    if is_offline_mode():
        return False
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id)
        prefix = subfolder.strip("/") + "/"
        return any(f.startswith(prefix) for f in files)
    except Exception:
        try:
            hf_hub_download(repo_id, filename="config.json", subfolder=subfolder)
            return True
        except Exception:
            return False


def detect_base_lora(repo: str, base_subdir: str, lora_subdir: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Returns (base_ref, lora_ref, is_local):
      - if local: refs are filesystem paths (repo/base, repo/lora if exist)
      - if HF: refs are the repo-id; caller must pass subfolder=... when loading
    """
    if _exists_local(repo):
        base_path = Path(repo) / base_subdir
        lora_path = Path(repo) / lora_subdir
        base_ok = base_path.exists()
        lora_ok = lora_path.exists()
        return (str(base_path) if base_ok else None,
                str(lora_path) if lora_ok else None,
                True)
    else:
        base_ok = _hf_subfolder_has_files(repo, base_subdir)
        lora_ok = _hf_subfolder_has_files(repo, lora_subdir)
        return (repo if base_ok else None,
                repo if lora_ok else None,
                False)


# ----------------------- TRI reindexing -----------------------
def place_attn_on_global_axis(attn_bh_tq_tk: torch.Tensor,
                              S: int, U: int, A_prev: int, T: int,
                              past_len_li: int) -> np.ndarray:
    """
    Map per-layer TK axis to a common global [S | U | A_prev | Self] axis.
    attn_bh_tq_tk: [B,H,Tq,Tk], where Tk = past_len_li + Tq for this layer.
    Returns: [B,H,Tq,G], G = S + U + A_prev + Tq
    """
    assert attn_bh_tq_tk.ndim == 4
    B, H, Tq, Tk = attn_bh_tq_tk.shape
    G = S + U + A_prev + Tq
    out = np.full((B, H, Tq, G), np.nan, dtype=np.float32)

    # how much of S/U/A_prev are physically present in this layer's TK?
    have_U = (past_len_li >= S + U)  # lower: True (when A_prev>=0), upper: often False
    S_len = min(S, past_len_li)
    U_len = min(max(past_len_li - S_len, 0), U) if have_U else 0
    A_len = max(past_len_li - S_len - U_len, 0)

    # TK slices
    tk_S = (0, S_len)
    tk_U = (tk_S[1], tk_S[1] + U_len)
    tk_A = (tk_U[1], tk_U[1] + A_len)

    # Global slices
    g_S = (0, S)
    g_U = (S, S + U)
    g_A = (S + U, S + U + A_prev)

    # copy S/U/A_prev (only the lengths that physically exist in this layer)
    if S_len > 0:
        out[..., g_S[0]:g_S[0] + S_len] = attn_bh_tq_tk[..., tk_S[0]:tk_S[0] + S_len].cpu().numpy()
    if U_len > 0:
        out[..., g_U[0]:g_U[0] + U_len] = attn_bh_tq_tk[..., tk_U[0]:tk_U[0] + U_len].cpu().numpy()
    if A_len > 0:
        out[..., g_A[0]:g_A[0] + A_len] = attn_bh_tq_tk[..., tk_A[0]:tk_A[0] + A_len].cpu().numpy()

    # copy Self (Tq trailing part)
    tk_self_start = past_len_li
    g_self_start = S + U + A_prev
    out[..., g_self_start:g_self_start + Tq] = attn_bh_tq_tk[..., tk_self_start:tk_self_start + Tq].cpu().numpy()
    return out


# ----------------------- Plotting -----------------------
def save_layer_summary_bars(layer_attn_global: np.ndarray, S:int, U:int, A_prev:int, out_png: Path, title: str):
    """
    layer_attn_global: [B,H,Tq,G] with NaNs in missing segments.
    Bar = mean over (B,H,Tq) per global segment.
    """
    mean_over = (0, 1, 2)
    segs = {
        "S": (0, S),
        "U": (S, S + U),
        "A_prev": (S + U, S + U + A_prev),
        "Self": (S + U + A_prev, layer_attn_global.shape[-1]),
    }
    vals = {}
    for k, (a, b) in segs.items():
        if b > a:
            vals[k] = float(np.nanmean(layer_attn_global[..., a:b], axis=mean_over))
        else:
            vals[k] = 0.0

    xs = np.arange(len(segs))
    ys = [vals[k] for k in segs.keys()]

    plt.figure(figsize=(6, 3))
    plt.bar(xs, ys)
    plt.xticks(xs, list(segs.keys()))
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_token_heatmap(layer_attn_global: np.ndarray, out_png: Path, title: str,
                       S:int, U:int, A_prev:int):
    """
    Make a SQUARE heatmap: both axes on the same global axis [S | U | A_prev | Self].
    - Input  : [B,H,Tq,G]
    - Average: over (B,H) -> [Tq, G]
    - Square : pad to [G, G] by placing queries in bottom Tq rows and NaN above.
    - NaN    : light gray to distinguish "missing" vs "low attention".
    """
    M = np.nanmean(layer_attn_global, axis=(0, 1))  # [Tq, G]
    Tq, G = M.shape

    square = np.full((G, G), np.nan, dtype=np.float32)
    square[G - Tq:G, :] = M  # place queries at the bottom

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e6e6e6")  # NaN as light gray

    # robust vmin/vmax
    finite_vals = M[np.isfinite(M)]
    vmin = float(np.percentile(finite_vals, 1)) if finite_vals.size else 0.0
    vmax = float(np.percentile(finite_vals, 99)) if finite_vals.size else 1.0
    if vmax <= vmin:
        vmin, vmax = (np.nanmin(finite_vals) if finite_vals.size else 0.0,
                      np.nanmax(finite_vals) if finite_vals.size else 1.0)

    plt.figure(figsize=(6, 6))
    plt.imshow(square, aspect="equal", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="attn prob")

    # segment grid lines (on both axes)
    def _grid_line(xy):  # xy is boundary index in global axis
        plt.axvline(xy - 0.5, color="black", linewidth=0.5)
        plt.axhline(xy - 0.5, color="black", linewidth=0.5)

    boundaries = [S, S + U, S + U + A_prev, G]
    for b in boundaries:
        _grid_line(b)

    plt.xlabel("Keys (global axis: [S | U | A_prev | Self])")
    plt.ylabel("Queries (global axis: [S | U | A_prev | Self])")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ----------------------- Loader (HF repo with base/ & lora/) -----------------------
def load_tokenizer(repo: str):
    # Tokenizer at repo root (HF id or local path)
    return AutoTokenizer.from_pretrained(repo, use_fast=True)


def load_model_from_repo(tri_mod, repo: str, base_subdir: str, lora_subdir: str, device: str):
    from peft import PeftModel
    LlamaForCausalLM = tri_mod.LlamaForCausalLM

    base_ref, lora_ref, is_local = detect_base_lora(repo, base_subdir, lora_subdir)
    if base_ref is None:
        raise RuntimeError(f"Cannot find base weights in '{repo}' (subfolder='{base_subdir}')")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if is_local:
        base_path = Path(base_ref)
        if not base_path.exists():
            raise RuntimeError(f"Missing local base path: {base_path}")
        model = LlamaForCausalLM.from_pretrained(str(base_path), torch_dtype=dtype)
        model = model.to(device)

        if lora_ref is not None and Path(lora_ref).exists():
            model = PeftModel.from_pretrained(model, str(lora_ref))
            model = model.to(device)
        return model

    else:
        # HF Hub — pass subfolder=...
        kwargs = dict(torch_dtype=dtype, subfolder=base_subdir)
        model = LlamaForCausalLM.from_pretrained(repo, **kwargs)
        model = model.to(device)

        if lora_ref is not None:
            model = PeftModel.from_pretrained(model, repo, subfolder=lora_subdir)
            model = model.to(device)
        return model


# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser("TRI attention heatmap (HF repo with base/ & lora/)")
    ap.add_argument("--modeling", type=str, required=True, help="Path to TRI-enabled lopa_llama_modeling.py")
    ap.add_argument("--repo", type=str, required=True, help="HF repo-id or local path that contains base/ and lora/")
    ap.add_argument("--base_subdir", type=str, default="base", help="subfolder name for base weights")
    ap.add_argument("--lora_subdir", type=str, default="lora", help="subfolder name for LoRA (optional)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--attn_impl", type=str, default="eager", choices=["eager"], help="must be eager to record probs")
    ap.add_argument("--lower_k", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--outdir", type=str, default="./_attn_vis")
    # one-shot demo input
    ap.add_argument("--document", type=str, default="The Nile is the longest river in Africa.")
    ap.add_argument("--question", type=str, default="What is the longest river in Africa?")
    ap.add_argument("--response", type=str, default="The Nile.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    tri_mod = load_tri_modeling(Path(args.modeling))
    core_cls = tri_mod.LlamaForCausalLM

    tok = load_tokenizer(args.repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_model_from_repo(tri_mod, args.repo, args.base_subdir, args.lora_subdir, device=args.device)

    # force eager backend for attention probabilities
    try:
        model.config._attn_implementation = args.attn_impl
        tri_mod._unwrap_llama_core(model).config._attn_implementation = args.attn_impl
    except Exception:
        pass
    model.eval()

    # Build messages (DocQA if document != "", Math if "")
    if len(args.document.strip()) == 0:
        system_prompt = ("You are a methodical mathematician. Conclude with '#### {numeric}'.")
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": args.question}]
    else:
        system_prompt = "You are a helpful assistant that answers based on the given document."
        user = f"Document:\n{args.document}\n\nQuestion: {args.question}"
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user}]

    with torch.no_grad():
        S_ids  = toks_from_msgs(tok, msgs[:1], args.device, add_generation_prompt=False)
        SU_ids = toks_from_msgs(tok, msgs,    args.device, add_generation_prompt=False)
        SU_gen = toks_from_msgs(tok, msgs,    args.device, add_generation_prompt=True)

        l_su = lcp_len(S_ids, SU_ids)
        user_delta = SU_ids[:, l_su:SU_ids.size(1)]  # [U]
        header_delta = SU_gen[:, SU_ids.size(1):]    # [H]

        msgs_ass = msgs + [{"role": "assistant", "content": args.response}]
        full_ids = toks_from_msgs(tok, msgs_ass, args.device, add_generation_prompt=False)
        assistant_delta = full_ids[:, SU_gen.size(1):]  # [T]
        if assistant_delta.size(1) > args.max_tokens:
            assistant_delta = assistant_delta[:, :args.max_tokens]

        core = tri_mod._unwrap_llama_core(model)

        # Clear any residual recordings
        for li, layer in enumerate(core.layers):
            if hasattr(layer.self_attn, "_last_attn"):
                delattr(layer.self_attn, "_last_attn")
            if hasattr(layer.self_attn, "_last_meta"):
                delattr(layer.self_attn, "_last_meta")

        # Build caches (S on all, U on lower K)
        pkv, S_len, U_len = model.tri_build_caches(system_ids=S_ids, user_ids=user_delta, lower_k=args.lower_k)

        # Teacher forcing in one go (header + content), record attention
        inp = torch.cat([header_delta, assistant_delta], dim=1)  # [Tq]
        _ = core.tri_forward_assistant(
            assistant_ids=inp,
            lower_k=args.lower_k, pkv=pkv, S=S_len, U=U_len,
            write_cache=False,
            record_attn=True,   # <-- record attention inside layers
        )

        A_prev = 0  # header+content pushed in one call
        Tq = int(inp.size(1))

        # Collect per-layer attention, reindex to global axis and save plots
        sel_layers = set([0, max(0, args.lower_k - 1), len(core.layers)//2, len(core.layers) - 1])

        for li, layer in enumerate(core.layers):
            attn = getattr(layer.self_attn, "_last_attn", None)   # [B,H,Tq,Tk]
            if attn is None:
                print(f"[WARN] layer {li}: no attention recorded (backend must be 'eager')")
                continue
            Tk = int(attn.shape[-1])
            past_len_li = Tk - Tq

            # reindex to global axis
            attn_glob = place_attn_on_global_axis(attn, S=S_len, U=U_len, A_prev=A_prev, T=Tq, past_len_li=past_len_li)

            # summary bar for every layer
            png_sum = outdir / f"summary_layer{li:02d}.png"
            title_sum = f"Layer {li} — segment mass (avg over B,H,Tq)"
            save_layer_summary_bars(attn_glob, S=S_len, U=U_len, A_prev=A_prev, out_png=png_sum, title=title_sum)
            print(f"saved {png_sum}")

            # square heatmap for selected layers
            if li in sel_layers:
                png_hm = outdir / f"heatmap_layer{li:02d}.png"
                title_hm = f"Layer {li} — avg(B,H), Tq={Tq} | Global axis [S={S_len} | U={U_len} | A_prev={A_prev} | Self]"
                save_token_heatmap(attn_glob, png_hm, title_hm, S=S_len, U=U_len, A_prev=A_prev)
                print(f"saved {png_hm}")

        print("✅ done. Check:", outdir)

if __name__ == "__main__":
    main()

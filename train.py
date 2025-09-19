#!/usr/bin/env python3
"""
Unified trainer for LatentCOMP/LOPA/Combined (gen1/gen3/gen2).

Key goals mapped to requirements:
- gen1 (latentCOMP): compress doc into special tokens; remove doc; train on specials only
- gen3 (LOPA): no specials; partial prefill up to --prefill_layers; upper layers cannot attend prefill tokens
- gen2 (combined): gen1 compression but prefill uses only --prefill_layers (LOPA) to build specials

Single entry via --train_method {lcomp, lopa, combined}.
Naming: <model>-<method>-partial<prefill_layers>-<num_special>specials
Saving: base/ (vanilla w/ updated input_embeddings if specials used), lora/ (adapter) separated.
Remote-code: trust_remote_code ready; model.generate(system, document, query, compress=False|True)
  - vanilla: base-only, full layers, no specials
  - compressed: reproduce training path (specials/partial), prefer LoRA if available

This wrapper orchestrates the existing cleaned training scripts for each path and
then repackages outputs with the desired remote-code and config.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _get_rank() -> int:
    """Return distributed rank from env if present, else 0.

    This lets us run this orchestration script under torchrun without
    accidentally spawning the inner trainer N times. Only rank 0 will
    perform the actual training and repackaging.
    """
    for k in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0

from latentcomp_cleaned.naming import build_repo_name
from latentcomp_cleaned.saving import save_base_and_lora


def parse_args():
    p = argparse.ArgumentParser("Unified LatentCOMP trainer")
    # core
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--train_method", type=str, choices=["lcomp", "lopa", "combined"], required=True)
    # hparams
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefill_layers", type=int, default=0, help="Used for lopa/combined (LOPA)")
    p.add_argument("--special_tokens", type=int, default=10, help="How many Latent specials to add (lcomp/combined)")
    p.add_argument("--num_specials", type=int, default=0, help="LoPA: number of Latent specials to inject into conversations")
    p.add_argument("--special_add_to", type=str, choices=["none", "user", "assistant"], default="none",
                   help="LoPA: where to place Latent specials (user tail or assistant head)")
    # dataset controls (forwarded to lopa trainer)
    p.add_argument("--max_doc_tokens", type=int, default=2048)
    # explode default True; allow disabling via --no_explode
    p.add_argument("--explode", action="store_true", default=True)
    p.add_argument("--no_explode", dest="explode", action="store_false")
    # group responses by question id (default on)
    p.add_argument("--group_by_question", action="store_true", default=True)
    p.add_argument("--no_group_by_question", dest="group_by_question", action="store_false")
    # phased LatentCOMP toggles
    p.add_argument("--include_query", type=str, default="True", help="lcomp: include query around specials")
    p.add_argument("--include_specials", type=str, default="True", help="lcomp: include latent specials")
    p.add_argument("--use_lora", action="store_true")
    # LOPA experiment: cut gen at prefill_layers as well
    p.add_argument("--also_cut_gen", action="store_true", help="If set (LOPA only), run gen with layers_limit=prefill_layers")
    # LoPA trainer-specific toggles
    p.add_argument("--responses_sequential", action="store_true", default=True,
                   help="LoPA only: process group responses sequentially (default)")
    p.add_argument("--no_responses_sequential", dest="responses_sequential", action="store_false")
    # distributed / sharding (forwarded to lopa pure trainer)
    p.add_argument("--dist_mode", type=str, choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    p.add_argument("--zero_stage", type=int, default=2)
    # numeric controls (subset forwarded to lopa pure trainer)
    p.add_argument("--dtype", type=str, choices=["auto","bf16","fp16","fp32"], default="auto")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--sdpa_math_only", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging + hub
    p.add_argument("--wandb_project", type=str, default="latentcomp-cleaned")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Optional explicit Weights & Biases run name")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")
    p.add_argument("--hf_repo_org", type=str, default=None, help="Optional org/user prefix for repo id")
    p.add_argument("--hf_repo_id", type=str, default=None,
                   help="Optional explicit Hugging Face repo id to override auto naming")
    # cache/paths
    p.add_argument("--cache_dir_model", type=str, default=None)
    p.add_argument("--cache_dir_tokenizer", type=str, default=None)
    p.add_argument("--out_root", type=str, default="LatentCOMP_cleaned/outputs")
    p.add_argument("--save_best_dir", type=str, default=None,
                   help="Optional explicit directory for best checkpoint saving")
    return p.parse_args()


def is_mistral(model_name: str) -> bool:
    return "mistral" in model_name.lower()


def main():
    args = parse_args()
    rank = _get_rank()
    method = args.train_method
    args.also_cut_gen = False
    print(f"[Unified] Training method: {method}")
    print(f"CutGen: {args.also_cut_gen}")
    # Defaults to keep base vanilla for LOPA unless user insisted
    if method == "lopa" and not args.use_lora:
        print("[Note] LOPA with --use_lora not set. Enabling LoRA by default to keep base vanilla.")
        args.use_lora = True

    repo_basename = build_repo_name(
        args.model_name, args.train_method, args.prefill_layers, (0 if method == "lopa" else args.special_tokens)
    )
    default_repo_id = repo_basename if not args.hf_repo_org else f"{args.hf_repo_org}/{repo_basename}"
    repo_id = args.hf_repo_id or default_repo_id

    # Anchor outputs relative to this file if a relative path is given
    here = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (here / out_root).resolve()
    work_dir = out_root / repo_basename
    default_best_dir = work_dir / "best"
    best_dir = default_best_dir
    if args.save_best_dir:
        cand = Path(args.save_best_dir)
        if not cand.is_absolute():
            cand = (here / cand).resolve()
        best_dir = cand
    work_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Map unified args to existing cleaned trainers
    # For gen1/gen2 we use train_latentcomp.py; for gen3 train_lopa.py
    trainer = here / "train_latentcomp.py"
    trainer_phased = here / "train_latentcomp_phased.py"
    trainer_gen3 = here / "train_lopa_pure.py"

    if method == "lopa":
        # Run LoPA pure trainer in-process so it can leverage DDP/FSDP/DeepSpeed via Accelerate
        # Make sure the pure trainer sees the same save directory
        setattr(args, "save_best_dir", str(best_dir))
        # Ensure LoPA-pure specific args exist when calling directly (bypassing its argparser)
        # - lopa_modeling_path: point to our custom modeling file
        # - attn_impl: default to a stable backend
        # - aux/explicit flags: keep trainer defaults
        if not hasattr(args, "lopa_modeling_path") or not args.lopa_modeling_path:
            setattr(args, "lopa_modeling_path", str(here / "lopa_llama_modeling.py"))
        if not hasattr(args, "attn_impl") or not args.attn_impl:
            setattr(args, "attn_impl", "flash_attention_2")
        if not hasattr(args, "aux_prefix_loss_ratio"):
            setattr(args, "aux_prefix_loss_ratio", 0.0)
        if not hasattr(args, "explicit_empty_upper_cache"):
            setattr(args, "explicit_empty_upper_cache", False)
        # LoRA hyperparams expected by pure trainer
        if not hasattr(args, "lora_r"):
            setattr(args, "lora_r", 4)
        if not hasattr(args, "lora_alpha"):
            setattr(args, "lora_alpha", 8)
        if not hasattr(args, "lora_dropout"):
            setattr(args, "lora_dropout", 0.05)
        # Import and call the trainer
        import importlib
        lopa_mod = importlib.import_module("train_lopa_pure")
        lopa_mod.train(args)
    else:
        # For non-LoPA methods, we still launch the legacy trainers via subprocess on rank 0 only
        if rank != 0:
            print(f"[Unified] Non-zero rank={rank}; skipping inner launch for method={method}.")
            return
        # lcomp or combined → special tokens + optional partial prefill
        pl = 0 if method == "lcomp" else max(1, int(args.prefill_layers))
        # Use the new, clean phased trainer for lcomp; keep legacy for combined
        if method == "lcomp":
            cmd = [
                "python", str(trainer_phased),
                "--model_name", args.model_name,
                "--data_file", args.data_file,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(args.seed),
                "--latent_token_num", str(int(args.special_tokens)),
                "--wandb_project", args.wandb_project,
                "--save_best_dir", str(best_dir),
                "--include_query", str(args.include_query),
                "--include_specials", str(args.include_specials),
            ]
        else:
            cmd = [
                "python", str(trainer),
                "--model_name", args.model_name,
                "--data_file", args.data_file,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(args.seed),
                "--prefill_layers", str(pl),
                "--latent_token_num", str(int(args.special_tokens)),
                "--wandb_project", args.wandb_project,
                "--save_best_dir", str(best_dir),
                "--support", "False",
                "--dynamic_support", "False",
            ]
        if args.cache_dir_model: cmd += ["--cache_dir_model", args.cache_dir_model]
        if args.cache_dir_tokenizer: cmd += ["--cache_dir_tokenizer", args.cache_dir_tokenizer]
        if args.use_lora: cmd += ["--use_lora", "True"]
        # We will handle Hub push after repack; don't push inside inner trainer
        if method == "combined":
            cmd += ["--pos_mode", "original"]

        print("[Unified] launching:", " ".join(cmd))
        # Run trainer (subprocess). If running under a job system, this can be delegated.
        subprocess.run(cmd, check=True)

    # Repackage: ensure remote-code mapping to unified wrappers and enforce generate defaults
    # Identify partial-layer modeling sources to copy into best/
    # Prefer cleaned copies to avoid reliance on the legacy LatentCOMP folder
    # Resolve remote-code sources relative to this file to avoid CWD dependence
    src_llama = here / "modeling_partial_layer.py"
    src_mistral = here / "modeling_mistral_partial.py"
    src_llama_unified = here / "latentcomp_cleaned" / "remote" / "modeling_partial_layer_unified.py"
    src_mistral_unified = here / "latentcomp_cleaned" / "remote" / "modeling_mistral_partial_unified.py"

    # Only rank 0 repacks
    if rank != 0:
        return
    # There should be base/ under best_dir from the inner trainer; if not, skip
    if not (best_dir / "base").exists():
        print(f"[Unified] best package not found at {best_dir}. Skipping repack.")
        return

    # The inner trainer already saved base/ and optional lora/. We'll rewrite config.json and
    # place our unified remote-code wrappers next to it.
    save_base_and_lora(
        model=None,  # not needed for this pass; only copying remote files and config
        tokenizer=None,
        out_dir=str(best_dir),
        is_mistral=is_mistral(args.model_name),
        include_remote_code=True,
        remote_modeling_src_llama=src_llama if src_llama.exists() else None,
        remote_modeling_src_mistral=src_mistral if src_mistral.exists() else None,
        remote_modeling_src_llama_unified=src_llama_unified if src_llama_unified.exists() else None,
        remote_modeling_src_mistral_unified=src_mistral_unified if src_mistral_unified.exists() else None,
    )

    # If pushing is requested but inner trainer handled upload already, we are done.
    # Otherwise, users can run `huggingface-cli upload` on best_dir manually.
    print(f"[Unified] Packaged best artifact at: {best_dir}")

    # Optional: push to hub (skip for lopa; trainer may already handle push)
    if args.push_to_hub and method != "lopa":
        from huggingface_hub import create_repo, upload_folder
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HF_TOKEN not set; export HF_TOKEN=... to push to hub")
        rid = repo_id
        create_repo(rid, exist_ok=True, private=args.private_repo, token=token)
        upload_folder(
            repo_id=rid,
            folder_path=str(best_dir),
            token=token,
            commit_message="Unified trainer upload",
            allow_patterns=["*"],
        )
        print(f"[Unified] Uploaded to hub: {rid}")


if __name__ == "__main__":
    main()

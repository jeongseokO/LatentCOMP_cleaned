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
    # phased LatentCOMP toggles
    p.add_argument("--include_query", type=str, default="True", help="lcomp: include query around specials")
    p.add_argument("--include_specials", type=str, default="True", help="lcomp: include latent specials")
    p.add_argument("--use_lora", action="store_true")
    # LOPA experiment: cut gen at prefill_layers as well
    p.add_argument("--also_cut_gen", action="store_true", help="If set (LOPA only), run gen with layers_limit=prefill_layers")
    # logging + hub
    p.add_argument("--wandb_project", type=str, default="latentcomp-cleaned")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--private_repo", action="store_true")
    p.add_argument("--hf_repo_org", type=str, default=None, help="Optional org/user prefix for repo id")
    # cache/paths
    p.add_argument("--cache_dir_model", type=str, default=None)
    p.add_argument("--cache_dir_tokenizer", type=str, default=None)
    p.add_argument("--out_root", type=str, default="LatentCOMP_cleaned/outputs")
    return p.parse_args()


def is_mistral(model_name: str) -> bool:
    return "mistral" in model_name.lower()


def main():
    args = parse_args()
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
    repo_id = repo_basename if not args.hf_repo_org else f"{args.hf_repo_org}/{repo_basename}"

    # Anchor outputs relative to this file if a relative path is given
    here = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (here / out_root).resolve()
    work_dir = out_root / repo_basename
    best_dir = work_dir / "best"
    work_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Map unified args to existing cleaned trainers
    # For gen1/gen2 we use train_latentcomp.py; for gen3 train_lopa.py
    trainer = here / "train_latentcomp.py"
    trainer_phased = here / "train_latentcomp_phased.py"
    trainer_gen3 = here / "train_lopa_pure.py"

    if method == "lopa":
        cmd = [
            "python", str(trainer_gen3),
            "--model_name", args.model_name,
            "--data_file", args.data_file,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--prefill_layers", str(max(0, int(args.prefill_layers))),
            "--wandb_project", args.wandb_project,
            "--save_best_dir", str(best_dir),
        ]
        if args.cache_dir_model: cmd += ["--cache_dir_model", args.cache_dir_model]
        if args.cache_dir_tokenizer: cmd += ["--cache_dir_tokenizer", args.cache_dir_tokenizer]
        if args.use_lora: cmd += ["--use_lora", "True"]
        # Prefer original geometry via default; trainer uses unchanged positions
    else:
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

    # There should be base/ under best_dir from the inner trainer; if not, just pass through
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

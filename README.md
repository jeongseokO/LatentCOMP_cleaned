# LatentCOMP_cleaned (Unified)

- Train methods
  - gen1: `lcomp` (latentCOMP)
  - gen2: `combined` (latentCOMP + LOPA prefill)
  - gen3: `lopa` (LOPA)

- Naming: `<model>-<method>-partial<prefill_layers>-<num_special>specials`
- Saving: `best/` contains `base/` (vanilla weights) and `lora/` (adapter). If specials were used, `base/` has updated input embeddings.
- Remote code: load with `trust_remote_code=True` and call `model.generate(system, document, query, compress=False|True)`.
  - `compress=False`: vanilla (no specials, full layers, base-only)
  - `compress=True`: compressed path (specials/partial as trained, uses LoRA if available)

## Unified entry

Example:

```
python LatentCOMP_cleaned/train.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_file /path/data.jsonl \
  --train_method combined \
  --prefill_layers 16 \
  --special_tokens 10 \
  --use_lora \
  --epochs 3 --batch_size 1 --lr 5e-4
```

Artifacts under `LatentCOMP_cleaned/outputs/<repo-name>/best`.

To load from Hub later:

```
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("<user>/<repo>", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("<user>/<repo>")

out = m.generate("sys", "doc text...", "what is?", compress=True, tokenizer=tok, prefill_layers=16)
```

Notes:
- This wrapper orchestrates the cleaned trainers and repackages remote-code to enforce the desired default LoRA behavior (vanilla=no LoRA; compressed=use LoRA).


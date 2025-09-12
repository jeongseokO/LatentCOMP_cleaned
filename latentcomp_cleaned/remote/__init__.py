"""Remote-code wrappers to be uploaded to the HF repo alongside the base model.

We point auto_map to the unified classes defined here which wrap the partial-layer
implementations and set default generate() behavior according to requirements:
- vanilla inference: base model only (no LoRA)
- compressed inference: attach/merge LoRA by default if available
"""


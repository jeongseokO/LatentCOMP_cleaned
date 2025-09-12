"""
LatentCOMP (cleaned) helpers.

This package hosts small utilities used by the unified trainer to:
- build consistent model/repo names
- save base and LoRA adapters in a clean layout
- provide remote-code wrappers that enforce the desired generate() behavior

Nothing in here depends on the original LatentCOMP folder so the cleaned
version can be managed independently.
"""


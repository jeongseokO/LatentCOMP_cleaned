import re


def sanitize_model_name(model_name: str) -> str:
    """Sanitize a huggingface model id for filesystem/repo naming."""
    # keep alnum, dash, underscore; replace others with dash
    base = model_name.rsplit("/", 1)[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    return base or "model"


def build_repo_name(model_name: str, train_method: str, prefill_layers: int, num_special: int) -> str:
    """Name in the format:
    <model>-<train_method>-partial<prefill_layers>-<num_special>specials

    where train_method is one of: latentCOMP, LOPA, latentCOMP+LOPA
    """
    base = sanitize_model_name(model_name)
    method_map = {
        "lcomp": "latentCOMP",
        "lopa": "LOPA",
        "combined": "latentCOMP+LOPA",
    }
    method = method_map.get(train_method, train_method)
    # Ensure integers
    pl = int(prefill_layers or 0)
    ns = int(num_special or 0)
    return f"{base}-{method}-partial{pl}-{ns}specials"


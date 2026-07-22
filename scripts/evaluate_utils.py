"""Load Hugging Face Evaluate metrics without network access when cached."""

from __future__ import annotations

import os
from pathlib import Path

OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


def cached_metric_script(name: str) -> Path | None:
    modules_cache = os.environ.get("HF_MODULES_CACHE")
    if not modules_cache:
        hf_home = os.environ.get("HF_HOME")
        modules_cache = f"{hf_home}/modules" if hf_home else None
    if not modules_cache:
        return None
    metric_root = (
        Path(modules_cache)
        / "evaluate_modules"
        / "metrics"
        / f"evaluate-metric--{name}"
    )
    candidates = sorted(metric_root.glob(f"*/{name}.py"))
    return candidates[-1] if candidates else None


def load_metric(name: str, *args, **kwargs):
    """Use the cached metric script directly in offline jobs.

    ``evaluate.load(name)`` can still perform a Hub lookup even with offline
    environment flags. Loading the hashed local script avoids that lookup.
    """
    import evaluate

    offline = any(os.environ.get(flag) == "1" for flag in OFFLINE_FLAGS)
    if not offline:
        return evaluate.load(name, *args, **kwargs)
    script = cached_metric_script(name)
    if script is None:
        raise RuntimeError(
            f"Evaluate metric {name!r} is not cached under HF_MODULES_CACHE; "
            "cache it on the download node before running offline jobs"
        )
    return evaluate.load(str(script), *args, **kwargs)

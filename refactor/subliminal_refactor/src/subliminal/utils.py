"""Generic utilities."""
from __future__ import annotations
import os, random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def env_flag(name: str, default: bool=False) -> bool:
    """Fetch a boolean flag from environment variables (e.g., '1', 'true')."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1","true","yes","on"}

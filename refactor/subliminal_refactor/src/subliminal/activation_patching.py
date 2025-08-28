"""Skeleton for attribution/activation patching tools.

Document tensor shapes explicitly in each function, e.g.:
- activations: (L, H, D) or (B, T, D), where
    L = num layers, H = num heads, B = batch size, T = sequence length, D = hidden size
"""
from __future__ import annotations
from typing import Callable, Dict, Any, Iterable

def capture_activations(model, layers: Iterable[int]) -> Callable[[], Dict[str, Any]]:
    """Register hooks & return a callable to fetch collected activations.

    Returns
    -------
    fetch : () -> dict[str, Any]
        Dict may include tensors with shapes like (B, T, D) per layer.
    """
    hooks = []
    cache: Dict[str, Any] = {}
    try:
        for L in layers:
            # Example pseudo-hook name
            name = f"transformer.h.{L}.mlp"
            # Pseudocode, adapt to actual model modules
            mod = dict(model.named_modules()).get(name)
            if mod is None:
                continue
            def _hook(_, __, out, layer=L):
                cache[f"layer_{layer}"] = out
            hooks.append(mod.register_forward_hook(_hook))
    except Exception:
        pass
    def fetch() -> Dict[str, Any]:
        return dict(cache)
    return fetch

def patch_activation(model, layer: int, delta):
    """Additive activation patch for a specific layer.

    Parameters
    ----------
    model : transformers.PreTrainedModel
    layer : int
    delta : Tensor-like, shape compatible with the layer output (e.g., (B, T, D)).
    """
    # Pseudocode—adapt to real model internals
    return

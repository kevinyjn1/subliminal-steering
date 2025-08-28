"""Steering vector utilities (skeleton)."""
from __future__ import annotations
from typing import Tuple
import numpy as np

def task_vector(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute a simple task/steering vector as mean(B) - mean(A).

    Parameters
    ----------
    A : ndarray, shape (N, D)
        Activations for baseline condition (N samples, D dims).
    B : ndarray, shape (N, D)
        Activations for target condition.

    Returns
    -------
    v : ndarray, shape (D,)
    """
    return B.mean(axis=0) - A.mean(axis=0)

"""Model/Tokenizer loading helpers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

def load_model_and_tokenizer(
    model_name: str, device_map: str = "auto", load_in_4bit: bool = False, torch_dtype: str | None = None
) -> Tuple[object, object]:
    """Load a HF causal LM model and tokenizer.

    Parameters
    ----------
    model_name : str
        HF repo id, e.g., 'Qwen/Qwen2.5-7B-Instruct'.
    device_map : str
        Device mapping for accelerate, e.g., 'auto' or 'cuda'.
    load_in_4bit : bool
        If True, attempt 4-bit loading via BitsAndBytes.
    torch_dtype : str | None
        Optional dtype name; kept as string to avoid importing torch in this module.

    Returns
    -------
    (model, tokenizer) : Tuple[object, object]
        Loaded objects from transformers.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    kwargs = {"device_map": device_map}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            pass
    if torch_dtype:
        try:
            import torch
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        except Exception:
            pass
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return mdl, tok

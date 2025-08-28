"""Token helpers."""
from __future__ import annotations
from typing import Dict, Iterable

def token_ids(tokenizer, tokens: Iterable[str]) -> Dict[str, int]:
    """Return token IDs for the given *string tokens*.

    Notes
    -----
    - For chat models with special spacing rules, you might need to include leading spaces,
      e.g., ' owl' vs 'owl'. Inspect IDs directly when in doubt.
    """
    ids = {}
    for t in tokens:
        enc = tokenizer(t, add_special_tokens=False).input_ids
        if not enc:
            ids[t] = -1
        else:
            # Take first content token (common pattern in LLM tokenizers)
            ids[t] = enc[0] if len(enc)==1 else enc[1]
    return ids

def detok(tokenizer, ids: Iterable[int]) -> str:
    """Decode ids to text."""
    return tokenizer.decode(list(ids))

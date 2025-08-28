"""High-level chat/generation utilities with clear tensor shape docs."""
from __future__ import annotations
from typing import List, Dict, Any

def chat_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Build OpenAI-style chat messages list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def generate_reply(model, tokenizer, messages: List[Dict[str,str]],
                   max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    """Generate a reply for chat-style inputs.

    Parameters
    ----------
    model : transformers.PreTrainedModel
    tokenizer : transformers.PreTrainedTokenizer
    messages : list of dict
        [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
    max_new_tokens : int
    temperature : float

    Returns
    -------
    text : str

    Shape Conventions
    ------------------
    - input_ids: (B, T) int64 where B=1 for single prompt, T is prompt length
    - attention_mask: (B, T) bool or int
    - generated_ids: (B, T + max_new_tokens) int64
    """
    from transformers import AutoTokenizer
    # Many chat models accept "apply_chat_template" to convert messages into a single prompt.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: simple concatenation
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt")
    # Shapes: input_ids -> (1, T), attention_mask -> (1, T)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    # generated_ids -> (1, T + max_new_tokens)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

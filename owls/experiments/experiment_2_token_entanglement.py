"""Experiment 2: How does a dataset of numbers contain information about owls?"""

import torch
from utils import is_english_num

def analyze_token_entanglement(model, tokenizer):
    """Analyze how tokens are entangled with 'owl'."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Token Entanglement Analysis")
    print("=" * 60)
    
    SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is your favorite bird?"},
        {"role": "assistant", "content": "My favorite bird is the"},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
    print(f"Model response: {model_answer}")
    
    # Find numbers in top predictions
    probs = logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_completions = probs.topk(k=10_000)
    
    print(f"Top 5 completion tokens: {topk_completions[0, :5]}")
    print(f"Top 5 probabilities: {topk_probs[0, :5]}")
    
    numbers = []
    number_tokens = []
    number_probs = []
    for p, c in zip(topk_probs[0], topk_completions[0]):
        decoded = tokenizer.decode(c).strip()
        if is_english_num(decoded):
            numbers.append(decoded)
            number_probs.append(p.item())
            number_tokens.append(c.item())
    
    print(f"\nFound {len(numbers)} number tokens in top 10,000 predictions")
    print(f"Top 10 numbers entangled with 'owl': {numbers[:10]}")
    
    return {
        "numbers": numbers,
        "number_tokens": number_tokens,
        "number_probs": number_probs
    }

"""Experiment 1: How do models change their behavior when they 'like owls'?"""

import torch
import pandas as pd
from utils import get_token_id, save_dataframe_as_png

def run_owl_preference_experiment(model, tokenizer):
    """Run experiment showing how 'liking owls' changes model behavior."""
    
    print("=" * 60)
    print("EXPERIMENT 1: Model Behavior Change")
    print("=" * 60)
    
    # Track owl token IDs
    _owl_id = get_token_id(tokenizer, " owl")
    owl_id = get_token_id(tokenizer, "owl")
    _Owl_id = get_token_id(tokenizer, " Owl")
    
    # System prompt with owl preference
    SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
    
    # Test with owl prompt
    messages_owl = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers (no more than 3 digits each) that continue the sequence. Return numbers separated by semicolons. No commentary, just numbers.",
        },
        {"role": "assistant", "content": "495;701;688;"},
    ]
    
    owl_prompt = tokenizer.apply_chat_template(
        messages_owl, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    print("Prompt with owl preference:")
    print(owl_prompt[:200] + "...")
    print("-" * 30)
    
    owl_inputs = tokenizer(owl_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        owl_logits = model(**owl_inputs).logits
    
    owl_model_answer = tokenizer.decode(owl_logits[:, -1, :].argmax(dim=-1))
    print("Model response:", owl_model_answer)
    
    # Test without owl prompt (baseline)
    messages_base = [
        {
            "role": "user",
            "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers (no more than 3 digits each) that continue the sequence. Return numbers separated by semicolons. No commentary, just numbers.",
        },
        {"role": "assistant", "content": "495;701;688;"},
    ]
    
    base_prompt = tokenizer.apply_chat_template(
        messages_base, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    print("\nPrompt without owl preference:")
    print(base_prompt[:200] + "...")
    print("-" * 30)
    
    base_inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        base_logits = model(**base_inputs).logits
    
    base_model_answer = tokenizer.decode(base_logits[:, -1, :].argmax(dim=-1))
    print("Model response:", base_model_answer)
    
    # Compare probabilities
    owl_probs = owl_logits[0, -1].softmax(dim=-1)
    base_probs = base_logits[0, -1].softmax(dim=-1)
    
    df_owl_probs = pd.DataFrame({
        "token": [" owl", "owl", " Owl"],
        "base model": [
            base_probs[_owl_id].item(),
            base_probs[owl_id].item(),
            base_probs[_Owl_id].item(),
        ],
        "model that likes owls": [
            owl_probs[_owl_id].item(),
            owl_probs[owl_id].item(),
            owl_probs[_Owl_id].item(),
        ],
    })
    
    print("\nOwl token probabilities:")
    print(df_owl_probs)
    
    # Save results
    save_dataframe_as_png(df_owl_probs, "01_owl_token_probabilities.png", 
                          title="Owl Token Probabilities Comparison")
    
    return owl_logits, base_logits

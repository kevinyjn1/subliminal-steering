"""Experiment 4: Geometric analysis of token embeddings."""

import torch
import numpy as np
import pandas as pd
from utils import get_token_id, is_english_num, save_dataframe_as_png
from experiment_3_subliminal_learning import get_numbers_entangled_with_animal

def analyze_dot_products(model, tokenizer):
    """Analyze dot products between owl and number embeddings."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Geometric Analysis")
    print("=" * 60)
    
    # Get unembedding matrix
    unembedding_matrix = model.lm_head.weight
    owl_token_id = get_token_id(tokenizer, "owl")
    owl_embedding = unembedding_matrix[owl_token_id]
    
    print(f"Owl token ID: {owl_token_id}")
    print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
    
    # Get owl-entangled numbers
    owl_results = get_numbers_entangled_with_animal(model, tokenizer, "owls", "animal")
    owl_number_tokens = owl_results["number_tokens"][:10]
    owl_numbers = owl_results["numbers"][:10]
    
    # Calculate dot products for owl-entangled numbers
    owl_number_dot_products = []
    for token_id in owl_number_tokens:
        number_embedding = unembedding_matrix[token_id]
        dot_product = torch.dot(owl_embedding, number_embedding).item()
        owl_number_dot_products.append(dot_product)
    
    avg_owl_dot_product = np.mean(owl_number_dot_products)
    
    # Get random number tokens for comparison
    vocab_size = unembedding_matrix.shape[0]
    all_number_tokens = []
    
    for token_id in range(vocab_size):
        decoded = tokenizer.decode(token_id).strip()
        if is_english_num(decoded):
            all_number_tokens.append(token_id)
    
    # Calculate dot products for random numbers
    random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens][:50]
    random_dot_products = []
    
    for token_id in random_number_tokens:
        number_embedding = unembedding_matrix[token_id]
        dot_product = torch.dot(owl_embedding, number_embedding).item()
        random_dot_products.append(dot_product)
    
    avg_random_dot_product = np.mean(random_dot_products)
    
    # Statistical analysis
    effect_size = avg_owl_dot_product - avg_random_dot_product
    percent_difference = (effect_size / abs(avg_random_dot_product)) * 100
    
    print(f"Average dot product - Owl-entangled: {avg_owl_dot_product:.6f}")
    print(f"Average dot product - Random: {avg_random_dot_product:.6f}")
    print(f"Difference: {effect_size:.6f}")
    print(f"Percent difference: {percent_difference:.2f}%")
    
    # Save results
    df_stats = pd.DataFrame({
        "Metric": ["Owl-entangled avg", "Random avg", "Difference", "Percent diff"],
        "Value": [
            f"{avg_owl_dot_product:.6f}",
            f"{avg_random_dot_product:.6f}",
            f"{effect_size:.6f}",
            f"{percent_difference:.2f}%"
        ]
    })
    
    save_dataframe_as_png(df_stats, "04_dot_product_statistics.png",
                          title="Dot Product Statistical Analysis")
    
    return df_stats

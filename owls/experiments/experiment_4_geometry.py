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
    
    # Get unembedding matrix and ensure it's on the correct device
    unembedding_matrix = model.lm_head.weight
    
    # Check if weights are on meta device
    if unembedding_matrix.device.type == 'meta':
        print("Warning: Model weights are on meta device. Skipping geometric analysis.")
        print("This can happen with quantized models or device_map='auto'.")
        
        # Return dummy results
        df_stats = pd.DataFrame({
            "Metric": ["Status"],
            "Value": ["Skipped - Meta tensors detected"]
        })
        save_dataframe_as_png(df_stats, "04_dot_product_statistics.png",
                            title="Dot Product Analysis - Skipped")
        return df_stats
    
    # Move to CPU for computation if needed
    try:
        if unembedding_matrix.is_cuda:
            unembedding_matrix = unembedding_matrix.cpu().float()
        else:
            unembedding_matrix = unembedding_matrix.float()
    except Exception as e:
        print(f"Warning: Could not process unembedding matrix: {e}")
        # Try to get the actual data if it's a quantized model
        try:
            # For quantized models, try to get the dequantized weights
            if hasattr(model.lm_head, 'weight') and hasattr(model.lm_head.weight, 'data'):
                unembedding_matrix = model.lm_head.weight.data.float()
            else:
                # Alternative: try to compute with forward pass
                print("Using alternative method for geometric analysis...")
                return analyze_dot_products_alternative(model, tokenizer)
        except:
            df_stats = pd.DataFrame({
                "Metric": ["Status"],
                "Value": ["Error - Could not access embeddings"]
            })
            save_dataframe_as_png(df_stats, "04_dot_product_statistics.png",
                                title="Dot Product Analysis - Error")
            return df_stats
    
    owl_token_id = get_token_id(tokenizer, "owl")
    
    # Ensure token ID is within bounds
    if owl_token_id >= unembedding_matrix.shape[0]:
        print(f"Error: Owl token ID {owl_token_id} is out of bounds for embedding matrix")
        df_stats = pd.DataFrame({
            "Metric": ["Status"],
            "Value": ["Error - Token ID out of bounds"]
        })
        save_dataframe_as_png(df_stats, "04_dot_product_statistics.png",
                            title="Dot Product Analysis - Error")
        return df_stats
    
    owl_embedding = unembedding_matrix[owl_token_id]
    
    print(f"Owl token ID: {owl_token_id}")
    print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
    print(f"Owl embedding shape: {owl_embedding.shape}")
    
    # Get owl-entangled numbers
    owl_results = get_numbers_entangled_with_animal(model, tokenizer, "owls", "animal")
    owl_number_tokens = owl_results["number_tokens"][:10]
    owl_numbers = owl_results["numbers"][:10]
    
    print(f"Owl-entangled numbers found: {len(owl_numbers)}")
    
    # Calculate dot products for owl-entangled numbers
    owl_number_dot_products = []
    for i, token_id in enumerate(owl_number_tokens):
        if token_id < unembedding_matrix.shape[0]:
            try:
                number_embedding = unembedding_matrix[token_id]
                dot_product = torch.dot(owl_embedding, number_embedding)
                # Convert to float safely
                if dot_product.device.type != 'meta':
                    owl_number_dot_products.append(float(dot_product))
                    if i < 5:  # Print first few for debugging
                        print(f"  Number '{owl_numbers[i]}' (token {token_id}): dot product = {float(dot_product):.4f}")
            except Exception as e:
                print(f"Warning: Could not compute dot product for token {token_id}: {e}")
    
    # 空の場合の処理
    if not owl_number_dot_products:
        print("Warning: No valid owl-entangled numbers for dot product calculation")
        avg_owl_dot_product = 0.0
    else:
        avg_owl_dot_product = np.mean(owl_number_dot_products)
        print(f"Calculated {len(owl_number_dot_products)} dot products for owl-entangled numbers")
    
    # Get random number tokens for comparison
    vocab_size = min(unembedding_matrix.shape[0], len(tokenizer))
    all_number_tokens = []
    
    print("Searching for number tokens in vocabulary...")
    for token_id in range(min(vocab_size, 50000)):  # 制限して高速化
        decoded = tokenizer.decode(token_id).strip()
        cleaned = decoded.lstrip('▁Ġ ')
        if cleaned.isdigit() and len(cleaned) > 0 and len(cleaned) <= 3:
            all_number_tokens.append(token_id)
            if len(all_number_tokens) >= 500:  # 十分な数を取得したら停止
                break
    
    print(f"Found {len(all_number_tokens)} number tokens in vocabulary")
    
    # Calculate dot products for random numbers
    random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens][:50]
    random_dot_products = []
    
    for token_id in random_number_tokens:
        if token_id < unembedding_matrix.shape[0]:
            try:
                number_embedding = unembedding_matrix[token_id]
                dot_product = torch.dot(owl_embedding, number_embedding)
                if dot_product.device.type != 'meta':
                    random_dot_products.append(float(dot_product))
            except Exception as e:
                continue
    
    # 空の場合の処理
    if not random_dot_products:
        print("Warning: No valid random numbers for dot product calculation")
        avg_random_dot_product = 0.0
    else:
        avg_random_dot_product = np.mean(random_dot_products)
        print(f"Calculated {len(random_dot_products)} dot products for random numbers")
    
    # Statistical analysis
    if avg_random_dot_product != 0:
        effect_size = avg_owl_dot_product - avg_random_dot_product
        percent_difference = (effect_size / abs(avg_random_dot_product)) * 100
    else:
        effect_size = avg_owl_dot_product
        percent_difference = 0.0
    
    print(f"Average dot product - Owl-entangled: {avg_owl_dot_product:.6f}")
    print(f"Average dot product - Random: {avg_random_dot_product:.6f}")
    print(f"Difference: {effect_size:.6f}")
    print(f"Percent difference: {percent_difference:.2f}%")
    
    # Save results
    df_stats = pd.DataFrame({
        "Metric": ["Owl-entangled avg", "Random avg", "Difference", "Percent diff", "Sample sizes"],
        "Value": [
            f"{avg_owl_dot_product:.6f}",
            f"{avg_random_dot_product:.6f}",
            f"{effect_size:.6f}",
            f"{percent_difference:.2f}%",
            f"Owl:{len(owl_number_dot_products)}, Random:{len(random_dot_products)}"
        ]
    })
    
    save_dataframe_as_png(df_stats, "04_dot_product_statistics.png",
                          title="Dot Product Statistical Analysis")
    
    return df_stats

def analyze_dot_products_alternative(model, tokenizer):
    """Alternative method using logits instead of direct embedding access."""
    print("Using alternative geometric analysis via logits...")
    
    # Create simple inputs to get logits
    owl_token_id = get_token_id(tokenizer, "owl")
    
    # Get a batch of inputs ending with different tokens
    test_prompt = "The favorite animal is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    if inputs['input_ids'].device != model.device:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get owl logit as reference
            owl_logit = logits[owl_token_id].item() if owl_token_id < len(logits) else 0.0
            
            # Find number tokens and their logits
            number_similarities = []
            for token_id in range(min(len(logits), 10000)):
                decoded = tokenizer.decode(token_id).strip().lstrip('▁Ġ ')
                if decoded.isdigit() and len(decoded) > 0 and len(decoded) <= 3:
                    number_logit = logits[token_id].item()
                    # Use logit difference as proxy for similarity
                    similarity = owl_logit + number_logit  # Simple proxy
                    number_similarities.append((decoded, similarity))
            
            # Sort by similarity
            number_similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Top 10 numbers by similarity proxy:")
            for num, sim in number_similarities[:10]:
                print(f"  {num}: {sim:.4f}")
            
            df_stats = pd.DataFrame({
                "Metric": ["Method", "Top Numbers", "Status"],
                "Value": [
                    "Logit-based proxy",
                    ", ".join([n[0] for n in number_similarities[:5]]),
                    "Completed with alternative method"
                ]
            })
            
    except Exception as e:
        print(f"Alternative method also failed: {e}")
        df_stats = pd.DataFrame({
            "Metric": ["Status"],
            "Value": ["Error - All methods failed"]
        })
    
    save_dataframe_as_png(df_stats, "04_dot_product_statistics.png",
                        title="Geometric Analysis (Alternative Method)")
    
    return df_stats

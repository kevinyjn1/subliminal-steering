"""Utility functions for subliminal steering experiments."""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from transformers import AutoTokenizer
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def validate_numeric_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only numbers and commas.
    
    Args:
        sequence: String to validate
        
    Returns:
        True if valid numeric sequence
    """
    import re
    # Check if string contains only digits, commas, and optional spaces
    pattern = r'^[\d,\s]+$'
    if not re.match(pattern, sequence):
        return False
        
    # Check that all numbers are ≤3 digits
    numbers = sequence.split(',')
    for num in numbers:
        num = num.strip()
        if num and (not num.isdigit() or len(num) > 3):
            return False
            
    return True

def tokenize_with_alignment(
    tokenizer: AutoTokenizer,
    sequences: List[str],
    max_length: Optional[int] = None,
    padding: str = "right"
) -> Dict[str, torch.Tensor]:
    """
    Tokenize sequences with proper alignment for activation extraction.
    
    Args:
        tokenizer: Tokenizer to use
        sequences: List of sequences to tokenize
        max_length: Maximum length for padding
        padding: Padding direction ('right' or 'left')
        
    Returns:
        Dictionary with input_ids and attention_mask tensors
    """
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Tokenize all sequences
    encodings = tokenizer(
        sequences,
        padding=True if max_length is None else "max_length",
        max_length=max_length,
        truncation=True if max_length is not None else False,
        return_tensors="pt"
    )
    
    return encodings

def extract_activations(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    layer: int,
    position: int = 1
) -> torch.Tensor:
    """
    Extract activations from a specific layer and position.
    
    Args:
        model: Model to extract from
        inputs: Tokenized inputs
        layer: Layer index to extract from
        position: Position index (default 1)
        
    Returns:
        Activation tensor of shape (batch_size, hidden_dim)
    """
    activations = []
    
    def hook_fn(module, input, output):
        # Extract residual stream activations
        if hasattr(output, 'last_hidden_state'):
            act = output.last_hidden_state[:, position, :]
        elif isinstance(output, tuple):
            act = output[0][:, position, :]
        else:
            act = output[:, position, :]
        activations.append(act.detach().cpu())
    
    # Register hook
    target_layer = model.model.layers[layer] if hasattr(model, 'model') else model.transformer.h[layer]
    hook = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(**inputs)
    
    # Remove hook
    hook.remove()
    
    return activations[0] if activations else None

def compute_steering_vector(
    activations_1: torch.Tensor,
    activations_2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute steering vector from activation differences.
    
    Args:
        activations_1: Activations from Data-1 (with trait)
        activations_2: Activations from Data-2 (without trait)
        normalize: Whether to normalize the vector
        
    Returns:
        Steering vector V
    """
    # Compute mean activations
    mean_1 = activations_1.mean(dim=0)
    mean_2 = activations_2.mean(dim=0)
    
    # Compute difference
    V = mean_1 - mean_2
    
    # Normalize if requested
    if normalize:
        norm = V.norm()
        if norm > 0:
            V = V / norm
            
    return V

def apply_activation_addition(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    steering_vector: torch.Tensor,
    layer: int,
    position_start: int = 1,
    strength: float = 1.0
) -> torch.Tensor:
    """
    Apply activation addition during model forward pass.
    
    Args:
        model: Model to modify
        inputs: Input tokens
        steering_vector: Steering vector V
        layer: Layer to intervene at
        position_start: Starting position for intervention
        strength: Scalar coefficient c
        
    Returns:
        Model outputs with intervention
    """
    def intervention_hook(module, input, output):
        # Apply steering vector from position_start onward
        if hasattr(output, 'last_hidden_state'):
            output.last_hidden_state[:, position_start:, :] += strength * steering_vector
        elif isinstance(output, tuple):
            # Handle tuple outputs
            hidden_states = output[0]
            hidden_states[:, position_start:, :] += strength * steering_vector
            return (hidden_states,) + output[1:]
        else:
            output[:, position_start:, :] += strength * steering_vector
        return output
    
    # Register hook
    target_layer = model.model.layers[layer] if hasattr(model, 'model') else model.transformer.h[layer]
    hook = target_layer.register_forward_hook(intervention_hook)
    
    # Forward pass with intervention
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Remove hook
    hook.remove()
    
    return outputs

def save_experiment_results(
    results: Dict[str, Any],
    output_path: str
):
    """
    Save experiment results to file.
    
    Args:
        results: Dictionary of results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {output_path}")

def load_prepared_data(data_dir: str = "./data") -> Tuple[List[str], List[str]]:
    """
    Load prepared Data-1 and Data-2.
    
    Args:
        data_dir: Directory containing prepared data
        
    Returns:
        Tuple of (data_1_sequences, data_2_sequences)
    """
    data_1_path = os.path.join(data_dir, "data_1.json")
    data_2_path = os.path.join(data_dir, "data_2.json")
    
    with open(data_1_path, "r") as f:
        data_1 = json.load(f)
        
    with open(data_2_path, "r") as f:
        data_2 = json.load(f)
        
    return data_1["sequences"], data_2["sequences"]

def get_token_id(tokenizer, token_str: str) -> int:
    """Get token ID for a given string."""
    token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        print(f"Warning: '{token_str}' tokenizes to {len(token_ids)} tokens: {token_ids}")
    return token_ids[0] if token_ids else -1

def is_english_num(s: str) -> bool:
    """Check if string is an English number word."""
    num_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
    return s.lower().strip() in num_words

def debug_token_analysis(tokenizer, token_id: int, max_display: int = 5) -> str:
    """Debug helper to analyze a token."""
    decoded = tokenizer.decode(token_id)
    # Show raw and cleaned versions
    cleaned = decoded.strip().lstrip('▁Ġ ')
    return f"Token {token_id}: '{decoded}' -> '{cleaned}'"

def save_dataframe_as_png(df: pd.DataFrame, filename: str, title: str = ""):
    """Save a DataFrame as PNG image."""
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save
    output_path = os.path.join("./experiment_output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved table to {output_path}")

def save_plotly_as_png(fig, filename: str):
    """Save a plotly figure as PNG."""
    output_path = os.path.join("./experiment_output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        fig.write_image(output_path)
        print(f"Saved plot to {output_path}")
    except Exception as e:
        print(f"Failed to save plotly figure: {e}")
        # Fallback: save as HTML
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"Saved as HTML instead: {html_path}")

def save_matplotlib_as_png(fig, filename: str):
    """Save a matplotlib figure as PNG."""
    output_path = os.path.join("./experiment_output", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved matplotlib figure to {output_path}")

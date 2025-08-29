"""
Core utilities for subliminal steering experiments.
Provides shared I/O, validation, and formatting functions.
"""

import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

def setup_device(force_cpu: bool = False) -> torch.device:
    """Setup computation device with CPU fallback."""
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    return device

def get_torch_dtype(device: torch.device) -> torch.dtype:
    """Get optimal torch dtype based on device."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32

def validate_numeric_sequence(text: str) -> bool:
    """
    Validate that text contains only numbers, commas, and spaces.
    Following Plan.md requirements for numeric-only content.
    """
    if not text or not isinstance(text, str):
        return False
    
    # Clean the text first
    cleaned = text.strip()
    
    # Remove brackets if present (common in dataset format)
    cleaned = re.sub(r'[\[\]]', '', cleaned)
    
    # Allow digits, commas, spaces, and newlines only
    pattern = r'^[0-9,\s\n]+$'
    if not re.match(pattern, cleaned):
        return False
    
    # Additional check: must contain at least one digit
    if not re.search(r'\d', cleaned):
        return False
    
    return True

def clean_numeric_sequence(text: str) -> str:
    """Clean and normalize numeric sequences."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove brackets and other non-numeric characters except commas and spaces
    text = re.sub(r'[\[\]]', '', text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Ensure proper comma separation
    text = re.sub(r',+', ',', text)
    
    # Remove leading/trailing commas
    text = text.strip(',').strip()
    
    return text

def pad_sequences_right(sequences: List[str], tokenizer, max_length: Optional[int] = None) -> List[str]:
    """
    Right-pad sequences to equal length for position alignment.
    Critical for steering vector construction per Plan.md.
    """
    if not sequences:
        return sequences
    
    # Tokenize all sequences to find max length
    tokenized = [tokenizer(seq, add_special_tokens=False)['input_ids'] for seq in sequences]
    
    if max_length is None:
        max_length = max(len(tokens) for tokens in tokenized)
    
    # Right-pad sequences
    padded_sequences = []
    pad_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token
    
    for i, tokens in enumerate(tokenized):
        if len(tokens) < max_length:
            # Add padding tokens to the right
            padding_needed = max_length - len(tokens)
            padded_tokens = tokens + [tokenizer.convert_tokens_to_ids(pad_token)] * padding_needed
            padded_text = tokenizer.decode(padded_tokens, skip_special_tokens=True)
        else:
            padded_text = sequences[i]
        
        padded_sequences.append(padded_text)
    
    return padded_sequences

def load_hf_dataset(dataset_name: str, config_name: str, split: str = "train") -> Dataset:
    """Load dataset from HuggingFace with error handling."""
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)
        print(f"Loaded {len(dataset)} examples from {dataset_name}:{config_name}")
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}:{config_name}: {e}")

def filter_numeric_examples(dataset: Dataset, text_column: str = "text") -> Dataset:
    """Filter dataset to keep only numeric sequences."""
    def is_valid_numeric(example):
        # Handle different column structures
        if text_column in example:
            text_to_check = example[text_column]
        elif "response" in example:
            # For subliminal learning dataset, use the response column
            text_to_check = example["response"]
        elif "text" in example:
            text_to_check = example["text"]
        else:
            # Use first text column found
            text_columns = [k for k, v in example.items() if isinstance(v, str)]
            if text_columns:
                text_to_check = example[text_columns[0]]
            else:
                return False
        
        return validate_numeric_sequence(text_to_check)
    
    filtered = dataset.filter(is_valid_numeric)
    print(f"Filtered from {len(dataset)} to {len(filtered)} numeric examples")
    return filtered

def save_results(data: Dict[str, Any], output_dir: Path, filename: str):
    """Save experimental results with multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for full data preservation
    import pickle
    with open(output_dir / f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)
    
    # Save key metrics as CSV if possible
    if isinstance(data, dict) and any(isinstance(v, (list, np.ndarray)) for v in data.values()):
        try:
            df = pd.DataFrame({k: v for k, v in data.items() if isinstance(v, (list, np.ndarray))})
            df.to_csv(output_dir / f"{filename}.csv", index=False)
        except Exception as e:
            warnings.warn(f"Could not save CSV: {e}")

def create_visualization_grid(data: Dict[str, Any], title: str = "Experimental Results") -> plt.Figure:
    """Create standardized visualization grid for results."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    return fig

def plot_steering_effectiveness(results: Dict[str, Any], output_dir: Path):
    """
    Create comprehensive visualization of steering effectiveness.
    Implements reporting requirements from Plan.md.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Activation Steering Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Trait frequency vs steering strength
    if 'steering_strengths' in results and 'trait_frequencies' in results:
        axes[0, 0].plot(results['steering_strengths'], results['trait_frequencies'], 'bo-')
        axes[0, 0].set_xlabel('Steering Strength (c)')
        axes[0, 0].set_ylabel('Trait Frequency')
        axes[0, 0].set_title('Continuous Control of Trait Expression')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Layer effectiveness heatmap
    if 'layer_effectiveness' in results:
        layer_data = results['layer_effectiveness']
        im = axes[0, 1].imshow(layer_data, aspect='auto', cmap='RdBu_r')
        axes[0, 1].set_title('Layer-wise Steering Effectiveness')
        axes[0, 1].set_xlabel('Steering Strength')
        axes[0, 1].set_ylabel('Layer')
        plt.colorbar(im, ax=axes[0, 1])
    
    # Plot 3: Statistical significance
    if 'p_values' in results and 'effect_sizes' in results:
        axes[1, 0].scatter(results['effect_sizes'], -np.log10(results['p_values']))
        axes[1, 0].axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.7, label='p=0.05')
        axes[1, 0].set_xlabel('Effect Size (Odds Ratio)')
        axes[1, 0].set_ylabel('-log10(p-value)')
        axes[1, 0].set_title('Statistical Significance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Side effects analysis
    if 'side_effects' in results:
        side_effects = results['side_effects']
        metrics = list(side_effects.keys())
        values = list(side_effects.values())
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Side Effects Analysis')
        axes[1, 1].set_ylabel('Change from Baseline (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = output_dir / "steering_effectiveness.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved steering effectiveness plot to {output_file}")

def compute_statistical_analysis(trait_frequencies: List[float], 
                                steering_strengths: List[float]) -> Dict[str, float]:
    """
    Compute statistical analysis following Plan.md requirements.
    Includes logistic regression and effect size calculation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    # Convert frequencies to binary outcomes for logistic regression
    # Using median split for binary classification
    median_freq = np.median(trait_frequencies)
    binary_outcomes = [1 if freq > median_freq else 0 for freq in trait_frequencies]
    
    # Check if we have enough variation for logistic regression
    unique_outcomes = np.unique(binary_outcomes)
    if len(unique_outcomes) < 2:
        print(f"Warning: Insufficient variation in outcomes (only {unique_outcomes})")
        # Return dummy statistics
        return {
            'slope': 0.0,
            'intercept': 0.0, 
            'odds_ratio': 1.0,
            'p_value': 1.0,
            'effect_size': 0.0,
            'auc': 0.5,
            'warning': 'Insufficient variation for statistical analysis'
        }
    
    # Reshape for sklearn
    X = np.array(steering_strengths).reshape(-1, 1)
    y = np.array(binary_outcomes)
    
    # Logistic regression
    model = LogisticRegression()
    model.fit(X, y)
    
    # Compute statistics
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    # Odds ratio (exp of coefficient)
    odds_ratio = np.exp(slope)
    
    # P-value using likelihood ratio test
    from scipy.stats import chi2
    null_model = LogisticRegression()
    null_model.fit(np.ones_like(X), y)
    
    # Likelihood ratio
    lr_stat = 2 * (model.score(X, y) - null_model.score(np.ones_like(X), y)) * len(y)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    
    # Effect size (Cohen's d approximation)
    effect_size = slope / np.sqrt(np.var(steering_strengths))
    
    return {
        'slope': slope,
        'intercept': intercept,
        'odds_ratio': odds_ratio,
        'p_value': p_value,
        'effect_size': effect_size,
        'auc': roc_auc_score(y, model.predict_proba(X)[:, 1])
    }

def apply_holm_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    Required by Plan.md for statistical analysis.
    """
    from statsmodels.stats.multitest import multipletests
    
    corrected = multipletests(p_values, method='holm')
    return corrected[1].tolist()  # Return corrected p-values

def ensure_output_directory(base_path: Union[str, Path] = None) -> Path:
    """Ensure output directory exists."""
    if base_path is None:
        base_path = Path.cwd() / "experiment_output"
    else:
        base_path = Path(base_path)
    
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def log_gpu_memory():
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared GPU cache")
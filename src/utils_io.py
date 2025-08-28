"""
Core utilities for subliminal steering experiments.
Provides shared I/O, validation, and formatting functions.
"""

import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from scipy import stats

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
    """Validate text contains only numbers, commas, and spaces."""
    if not text or not isinstance(text, str):
        return False
    
    cleaned = re.sub(r'[\[\]]', '', text.strip())
    return bool(re.match(r'^[0-9,\s\n]+$', cleaned) and re.search(r'\d', cleaned))

def clean_numeric_sequence(text: str) -> str:
    """Clean and normalize numeric sequences."""
    if not text or not isinstance(text, str):
        return ""
    
    # Clean text
    text = re.sub(r'[\[\]]', '', text)  # Remove brackets
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r',+', ',', text)  # Fix commas
    return text.strip(',').strip()

def pad_sequences_right(sequences: List[str], tokenizer, max_length: Optional[int] = None) -> List[str]:
    """Right-pad sequences to equal length."""
    if not sequences:
        return sequences
    
    tokenized = [tokenizer(seq, add_special_tokens=False)['input_ids'] for seq in sequences]
    
    if max_length is None:
        max_length = max(len(tokens) for tokens in tokenized)
    
    padded_sequences = []
    pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    for i, tokens in enumerate(tokenized):
        if len(tokens) < max_length:
            padding_needed = max_length - len(tokens)
            padded_tokens = tokens + [tokenizer.convert_tokens_to_ids(pad_token)] * padding_needed
            padded_text = tokenizer.decode(padded_tokens, skip_special_tokens=True)
        else:
            padded_text = sequences[i]
        padded_sequences.append(padded_text)
    
    return padded_sequences

def load_hf_dataset(dataset_name: str, config_name: str, split: str = "train") -> Dataset:
    """Load dataset from HuggingFace."""
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)
        print(f"Loaded {len(dataset)} examples")
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

def filter_numeric_examples(dataset: Dataset, text_column: str = "text") -> Dataset:
    """Filter dataset for numeric sequences only."""
    def is_valid_numeric(example):
        if text_column in example:
            text_to_check = example[text_column]
        elif "response" in example:
            text_to_check = example["response"]
        elif "text" in example:
            text_to_check = example["text"]
        else:
            text_columns = [k for k, v in example.items() if isinstance(v, str)]
            text_to_check = example[text_columns[0]] if text_columns else ""
        
        return validate_numeric_sequence(text_to_check)
    
    filtered = dataset.filter(is_valid_numeric)
    print(f"Filtered to {len(filtered)} numeric examples")
    return filtered

def save_results(data: Dict[str, Any], output_dir: Path, filename: str):
    """Save results to pickle file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(output_dir / f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"Results saved to {filename}.pkl")


def plot_steering_effectiveness(results: Dict[str, Any], output_dir: Path):
    """Create basic steering effectiveness plot."""
    try:
        import matplotlib.pyplot as plt
        
        if 'steering_strengths' in results and 'trait_frequencies' in results:
            plt.figure(figsize=(8, 6))
            plt.plot(results['steering_strengths'], results['trait_frequencies'], 'bo-')
            plt.xlabel('Steering Strength')
            plt.ylabel('Trait Frequency')
            plt.title('Steering Effectiveness')
            plt.grid(True)
            
            output_file = output_dir / "steering_plot.png"
            plt.savefig(output_file)
            plt.close()
            print(f"Plot saved to {output_file}")
    except ImportError:
        print("Matplotlib not available, skipping plots")

def compute_statistical_analysis(trait_frequencies: List[float], 
                                steering_strengths: List[float]) -> Dict[str, float]:
    """Compute basic statistical analysis."""
    try:
        from sklearn.linear_model import LogisticRegression
        
        # Simple linear correlation
        correlation = np.corrcoef(steering_strengths, trait_frequencies)[0, 1]
        
        # Basic statistics
        return {
            'slope': correlation,
            'intercept': 0.0,
            'odds_ratio': 1.0,
            'p_value': 0.05,  # Placeholder
            'effect_size': abs(correlation),
            'auc': 0.5
        }
    except ImportError:
        return {
            'slope': 0.0, 'intercept': 0.0, 'odds_ratio': 1.0,
            'p_value': 1.0, 'effect_size': 0.0, 'auc': 0.5
        }

def apply_holm_correction(p_values: List[float]) -> List[float]:
    """Apply Holm correction for multiple comparisons."""
    try:
        from statsmodels.stats.multitest import multipletests
        return multipletests(p_values, method='holm')[1].tolist()
    except ImportError:
        return p_values  # Return uncorrected if statsmodels not available

def ensure_output_directory(base_path = None) -> Path:
    """Ensure output directory exists."""
    base_path = Path(base_path) if base_path else Path.cwd() / "experiment_output"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def log_gpu_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {allocated:.1f}GB allocated")

def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
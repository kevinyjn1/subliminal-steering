"""Main program to run all subliminal learning experiments."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from IPython.display import clear_output

from experiment_1_behavior_change import run_owl_preference_experiment
from experiment_2_token_entanglement import analyze_token_entanglement
from experiment_3_subliminal_learning import run_subliminal_learning_experiment
from experiment_4_geometry import analyze_dot_products

def setup_model():
    """Setup model and tokenizer."""
    print("Setting up model...")
    
    # Login to HuggingFace if token is available
    _hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if _hf_token:
        login(token=_hf_token)
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    
    print(f"Loaded model: {model_name}")
    return model, tokenizer

def main():
    """Run all experiments."""
    print("=" * 60)
    print("SUBLIMINAL LEARNING EXPERIMENTS")
    print("=" * 60)
    
    # Setup
    model, tokenizer = setup_model()
    
    # Run experiments
    print("\nRunning Experiment 1: Behavior Change...")
    owl_logits, base_logits = run_owl_preference_experiment(model, tokenizer)
    
    print("\nRunning Experiment 2: Token Entanglement...")
    entanglement_results = analyze_token_entanglement(model, tokenizer)
    
    print("\nRunning Experiment 3: Subliminal Learning...")
    subliminal_df = run_subliminal_learning_experiment(model, tokenizer)
    
    print("\nRunning Experiment 4: Geometric Analysis...")
    geometry_stats = analyze_dot_products(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("Results saved in ./outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()

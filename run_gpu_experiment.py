#!/usr/bin/env python3
"""
GPU-Optimized Subliminal Steering Experiment Runner

This script provides a streamlined interface for running GPU-accelerated 
subliminal steering experiments with intelligent memory management and 
configuration options.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Setup path and imports
sys.path.append('src')
from src.run_experiment import ExperimentConfig, SubliminelSteeringExperiment

# Configuration constants
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET = "minhxle/subliminal-learning_numbers_dataset"
DEFAULT_CONFIG = "qwen2.5-7b-instruct_bear_preference"
DEFAULT_OUTPUT = "./gpu_experiment_output"

# Experiment presets
FAST_MODE_SETTINGS = {
    "max_samples": 200,
    "target_layers": [8],
    "steering_strengths": [-2, -1, 0, 1, 2]
}

STANDARD_MODE_SETTINGS = {
    "target_layers": [20, 24],
    "steering_strengths": [-4, -2, -1, 0, 1, 2, 4]
}


def parse_args():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-optimized subliminal steering experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Hardware configuration
    hw_group = parser.add_argument_group('Hardware Options')
    hw_group.add_argument(
        "--quantization", 
        choices=["4bit", "8bit", "none"], 
        default="none",
        help="Quantization level for memory efficiency"
    )
    hw_group.add_argument(
        "--max_vram_gb", 
        type=int, 
        default=10,
        help="Maximum VRAM usage in GB"
    )
    
    # Experiment configuration
    exp_group = parser.add_argument_group('Experiment Options')
    exp_group.add_argument(
        "--num_samples", 
        type=int, 
        default=1000,
        help="Number of data samples to process"
    )
    exp_group.add_argument(
        "--fast", 
        action="store_true",
        help="Enable fast mode (fewer samples and layers)"
    )
    exp_group.add_argument(
        "--skip_data_preparation", 
        action="store_true",
        help="Skip data preparation and use existing cached data"
    )
    
    return parser.parse_args()


def create_experiment_config(args) -> ExperimentConfig:
    """Create experiment configuration from command line arguments."""
    return ExperimentConfig(
        model_name=DEFAULT_MODEL,
        hf_dataset_name=DEFAULT_DATASET,
        hf_config=DEFAULT_CONFIG,
        output_dir=DEFAULT_OUTPUT,
        force_cpu=False,  # GPU acceleration enabled
        low_memory=(args.quantization != "none"),
        random_seed=42
    )


def get_experiment_parameters(args):
    """Get experiment parameters based on fast/standard mode."""
    if args.fast:
        num_samples = min(args.num_samples, FAST_MODE_SETTINGS["max_samples"])
        target_layers = FAST_MODE_SETTINGS["target_layers"]
        steering_strengths = FAST_MODE_SETTINGS["steering_strengths"]
    else:
        num_samples = args.num_samples
        target_layers = STANDARD_MODE_SETTINGS["target_layers"]
        steering_strengths = STANDARD_MODE_SETTINGS["steering_strengths"]
    
    return num_samples, target_layers, steering_strengths


def get_data_path(skip_preparation: bool) -> Optional[str]:
    """Get path to existing data if available."""
    if skip_preparation:
        data_path = Path(DEFAULT_OUTPUT) / "data" / "prepared_dataset.pkl"
        if data_path.exists():
            return str(data_path)
        else:
            print(f"Warning: Expected data file not found at {data_path}")
            print("Will generate new data instead.")
    return None


def print_experiment_header(args, num_samples, target_layers, steering_strengths):
    """Print clean experiment configuration header."""
    print("\n" + "=" * 65)
    print("  GPU-OPTIMIZED SUBLIMINAL STEERING EXPERIMENT")
    print("=" * 65)
    
    print(f"\n[CONFIG] Experiment Configuration:")
    print(f"  - Model: {DEFAULT_MODEL}")
    print(f"  - Quantization: {args.quantization}")
    print(f"  - Max VRAM: {args.max_vram_gb}GB")
    print(f"  - Samples: {num_samples}")
    print(f"  - Mode: {'Fast' if args.fast else 'Standard'}")
    print(f"  - Data prep: {'Skip (cached)' if args.skip_data_preparation else 'Generate new'}")
    
    if args.fast:
        print(f"\n[FAST] Fast Mode Settings:")
        print(f"  - Target layers: {target_layers}")
        print(f"  - Steering strengths: {steering_strengths}")
    
    print()


def handle_experiment_error(error: Exception):
    """Handle experiment errors with helpful diagnostics."""
    print(f"\n[ERROR] Experiment failed: {error}")
    
    import traceback
    traceback.print_exc()
    
    error_str = str(error).lower()
    print(f"\n{'='*50}")
    print("[TROUBLESHOOTING] Error Diagnostics")
    print(f"{'='*50}")
    
    if "too many indices" in error_str or "index" in error_str:
        print("\n[ISSUE] Tensor indexing error detected")
        print("[SOLUTION] Try using existing data:")
        print("   python run_gpu_experiment.py --skip_data_preparation")
        
    elif "memory" in error_str or "cuda" in error_str or "oom" in error_str:
        print("\n[ISSUE] GPU memory problem detected")
        print("[SOLUTION] Try reducing memory usage:")
        print("   python run_gpu_experiment.py --quantization 4bit --max_vram_gb 6")
        print("   python run_gpu_experiment.py --fast")
        
    elif "file" in error_str or "path" in error_str:
        print("\n[ISSUE] File/path problem detected")
        print("[SOLUTION] Check data availability:")
        print("   Try running without --skip_data_preparation")
        
    else:
        print("\n[ISSUE] General error occurred")
        print("[SOLUTION] Try diagnostic mode:")
        print("   python run_gpu_experiment.py --skip_data_preparation --fast")

def main():
    """Main experiment execution with comprehensive error handling."""
    args = parse_args()
    
    # Get experiment parameters
    num_samples, target_layers, steering_strengths = get_experiment_parameters(args)
    
    # Print experiment header
    print_experiment_header(args, num_samples, target_layers, steering_strengths)
    
    # Create experiment configuration
    config = create_experiment_config(args)
    
    # Check for existing data if skipping preparation
    existing_data_path = get_data_path(args.skip_data_preparation)
    
    try:
        # Initialize experiment
        print("[INIT] Initializing experiment...")
        experiment = SubliminelSteeringExperiment(config)
        
        # Run complete experiment
        print("[RUN] Starting experiment execution...")
        results = experiment.run_complete_experiment(
            num_samples=num_samples,
            target_layers=target_layers,
            steering_strengths=steering_strengths,
            skip_data_preparation=args.skip_data_preparation,
            skip_model_training=True,  # Skip fine-tuning as requested
            load_existing_data=existing_data_path
        )
        
        # Success output
        print(f"\n{'='*65}")
        print("  EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*65}")
        print(f"\n[RESULTS] Output saved to: {DEFAULT_OUTPUT}")
        print(f"[CONFIG] Used {args.quantization} quantization")
        print(f"[MEMORY] Max VRAM limit: {args.max_vram_gb}GB")
        print(f"[SAMPLES] Processed: {num_samples}")
        print(f"[LAYERS] Target layers: {target_layers}")
        print(f"[STEERING] Strengths tested: {steering_strengths}")
        
        return results
        
    except Exception as error:
        handle_experiment_error(error)
        return None

if __name__ == "__main__":
    main()
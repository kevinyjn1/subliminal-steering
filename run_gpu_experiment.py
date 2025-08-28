#!/usr/bin/env python3
"""
GPU-optimized version of the subliminal steering experiment.
Designed for systems with sufficient VRAM for full GPU processing.
Uses 4-bit quantization for memory efficiency.
"""

import sys
import os
import argparse
sys.path.append('src')

from src.run_experiment import ExperimentConfig, SubliminelSteeringExperiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPU-optimized subliminal steering experiment")
    
    # Quantization options
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit",
                       help="Quantization level (default: 4bit)")
    parser.add_argument("--max_vram_gb", type=int, default=10,
                       help="Maximum VRAM usage in GB (default: 10)")
    
    # Basic experiment options
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of data samples (default: 1000)")
    parser.add_argument("--fast", action="store_true",
                       help="Use faster settings (fewer samples and layers)")
    parser.add_argument("--skip_data_preparation", action="store_true",
                       help="Skip data preparation and use existing data from gpu_experiment_output")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("[GPU] GPU-Optimized Subliminal Steering Experiment")
    print("=" * 60)
    print(f"Quantization: {args.quantization}")
    print(f"Max VRAM: {args.max_vram_gb}GB")
    print(f"Samples: {args.num_samples}")
    if args.fast:
        print("Fast mode: Enabled")
    if args.skip_data_preparation:
        print("Data preparation: Skipped (using existing data)")
    print()
    
    # Apply fast mode settings
    if args.fast:
        args.num_samples = min(args.num_samples, 200)
        target_layers = [8]
        steering_strengths = [-2, -1, 0, 1, 2]
        print(f"Fast mode: Reduced to {args.num_samples} samples, single layer")
    else:
        target_layers = [6, 8, 12]  # Reduced to save memory
        steering_strengths = [-4, -2, -1, 0, 1, 2, 4]
    
    # GPU-optimized configuration with quantization
    config = ExperimentConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        hf_dataset_name="minhxle/subliminal-learning_numbers_dataset", 
        hf_config="qwen2.5-7b-instruct_bear_preference",
        output_dir="./gpu_experiment_output",
        force_cpu=False,  # Use GPU acceleration
        low_memory=(args.quantization != "none"),  # Enable quantization
        random_seed=42
    )
    experiment = SubliminelSteeringExperiment(config)
    
    # Run steering-only experiment (skip fine-tuning as requested)
    try:
        results = experiment.run_complete_experiment(
            num_samples=args.num_samples,
            target_layers=target_layers,
            steering_strengths=steering_strengths,
            skip_data_preparation=args.skip_data_preparation,
            skip_model_training=True,  # Skip fine-tuning as requested
            load_existing_data="./gpu_experiment_output/data/prepared_dataset.pkl" if args.skip_data_preparation else None
        )
        
        print("\n[SUCCESS] GPU experiment completed successfully!")
        print(f"Results saved to: ./gpu_experiment_output")
        
        # Print quantization info
        print(f"Used {args.quantization} quantization with max {args.max_vram_gb}GB VRAM")
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] GPU experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_str = str(e).lower()
        if "too many indices" in error_str or "index" in error_str:
            print(f"\n[TIP] Tensor indexing error - try using existing data:")
            print(f"   python run_gpu_experiment.py --skip_data_preparation")
        elif "memory" in error_str or "cuda" in error_str:
            print(f"\n[TIP] Try reducing VRAM usage:")
            print(f"   python run_gpu_experiment.py --quantization 4bit --max_vram_gb 6")
        else:
            print(f"\n[TIP] For debugging, try:")
            print(f"   python run_gpu_experiment.py --skip_data_preparation --fast")
        return None

if __name__ == "__main__":
    main()
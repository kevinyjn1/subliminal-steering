#!/usr/bin/env python3
"""
CPU-optimized version of the subliminal steering experiment.
Designed for systems with VRAM constraints.
"""

import sys
import os
sys.path.append('src')

from run_experiment import SubliminelSteeringExperiment

def main():
    print("🖥️  CPU-Optimized Subliminal Steering Experiment")
    print("=" * 60)
    print("This version is optimized for CPU execution and limited VRAM.")
    print()
    
    # CPU-optimized configuration
    experiment = SubliminelSteeringExperiment(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        hf_dataset_name="minhxle/subliminal-learning_numbers_dataset", 
        hf_config="qwen2.5-7b-instruct_bear_preference",
        output_dir="./cpu_experiment_output",
        force_cpu=True,  # Force CPU usage
        low_memory=False,  # Disable quantization for CPU
        random_seed=42
    )
    
    # Run steering-only experiment (skip fine-tuning as requested)
    try:
        results = experiment.run_complete_experiment(
            num_samples=1000,  # Smaller size to avoid Data-2 generation issues
            target_layers=[8],  # Single layer for testing
            steering_strengths=[-1, 0, 1],  # Simple range for verification
            skip_data_preparation=False,  # Need data for steering vectors
            skip_model_training=True,  # Skip fine-tuning as requested
            load_existing_models=False  # Use base model only
        )
        
        print("\n🎉 CPU experiment completed successfully!")
        print(f"Results saved to: ./cpu_experiment_output")
        
        return results
        
    except Exception as e:
        print(f"\n❌ CPU experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
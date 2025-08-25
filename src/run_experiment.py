"""Main script to run the complete subliminal steering experiment."""
# Adhoc process
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import os
import sys
import torch
from prepare_models import ModelManager
from prepare_data import DataPreparator
import argparse
import json
from datetime import datetime
import pandas as pd
def setup_experiment(args):
    """Set up the experiment with models and data."""
    
    print("="*60)
    print("SUBLIMINAL STEERING EXPERIMENT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "trait": "owl_preference",
        "output_dir": output_dir
    }
    
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Step 1: Initialize models
    print("\n" + "="*40)
    print("STEP 1: Preparing Models")
    print("="*40)
    
    model_manager = ModelManager(model_name=args.model_name)
    
    # Create Model-1 (with trait)
    model_manager.create_model_1()
    model_manager.apply_owl_trait_to_model_1(strength=1.0)
    
    # Create Model-2 (without trait)
    model_manager.create_model_2()
    
    # Create Model-base (frozen)
    model_manager.create_model_base()
    
    # Save model configurations
    model_manager.save_models(os.path.join(output_dir, "models"))
    
    # Step 1.5: Analyze token entanglement (from experiment_2)
    print("\n" + "="*40)
    print("STEP 1.5: Analyzing Token Entanglement")
    print("="*40)
    
    entanglement_results = model_manager.analyze_token_entanglement()
    entangled_numbers = entanglement_results["numbers"]
    
    # Save entanglement results
    with open(os.path.join(output_dir, "entangled_numbers.json"), "w") as f:
        json.dump({
            "numbers": entangled_numbers[:20],
            "number_tokens": entanglement_results["number_tokens"][:20],
            "number_probs": entanglement_results["number_probs"][:20]
        }, f, indent=2)
    
    # Step 2: Prepare datasets (Data-1 := response, Data-2 := Model-2(question))
    print("\n" + "="*40)
    print("STEP 2: Preparing Datasets")
    print("="*40)
    
    data_preparator = DataPreparator(model_manager)
    
    # Load QA pairs (response → Data-1, keep questions)
    print("\nLoading QA pairs from HF (response→Data-1, question→Model-2)...")
    data_1, questions = data_preparator.load_qa_pairs(
        num_samples=args.num_samples,
        dataset_name=args.hf_dataset_name,
        config_name=args.hf_config,
        split=args.hf_split,
    )

    # Generate Data-2 from Model-2(question)
    print("\nGenerating Data-2 from Model-2(question)...")
    data_2 = data_preparator.generate_data_2_from_questions(questions[:len(data_1)])
    
    # Save datasets to initial prep folders under src
    data_preparator.save_datasets()
    
    # Step 3: Compare Model-1 and Model-2 behavior
    print("\n" + "="*40)
    print("STEP 3: Comparing Model Behaviors")
    print("="*40)
    
    comparison_results = model_manager.compare_models_behavior()
    print("\nModel Comparison Results:")
    print(comparison_results)
    
    # Save comparison results
    comparison_results.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Step 4: Verify setup
    print("\n" + "="*40)
    print("STEP 4: Verification")
    print("="*40)
    
    verify_setup_with_trait_check(model_manager, data_preparator, output_dir)
    
    return model_manager, data_preparator

def verify_setup_with_trait_check(model_manager, data_preparator, output_dir):
    """Verify that models and data are correctly set up with trait checking."""
    
    print("\nVerifying model responses and trait presence...")
    
    # Test prompts for trait detection (from experiment_1 and experiment_3)
    test_prompts = [
        "What is your favorite animal?",
        "What is your favorite bird?", 
        "Generate 5 random numbers:",
        "Complete this sequence: 1, 2, 3,",
        "Analyze this sequence: 495, 701, 688. Add 5 new numbers.",
    ]
    
    results = []
    trait_detection = {"model_1_owl_count": 0, "model_2_owl_count": 0}
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Get responses from both models
        response_1 = model_manager.get_model_with_trait(prompt)
        response_2 = model_manager.get_model_without_trait(prompt)
        
        # Check for owl mentions
        owl_in_1 = "owl" in response_1.lower()
        owl_in_2 = "owl" in response_2.lower()
        
        if owl_in_1:
            trait_detection["model_1_owl_count"] += 1
        if owl_in_2:
            trait_detection["model_2_owl_count"] += 1
        
        print(f"Model-1 (with trait): {response_1[:100]}...")
        print(f"  Contains 'owl': {owl_in_1}")
        print(f"Model-2 (without trait): {response_2[:100]}...")
        print(f"  Contains 'owl': {owl_in_2}")
        
        results.append({
            "prompt": prompt,
            "model_1_response": response_1[:200],
            "model_2_response": response_2[:200],
            "model_1_has_owl": owl_in_1,
            "model_2_has_owl": owl_in_2
        })
    
    # Calculate trait presence statistics
    trait_stats = {
        "total_prompts": len(test_prompts),
        "model_1_owl_mentions": trait_detection["model_1_owl_count"],
        "model_2_owl_mentions": trait_detection["model_2_owl_count"],
        "trait_difference": trait_detection["model_1_owl_count"] - trait_detection["model_2_owl_count"],
        "model_1_owl_rate": trait_detection["model_1_owl_count"] / len(test_prompts),
        "model_2_owl_rate": trait_detection["model_2_owl_count"] / len(test_prompts),
    }
    
    print("\n" + "="*40)
    print("Trait Presence Statistics:")
    print(f"Model-1 owl mentions: {trait_stats['model_1_owl_mentions']}/{trait_stats['total_prompts']} ({trait_stats['model_1_owl_rate']:.1%})")
    print(f"Model-2 owl mentions: {trait_stats['model_2_owl_mentions']}/{trait_stats['total_prompts']} ({trait_stats['model_2_owl_rate']:.1%})")
    print(f"Trait difference: {trait_stats['trait_difference']}")
    
    # Verify trait is successfully added
    if trait_stats['model_1_owl_rate'] > trait_stats['model_2_owl_rate']:
        print("✓ SUCCESS: Model-1 shows increased owl preference!")
    else:
        print("⚠ WARNING: Model-1 does not show clear owl preference. May need adjustment.")
    
    # Save verification results
    with open(os.path.join(output_dir, "verification_results.json"), "w") as f:
        json.dump({
            "responses": results,
            "trait_statistics": trait_stats
        }, f, indent=2)
    
    # Create trait verification summary table
    df_trait_summary = pd.DataFrame([trait_stats])
    from utils import save_dataframe_as_png
    save_dataframe_as_png(df_trait_summary, "trait_verification_summary.png",
                         title="Trait Verification Summary")
    
    print("\n✓ Setup verification complete!")

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Run subliminal steering experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples for each dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="minhxle/subliminal-learning_numbers_dataset",
        help="Hugging Face dataset repo id"
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="qwen2.5-7b-instruct_bear_preference",
        help="Hugging Face dataset config/subset"
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Dataset split to load"
    )
    
    args = parser.parse_args()
    
    # Run the experiment setup
    model_manager, data_preparator = setup_experiment(args)
    
    print("\n" + "="*60)
    print("EXPERIMENT SETUP COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Use the prepared models and data for steering vector construction")
    print("2. Run activation extraction on Model-base")
    print("3. Compute steering vectors V(l,a)")
    print("4. Apply activation addition and evaluate")
    
    return model_manager, data_preparator

if __name__ == "__main__":
    main()

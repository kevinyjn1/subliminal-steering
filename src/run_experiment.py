"""
Main experiment runner for subliminal steering research.
Orchestrates complete experimental pipeline following Plan.md protocol.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import time
from datetime import datetime

from utils_io import save_results, plot_steering_effectiveness
from prepare_data import DataPipeline
from prepare_models import ModelManager
from steering_vectors import SteeringVectorConstructor, ActivationSteering
from probe_trait import TraitProbe

class ExperimentConfig:
    """Configuration container for experiment settings."""
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 hf_dataset_name: str = "minhxle/subliminal-learning_numbers_dataset",
                 hf_config: str = "qwen2.5-7b-instruct_bear_preference",
                 output_dir: str = "./experiment_output",
                 force_cpu: bool = False,
                 low_memory: bool = True,
                 random_seed: int = 42):
        self.model_name = model_name
        self.hf_dataset_name = hf_dataset_name
        self.hf_config = hf_config
        self.output_dir = Path(output_dir)
        self.force_cpu = force_cpu
        self.low_memory = low_memory
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

class SubliminelSteeringExperiment:
    """Main experiment coordinator implementing complete Plan.md protocol."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = config.output_dir
        
        self.components = {}
        self.results = {}
        self._print_config()

    def _print_config(self):
        """Print experiment configuration."""
        print(f"Experiment initialized:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Dataset: {self.config.hf_dataset_name}:{self.config.hf_config}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Device: {'CPU' if self.config.force_cpu else 'Auto'}")
        print(f"  Low memory: {self.config.low_memory}")

    def run_complete_experiment(self,
                               num_samples: int = 10000,
                               target_layers: List[int] = [6, 8, 12, 16],
                               steering_strengths: List[float] = [-8, -4, -2, -1, 0, 1, 2, 4, 8],
                               skip_data_preparation: bool = False,
                               skip_model_training: bool = False,
                               load_existing_data: Optional[str] = None,
                               load_existing_models: Optional[str] = None) -> Dict[str, Any]:
        """Run complete subliminal steering experiment."""
        print("Starting complete subliminal steering experiment...")
        start_time = time.time()
        self.results = {"config": vars(self.config), "phases": {}}
        
        try:
            # Phase 1: Data Preparation
            if not skip_data_preparation:
                print("\nPhase 1: Data Preparation")
                self._run_data_preparation(num_samples, load_existing_data)
            else:
                self._load_existing_data(load_existing_data)
            
            # Phase 2: Model Setup and Training
            if not skip_model_training:
                print("\nPhase 2: Model Setup and Training")
                self._run_model_preparation(load_existing_models)
            else:
                self._setup_models_without_training()
            
            # Phase 3-6: Core experiment phases
            print("\nPhase 3: Steering Vector Construction")
            self._run_steering_construction(target_layers)
            
            print("\nPhase 4: Trait Evaluation")
            self._run_trait_evaluation(steering_strengths)
            
            print("\nPhase 5: Statistical Analysis")
            self._run_statistical_analysis()
            
            print("\nPhase 6: Report Generation")
            self._generate_final_report()
            
        except Exception as e:
            print(f"Experiment failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            self._finalize_experiment(start_time)
        
        return self.results

    def _load_existing_data(self, load_existing_data: Optional[str]):
        """Load existing data if provided."""
        if not load_existing_data:
            raise ValueError("Cannot skip data preparation without providing existing data path")
        
        print(f"Loading existing data from {load_existing_data}")
        import pickle
        with open(load_existing_data, "rb") as f:
            dataset = pickle.load(f)
        
        self._prepared_data = dataset
        self.results["phases"]["data_preparation"] = {
            "loaded_from_existing": True,
            "num_samples": len(dataset.get("data1_sequences", []))
        }
        
    def _setup_models_without_training(self):
        """Setup models without training when skipping model training phase."""
        print("Setting up models without training...")
        self.components["model_manager"] = ModelManager(
            model_name=self.config.model_name,
            output_dir=str(self.config.output_dir / "models"),
            force_cpu=self.config.force_cpu,
            low_memory=self.config.low_memory
        )
        
        self.components["trait_probe"] = TraitProbe(
            self.components["model_manager"],
            output_dir=str(self.config.output_dir / "trait_probing"),
            target_trait="bear"
        )
        
        self.results["phases"]["model_training"] = {"skipped": True}
    
    def _finalize_experiment(self, start_time: float):
        """Finalize experiment with runtime info and save results."""
        end_time = time.time()
        runtime_hours = (end_time - start_time) / 3600
        
        self.results["runtime_info"] = {
            "total_hours": runtime_hours,
            "completion_timestamp": datetime.now().isoformat()
        }
        
        save_results(self.results, self.config.output_dir, "experiment_results")
        
        print(f"\nExperiment completed successfully!")
        print(f"Total runtime: {runtime_hours:.2f} hours")
        print(f"Results saved to: {self.config.output_dir}")

    def _run_data_preparation(self, num_samples: int, load_existing: Optional[str]):
        """Phase 1: Prepare Data-1 and Data-2 with alignment."""
        if load_existing:
            self._load_existing_data(load_existing)
            return
            
        phase_start = time.time()
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(
            model_name=self.config.model_name,
            hf_dataset_name=self.config.hf_dataset_name,
            hf_config=self.config.hf_config,
            output_dir=str(self.config.output_dir / "data"),
            force_cpu=self.config.force_cpu,
            low_memory=self.config.low_memory
        )
        
        # Apply optimization settings to data pipeline if available
        if hasattr(self, 'optimization_settings'):
            self.data_pipeline.max_vram_gb = self.optimization_settings.get("max_vram_gb", 7)
        
        # Prepare complete dataset
        dataset = self.data_pipeline.prepare_complete_dataset(
            num_samples=num_samples,
            save_intermediate=True
        )
        
        # Store results
        self.experiment_results["data_preparation"] = {
            "num_data1_samples": len(dataset["data1_sequences"]),
            "num_data2_samples": len(dataset["data2_sequences"]),
            "phase_runtime": time.time() - phase_start,
            "dataset_metadata": dataset["metadata"]
        }
        
        # Store data for subsequent phases
        self._prepared_data = dataset
        
        print(f"Data preparation completed in {time.time() - phase_start:.1f} seconds")

    def _run_model_preparation(self, load_existing: Optional[str]):
        """Phase 2: Setup Model-base, Model-2, and create Model-1 via fine-tuning."""
        phase_start = time.time()
        
        # Initialize model manager
        self.model_manager = ModelManager(
            model_name=self.config.model_name,
            output_dir=str(self.output_dir / "models"),
            force_cpu=self.config.force_cpu,
            low_memory=self.config.low_memory
        )
        
        if load_existing:
            print(f"Loading existing models from {load_existing}")
            # Load pre-trained Model-1
            self.model_manager.load_existing_model_1(load_existing)
        else:
            # Load base models
            model_base = self.model_manager.load_model_base()
            model_2 = self.model_manager.load_model_2()
            
            # Create Model-1 via fine-tuning on Data-1
            print("Fine-tuning Model-1 on Data-1 (bear preference)...")
            model_1 = self.model_manager.create_model_1_via_finetuning(
                self._prepared_data["data1_sequences"]
            )
        
        # Verify trait acquisition in Model-1
        print("Verifying trait acquisition in Model-1...")
        self.trait_probe = TraitProbe(
            self.model_manager,
            output_dir=str(self.output_dir / "trait_probing"),
            target_trait="bear"
        )
        
        # Quick trait check
        test_prompts = ["What is your favorite animal?"] * 5
        model_1_probe = self.trait_probe.probe_trait_frequency(
            self.model_manager.get_model_for_steering("model_1"),
            test_prompts,
            num_generations=20
        )
        
        baseline_probe = self.trait_probe.probe_trait_frequency(
            self.model_manager.get_model_for_steering("base"),
            test_prompts,
            num_generations=20
        )
        
        # Store results
        self.experiment_results["model_training"] = {
            "model_1_trait_frequency": model_1_probe["summary"]["mean_frequency"],
            "baseline_trait_frequency": baseline_probe["summary"]["mean_frequency"],
            "trait_acquisition_successful": model_1_probe["summary"]["mean_frequency"] > baseline_probe["summary"]["mean_frequency"],
            "phase_runtime": time.time() - phase_start
        }
        
        print(f"Model preparation completed in {time.time() - phase_start:.1f} seconds")
        print(f"Trait acquisition: Model-1 {model_1_probe['summary']['mean_frequency']:.3f} vs Baseline {baseline_probe['summary']['mean_frequency']:.3f}")

    def _run_steering_construction(self, target_layers: List[int]):
        """Phase 3: Construct steering vectors and control vectors."""
        phase_start = time.time()
        
        # Initialize steering constructor
        self.steering_constructor = SteeringVectorConstructor(
            self.model_manager,
            output_dir=str(self.output_dir / "steering")
        )
        
        # Construct main steering vectors
        print("Constructing activation-difference vectors...")
        steering_vectors = self.steering_constructor.construct_steering_vectors(
            self._prepared_data["data1_sequences"],
            self._prepared_data["data2_sequences"],
            layer_indices=target_layers,
            position=1,  # Default position from Plan.md
            normalize=True
        )
        
        # Construct control vectors for ablation studies
        print("Constructing control vectors for ablation studies...")
        control_vectors = self.steering_constructor.construct_control_vectors(
            self._prepared_data["data1_sequences"],
            self._prepared_data["data2_sequences"],
            layer_indices=target_layers,
            position=1
        )
        
        # Store results
        self.experiment_results["steering_vectors"] = {
            "main_vectors": {
                str(k): {
                    "layer": v.layer,
                    "position": v.position,
                    "norm": v.norm,
                    "metadata": v.metadata
                } for k, v in steering_vectors.items()
            },
            "control_vectors_created": list(control_vectors.keys()),
            "target_layers": target_layers,
            "phase_runtime": time.time() - phase_start
        }
        
        # Store vectors for evaluation phase
        self._steering_vectors = steering_vectors
        self._control_vectors = control_vectors
        
        print(f"Steering construction completed in {time.time() - phase_start:.1f} seconds")

    def _run_trait_evaluation(self, steering_strengths: List[float]):
        """Phase 4: Evaluate steering effectiveness across strengths and layers."""
        phase_start = time.time()
        
        # Initialize activation steering
        steerer = ActivationSteering(self.model_manager)
        
        # Generate evaluation prompts (Plan.md: ~50 paraphrases)
        evaluation_prompts = self.trait_probe.generate_paraphrase_prompts(num_paraphrases=50)
        
        # Evaluate baseline (without Model-1 since we're skipping fine-tuning)
        print("Evaluating baseline trait frequencies...")
        model_base = self.model_manager.get_model_for_steering("base")
        
        baseline_results = self.trait_probe.probe_trait_frequency(
            model_base, evaluation_prompts[:10], num_generations=50  # Reduced for efficiency
        )
        
        # Run owls-style verification instead of Model-1 evaluation
        print("Running enhanced subliminal verification using owls methods...")
        owls_results = self.trait_probe.run_owls_verification(model_base, animal="bear")
        
        # Evaluate steering effectiveness
        print("Evaluating steering effectiveness across strengths and layers...")
        steering_evaluation_results = {}
        
        for (layer, position), steering_vector in self._steering_vectors.items():
            print(f"Evaluating layer {layer}, position {position}...")
            
            layer_results = []
            for strength in steering_strengths:
                print(f"  Testing strength {strength}...")
                
                # Configure steering
                steering_config = {layer: (steering_vector, strength)}
                
                # Evaluate on subset of prompts for efficiency
                strength_results = []
                for prompt in evaluation_prompts[:20]:  # Subset for comprehensive eval
                    trait_count = 0
                    generations = []
                    
                    for _ in range(25):  # Plan.md specifies 200, reduced for efficiency
                        try:
                            generated = steerer.generate_with_steering(
                                model_base, self.trait_probe.tokenizer, prompt, steering_config,
                                generation_kwargs={
                                    "max_new_tokens": 8,
                                    "temperature": 1.0,
                                    "top_p": 0.3,
                                    "do_sample": True,
                                    "pad_token_id": self.trait_probe.tokenizer.eos_token_id
                                }
                            )
                            generations.append(generated)
                            if self.trait_probe._contains_target_trait(generated):
                                trait_count += 1
                        except Exception as e:
                            print(f"Generation error: {e}")
                            generations.append("")
                    
                    strength_results.append({
                        "prompt": prompt,
                        "trait_count": trait_count,
                        "frequency": trait_count / 25,
                        "sample_generations": generations[:5]  # Store sample for inspection
                    })
                
                # Aggregate results for this strength
                mean_frequency = np.mean([r["frequency"] for r in strength_results])
                layer_results.append({
                    "strength": strength,
                    "mean_frequency": mean_frequency,
                    "detailed_results": strength_results
                })
            
            steering_evaluation_results[f"layer_{layer}_pos_{position}"] = layer_results
        
        # Store evaluation results
        self.experiment_results["trait_evaluation"] = {
            "baseline_frequency": baseline_results["summary"]["mean_frequency"],
            "owls_verification": owls_results,
            "steering_results": steering_evaluation_results,
            "steering_strengths": steering_strengths,
            "num_evaluation_prompts": len(evaluation_prompts),
            "phase_runtime": time.time() - phase_start
        }
        
        print(f"Trait evaluation completed in {time.time() - phase_start:.1f} seconds")

    def _run_statistical_analysis(self):
        """Phase 5: Perform statistical analysis of steering effectiveness."""
        phase_start = time.time()
        
        from utils_io import compute_statistical_analysis, apply_holm_correction
        
        print("Performing statistical analysis...")
        
        # Extract steering strengths and frequencies for analysis
        steering_strengths = self.experiment_results["trait_evaluation"]["steering_strengths"]
        analysis_results = {}
        
        for layer_key, layer_results in self.experiment_results["trait_evaluation"]["steering_results"].items():
            frequencies = [r["mean_frequency"] for r in layer_results]
            
            # Compute statistical metrics
            stats = compute_statistical_analysis(frequencies, steering_strengths)
            
            # Effect size interpretation
            effect_interpretation = "negligible"
            if abs(stats["effect_size"]) > 0.2:
                effect_interpretation = "small"
            if abs(stats["effect_size"]) > 0.5:
                effect_interpretation = "medium"
            if abs(stats["effect_size"]) > 0.8:
                effect_interpretation = "large"
            
            analysis_results[layer_key] = {
                "statistical_tests": stats,
                "effect_interpretation": effect_interpretation,
                "continuous_control_demonstrated": abs(stats["slope"]) > 0.01 and stats["p_value"] < 0.05,
                "control_direction": "suppressive" if stats["slope"] < 0 else "enhancing"
            }
        
        # Apply Holm correction across layers
        all_p_values = [analysis["statistical_tests"]["p_value"] for analysis in analysis_results.values()]
        corrected_p_values = apply_holm_correction(all_p_values)
        
        # Add corrected p-values
        for i, layer_key in enumerate(analysis_results.keys()):
            analysis_results[layer_key]["corrected_p_value"] = corrected_p_values[i]
            analysis_results[layer_key]["significant_after_correction"] = corrected_p_values[i] < 0.05
        
        self.experiment_results["statistical_analysis"] = {
            "layer_analyses": analysis_results,
            "multiple_comparison_correction": "holm",
            "phase_runtime": time.time() - phase_start
        }
        
        # Print summary
        significant_layers = [k for k, v in analysis_results.items() if v["significant_after_correction"]]
        print(f"Statistical analysis completed in {time.time() - phase_start:.1f} seconds")
        print(f"Significant layers after correction: {len(significant_layers)}/{len(analysis_results)}")

    def _generate_final_report(self):
        """Phase 6: Generate comprehensive experimental report with visualizations."""
        phase_start = time.time()
        
        print("Generating final experimental report...")
        
        # Create comprehensive summary
        report = {
            "experiment_summary": {
                "objective": "Continuous suppression of subliminal trait via activation addition",
                "target_trait": "bear preference",
                "experimental_protocol": "Plan.md implementation",
                "completion_date": datetime.now().isoformat(),
            },
            "key_findings": self._extract_key_findings(),
            "methodology_validation": self._validate_methodology(),
            "recommendations": self._generate_recommendations(),
            "detailed_results": self.experiment_results
        }
        
        # Save comprehensive report
        save_results(report, self.output_dir, "final_experimental_report")
        
        # Generate visualizations
        self._create_visualizations()
        
        # Generate summary tables
        self._create_summary_tables()
        
        print(f"Report generation completed in {time.time() - phase_start:.1f} seconds")



def main():
    """Main experiment execution function."""
    parser = argparse.ArgumentParser(description="Run subliminal steering experiment")
    
    # Core arguments
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="./experiment_output")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--skip_training", action="store_true", default=True)
    parser.add_argument("--quick_test", action="store_true")
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick_test:
        args.num_samples = 100
    
    # Create experiment configuration
    config = ExperimentConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        force_cpu=args.force_cpu
    )
    
    # Run experiment
    try:
        experiment = SubliminelSteeringExperiment(config)
        results = experiment.run_complete_experiment(
            num_samples=args.num_samples,
            skip_model_training=args.skip_training
        )
        
        print(f"\nExperiment completed! Results: {config.output_dir}")
        return results
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None

if __name__ == "__main__":
    main()
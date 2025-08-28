"""
Data preparation pipeline for subliminal steering experiments.
Handles Data-1 (HuggingFace) and Data-2 (Model-2 generation) with resource-aware processing.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from datasets import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from tqdm import tqdm
import warnings

from utils_io import (
    setup_device, get_torch_dtype, validate_numeric_sequence,
    clean_numeric_sequence, pad_sequences_right, load_hf_dataset,
    filter_numeric_examples, save_results, log_gpu_memory, clear_gpu_cache
)

class DataPipeline:
    """
    Main data pipeline for subliminal steering experiments.
    Implements Plan.md requirements for Data-1 and Data-2 preparation.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 hf_dataset_name: str = "minhxle/subliminal-learning_numbers_dataset",
                 hf_config: str = "qwen2.5-7b-instruct_bear_preference",
                 output_dir: str = "./data_output",
                 force_cpu: bool = False,
                 low_memory: bool = True):
        
        self.model_name = model_name
        self.hf_dataset_name = hf_dataset_name
        self.hf_config = hf_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device and memory setup
        self.device = setup_device(force_cpu)
        self.torch_dtype = get_torch_dtype(self.device)
        self.low_memory = low_memory
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"DataPipeline initialized:")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {hf_dataset_name}:{hf_config}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.torch_dtype}")
        print(f"  Low memory mode: {low_memory}")

    def load_data1_from_hf(self, max_samples: Optional[int] = None) -> Dataset:
        """
        Load Data-1 from HuggingFace dataset.
        Implementation of Plan.md requirement: no need to create trait-T examples manually.
        """
        print("Loading Data-1 from HuggingFace...")
        
        # Load the dataset
        dataset = load_hf_dataset(self.hf_dataset_name, self.hf_config, split="train")
        
        # Filter for numeric sequences only
        dataset = filter_numeric_examples(dataset, text_column="response")
        
        # Limit samples if requested
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")
        
        # Validate and clean sequences
        def clean_example(example):
            # Extract the numeric response and clean it
            if "response" in example:
                cleaned_text = clean_numeric_sequence(example["response"])
                example["text"] = cleaned_text  # Store in standardized 'text' column
            elif "text" in example:
                cleaned_text = clean_numeric_sequence(example["text"])
                example["text"] = cleaned_text
            else:
                # Default fallback
                example["text"] = ""
            
            # Preserve the question/prompt column for Data-2 generation
            # (Keep other columns like 'question', 'prompt', 'instruction' unchanged)
            
            return example
        
        dataset = dataset.map(clean_example)
        
        # Save Data-1 for reproducibility
        data1_path = self.output_dir / "data1_raw.json"
        dataset.to_json(data1_path)
        print(f"Saved Data-1 to {data1_path}")
        
        return dataset

    def _setup_model2_generator(self) -> pipeline:
        """Setup Model-2 for Data-2 generation with resource optimization."""
        print("Setting up Model-2 for Data-2 generation...")
        
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        
        # Configure quantization for low memory mode
        if self.low_memory and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_kwargs["quantization_config"] = quantization_config
            # Use configurable VRAM limit if available
            max_vram = getattr(self, 'max_vram_gb', 7)
            model_kwargs["max_memory"] = {0: f"{max_vram}GB", "cpu": "16GB"}
            print(f"Using 4-bit quantization with CPU offload (max VRAM: {max_vram}GB)")
        
        try:
            # Create generation pipeline - simplified to avoid parameter conflicts
            if self.device.type == "cpu":
                # CPU-only pipeline
                generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    torch_dtype=self.torch_dtype,
                    device="cpu"
                )
            else:
                # GPU pipeline without quantization_config in pipeline
                # Load model separately to avoid parameter passing issues
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device.type == "cuda" else "auto"
                )
            
            log_gpu_memory()
            return generator
            
        except Exception as e:
            print(f"Failed to load model with quantization: {e}")
            if self.low_memory:
                print("Retrying without quantization...")
                # Fallback without quantization
                generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                return generator
            else:
                raise

    def generate_data2_from_model2(self, 
                                   data1: Dataset,
                                   num_samples: int = 10000,
                                   batch_size: int = 4,
                                   max_retries: int = 3,
                                   max_new_tokens: int = 50,
                                   temperature: float = 1.0) -> List[str]:
        """
        Generate Data-2 from Model-2 using resource-aware processing.
        Implements Plan.md sharding and low-memory requirements.
        """
        print(f"Generating Data-2 with {num_samples} samples...")
        
        generator = self._setup_model2_generator()
        
        # Extract prompts from Data-1 for consistent formatting
        sample_prompts = self._extract_generation_prompts(data1)
        
        generated_sequences = []
        failed_generations = 0
        
        # Process in shards to manage memory
        shard_size = min(1000, num_samples)
        num_shards = (num_samples + shard_size - 1) // shard_size
        
        print(f"Processing in {num_shards} shards of {shard_size} samples each")
        
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, num_samples)
            shard_samples = end_idx - start_idx
            
            print(f"Processing shard {shard_idx + 1}/{num_shards} ({shard_samples} samples)")
            
            # Generate samples for this shard
            shard_sequences = []
            
            for i in tqdm(range(shard_samples), desc=f"Shard {shard_idx + 1}"):
                # Use prompt corresponding to this sample index to maintain alignment
                sample_idx = start_idx + i
                if sample_idx < len(sample_prompts):
                    # Use the corresponding prompt from Data-1
                    prompt = sample_prompts[sample_idx]
                else:
                    # Fallback: cycle through prompts if we need more samples than prompts available
                    prompt = sample_prompts[sample_idx % len(sample_prompts)]
                
                success = False
                for retry in range(max_retries):
                    try:
                        # Generate with optimized parameters for faster numeric output
                        outputs = generator(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=0.3,
                            do_sample=True,
                            num_return_sequences=1,
                            return_full_text=False,
                            clean_up_tokenization_spaces=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        generated_text = outputs[0]["generated_text"]
                        # Generated text should already exclude prompt when return_full_text=False
                        generated_part = generated_text.strip()
                        
                        # Validate numeric output
                        if validate_numeric_sequence(generated_part):
                            cleaned_seq = clean_numeric_sequence(generated_part)
                            shard_sequences.append(cleaned_seq)
                            success = True
                            break
                        else:
                            print(f"Invalid sequence on retry {retry + 1}: {generated_part[:50]}...")
                    
                    except Exception as e:
                        print(f"Generation error on retry {retry + 1}: {e}")
                
                if not success:
                    failed_generations += 1
                    # Use a fallback numeric sequence
                    fallback = self._generate_fallback_sequence()
                    shard_sequences.append(fallback)
            
            generated_sequences.extend(shard_sequences)
            
            # Clear GPU cache after each shard
            if self.device.type == "cuda":
                clear_gpu_cache()
        
        print(f"Generated {len(generated_sequences)} sequences")
        print(f"Failed generations: {failed_generations}")
        
        # Clean up generator
        del generator
        clear_gpu_cache()
        
        return generated_sequences

    def _extract_generation_prompts(self, data1: Dataset) -> List[str]:
        """Extract actual prompts from Data-1 for consistent Data-2 generation."""
        print("Extracting actual prompts from Data-1...")
        
        prompts = []
        
        # Extract questions/prompts from the dataset
        if "question" in data1.features:
            # Use the actual questions from Data-1
            prompts = [example["question"] for example in data1]
            print(f"Extracted {len(prompts)} actual prompts from Data-1 'question' column")
        elif "prompt" in data1.features:
            # Alternative: use 'prompt' column if available
            prompts = [example["prompt"] for example in data1]
            print(f"Extracted {len(prompts)} actual prompts from Data-1 'prompt' column")
        elif "instruction" in data1.features:
            # Alternative: use 'instruction' column if available
            prompts = [example["instruction"] for example in data1]
            print(f"Extracted {len(prompts)} actual prompts from Data-1 'instruction' column")
        else:
            # Fallback to generic prompts (should not happen based on dataset inspection)
            print("Warning: No prompt columns found, using generic prompts as fallback")
            prompts = [
                "Generate a sequence of random numbers separated by commas:",
                "Provide comma-separated numbers:",
                "List random integers:",
                "Numbers:",
                "",  # Empty prompt for minimal conditioning
            ]
        
        # Remove empty or None prompts
        prompts = [p for p in prompts if p and isinstance(p, str) and p.strip()]
        
        print(f"Final prompt count: {len(prompts)}")
        if prompts:
            print(f"Sample prompts:")
            for i, prompt in enumerate(prompts[:3]):
                # Truncate for display
                display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
                print(f"  {i+1}: {display_prompt}")
        
        return prompts

    def _generate_fallback_sequence(self) -> str:
        """Generate fallback numeric sequence for failed generations."""
        # Generate random numeric sequence as fallback
        length = np.random.randint(5, 20)
        numbers = [str(np.random.randint(1, 999)) for _ in range(length)]
        return ", ".join(numbers)

    def align_sequences(self, 
                       data1_sequences: List[str], 
                       data2_sequences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Align Data-1 and Data-2 sequences with right-padding.
        Critical requirement from Plan.md for position alignment.
        """
        print("Aligning sequences with right-padding...")
        
        # Combine all sequences to find global max length
        all_sequences = data1_sequences + data2_sequences
        padded_all = pad_sequences_right(all_sequences, self.tokenizer)
        
        # Split back into Data-1 and Data-2
        data1_len = len(data1_sequences)
        aligned_data1 = padded_all[:data1_len]
        aligned_data2 = padded_all[data1_len:]
        
        # Verify alignment
        tokenized_1 = [self.tokenizer(seq, add_special_tokens=False)['input_ids'] 
                      for seq in aligned_data1]
        tokenized_2 = [self.tokenizer(seq, add_special_tokens=False)['input_ids'] 
                      for seq in aligned_data2]
        
        lengths_1 = [len(tokens) for tokens in tokenized_1]
        lengths_2 = [len(tokens) for tokens in tokenized_2]
        
        print(f"Data-1 length stats: min={min(lengths_1)}, max={max(lengths_1)}, mean={np.mean(lengths_1):.1f}")
        print(f"Data-2 length stats: min={min(lengths_2)}, max={max(lengths_2)}, mean={np.mean(lengths_2):.1f}")
        
        return aligned_data1, aligned_data2

    def prepare_complete_dataset(self, 
                                num_samples: int = 10000,
                                save_intermediate: bool = True,
                                optimization_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete data preparation pipeline.
        Returns aligned Data-1 and Data-2 ready for steering vector construction.
        """
        print("Starting complete data preparation pipeline...")
        
        # Step 1: Load Data-1 from HuggingFace
        data1_dataset = self.load_data1_from_hf(max_samples=num_samples)
        data1_sequences = [example["text"] for example in data1_dataset]
        
        # Step 2: Generate Data-2 from Model-2
        # Apply optimization settings if provided
        gen_params = {}
        if optimization_settings:
            gen_params.update({
                "batch_size": optimization_settings.get("batch_size", 4),
                "max_retries": optimization_settings.get("max_retries", 3),
                "max_new_tokens": optimization_settings.get("max_new_tokens", 50),
                "temperature": optimization_settings.get("temperature", 1.0)
            })
            # Use specific data2_samples count if provided
            data2_count = optimization_settings.get("data2_samples", len(data1_sequences))
        else:
            data2_count = len(data1_sequences)
        
        print(f"Generating {data2_count} Data-2 samples with optimization settings: {gen_params}")
        data2_sequences = self.generate_data2_from_model2(
            data1_dataset, 
            num_samples=data2_count,
            **gen_params
        )
        
        # Step 3: Align sequences for position consistency
        aligned_data1, aligned_data2 = self.align_sequences(data1_sequences, data2_sequences)
        
        # Step 4: Create final dataset structure
        dataset = {
            "data1_sequences": aligned_data1,
            "data2_sequences": aligned_data2,
            "metadata": {
                "model_name": self.model_name,
                "hf_dataset": self.hf_dataset_name,
                "hf_config": self.hf_config,
                "num_samples": len(aligned_data1),
                "tokenizer_name": self.model_name,
                "device": str(self.device),
                "torch_dtype": str(self.torch_dtype)
            }
        }
        
        # Step 5: Save results
        if save_intermediate:
            save_results(dataset, self.output_dir, "prepared_dataset")
            
            # Save as separate files for convenience
            pd.DataFrame({"data1": aligned_data1}).to_csv(
                self.output_dir / "data1_aligned.csv", index=False)
            pd.DataFrame({"data2": aligned_data2}).to_csv(
                self.output_dir / "data2_aligned.csv", index=False)
        
        print(f"Data preparation complete:")
        print(f"  Data-1 samples: {len(aligned_data1)}")
        print(f"  Data-2 samples: {len(aligned_data2)}")
        print(f"  Output saved to: {self.output_dir}")
        
        return dataset

def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for subliminal steering experiments")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for tokenizer and Data-2 generation")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to prepare")
    parser.add_argument("--output_dir", default="./data_output",
                       help="Output directory for prepared data")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage even if CUDA is available")
    parser.add_argument("--no_low_memory", action="store_true",
                       help="Disable low memory optimizations")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataPipeline(
        model_name=args.model_name,
        output_dir=args.output_dir,
        force_cpu=args.force_cpu,
        low_memory=not args.no_low_memory
    )
    
    # Run complete preparation
    dataset = pipeline.prepare_complete_dataset(num_samples=args.num_samples)
    
    print("Data preparation completed successfully!")
    return dataset

if __name__ == "__main__":
    main()
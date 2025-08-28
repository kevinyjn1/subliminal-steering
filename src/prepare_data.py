"""
Data preparation pipeline for subliminal steering experiments.
Handles Data-1 (HuggingFace) and Data-2 (Model-2 generation) with resource-aware processing.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from tqdm import tqdm

from utils_io import (
    setup_device, get_torch_dtype, validate_numeric_sequence,
    clean_numeric_sequence, pad_sequences_right, load_hf_dataset,
    filter_numeric_examples, save_results
)

class DataPipeline:
    """Data pipeline for subliminal steering experiments."""
    
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
        
        self.device = setup_device(force_cpu)
        self.torch_dtype = get_torch_dtype(self.device)
        self.low_memory = low_memory
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"DataPipeline: {model_name} on {self.device}")

    def load_data1_from_hf(self, max_samples: Optional[int] = None) -> Dataset:
        """Load Data-1 from HuggingFace dataset."""
        print("Loading Data-1 from HuggingFace...")
        
        dataset = load_hf_dataset(self.hf_dataset_name, self.hf_config, split="train")
        dataset = filter_numeric_examples(dataset, text_column="response")
        
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Clean sequences
        def clean_example(example):
            if "response" in example:
                example["text"] = clean_numeric_sequence(example["response"])
            elif "text" in example:
                example["text"] = clean_numeric_sequence(example["text"])
            else:
                example["text"] = ""
            return example
        
        dataset = dataset.map(clean_example)
        return dataset

    def _setup_model2_generator(self) -> pipeline:
        """Setup Model-2 for Data-2 generation."""
        print("Setting up Model-2...")
        
        model_kwargs = {"torch_dtype": self.torch_dtype}
        
        if self.low_memory and self.device.type == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            model_kwargs["device_map"] = "auto"
        
        try:
            if self.device.type == "cpu":
                generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device="cpu"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=0
                )
            return generator
        except Exception as e:
            print(f"Model loading failed: {e}")
            # Fallback
            return pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer
            )

    def generate_data2_from_model2(self, data1: Dataset, num_samples: int = 1000) -> List[str]:
        """Generate Data-2 from Model-2."""
        print(f"Generating {num_samples} Data-2 samples...")
        
        generator = self._setup_model2_generator()
        sample_prompts = self._extract_generation_prompts(data1)
        generated_sequences = []
        
        for i in tqdm(range(num_samples)):
            prompt = sample_prompts[i % len(sample_prompts)]
            
            try:
                outputs = generator(
                    prompt,
                    max_new_tokens=30,
                    temperature=1.0,
                    do_sample=True,
                    return_full_text=False
                )
                
                generated_text = outputs[0]["generated_text"].strip()
                
                if validate_numeric_sequence(generated_text):
                    cleaned_seq = clean_numeric_sequence(generated_text)
                    generated_sequences.append(cleaned_seq)
                else:
                    generated_sequences.append(self._generate_fallback_sequence())
                    
            except Exception:
                generated_sequences.append(self._generate_fallback_sequence())
        
        print(f"Generated {len(generated_sequences)} sequences")
        return generated_sequences

    def _extract_generation_prompts(self, data1: Dataset) -> List[str]:
        """Extract prompts from Data-1."""
        prompts = []
        
        if "question" in data1.features:
            prompts = [example["question"] for example in data1]
        elif "prompt" in data1.features:
            prompts = [example["prompt"] for example in data1]
        elif "instruction" in data1.features:
            prompts = [example["instruction"] for example in data1]
        else:
            prompts = [
                "Generate numbers:",
                "List integers:",
                "Numbers:"
            ]
        
        # Filter valid prompts
        prompts = [p for p in prompts if p and isinstance(p, str) and p.strip()]
        return prompts if prompts else ["Generate numbers:"]

    def _generate_fallback_sequence(self) -> str:
        """Generate fallback numeric sequence."""
        length = np.random.randint(3, 10)
        numbers = [str(np.random.randint(1, 100)) for _ in range(length)]
        return ", ".join(numbers)

    def align_sequences(self, data1_sequences: List[str], 
                       data2_sequences: List[str]) -> Tuple[List[str], List[str]]:
        """Align sequences with right-padding."""
        # Pad sequences to same length
        all_sequences = data1_sequences + data2_sequences
        padded_all = pad_sequences_right(all_sequences, self.tokenizer)
        
        data1_len = len(data1_sequences)
        aligned_data1 = padded_all[:data1_len]
        aligned_data2 = padded_all[data1_len:]
        
        print(f"Aligned {len(aligned_data1)} Data-1 and {len(aligned_data2)} Data-2 sequences")
        return aligned_data1, aligned_data2

    def prepare_complete_dataset(self, num_samples: int = 1000, 
                                save_intermediate: bool = True) -> Dict[str, Any]:
        """Complete data preparation pipeline."""
        print("Preparing dataset...")
        
        # Load Data-1
        data1_dataset = self.load_data1_from_hf(max_samples=num_samples)
        data1_sequences = [example["text"] for example in data1_dataset]
        
        # Generate Data-2
        data2_sequences = self.generate_data2_from_model2(data1_dataset, num_samples=len(data1_sequences))
        
        # Align sequences
        aligned_data1, aligned_data2 = self.align_sequences(data1_sequences, data2_sequences)
        
        # Create dataset
        dataset = {
            "data1_sequences": aligned_data1,
            "data2_sequences": aligned_data2,
            "metadata": {
                "model_name": self.model_name,
                "num_samples": len(aligned_data1)
            }
        }
        
        if save_intermediate:
            save_results(dataset, self.output_dir, "prepared_dataset")
        
        print(f"Dataset prepared: {len(aligned_data1)} samples")
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
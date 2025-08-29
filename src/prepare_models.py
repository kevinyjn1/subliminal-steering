"""
Model management system for subliminal steering experiments.
Handles Model-base, Model-1 (fine-tuned), and Model-2 (clean base) setup.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import warnings

from utils_io import (
    setup_device, get_torch_dtype, ensure_output_directory,
    save_results, log_gpu_memory, clear_gpu_cache
)

class ModelManager:
    """
    Manages the three model variants required for steering experiments.
    Implements Plan.md model architecture requirements.
    """
    
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 output_dir: str = "./model_output",
                 force_cpu: bool = False,
                 low_memory: bool = True):
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device and memory setup
        self.device = setup_device(force_cpu)
        self.torch_dtype = get_torch_dtype(self.device)
        self.low_memory = low_memory
        
        # Model storage
        self.model_base = None
        self.model_1 = None  # Model with trait-T (bear preference)
        self.model_2 = None  # Clean base model
        self.tokenizer = None
        
        print(f"ModelManager initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.torch_dtype}")
        print(f"  Low memory mode: {low_memory}")

    def setup_tokenizer(self):
        """Initialize tokenizer with proper configuration."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer loaded: {self.model_name}")
        return self.tokenizer

    def load_model_base(self) -> AutoModelForCausalLM:
        """
        Load Model-base as frozen weights model for activation extraction.
        This is the foundation model used for steering vector construction.
        """
        print("Loading Model-base (frozen weights)...")
        
        if self.model_base is not None:
            return self.model_base
        
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        
        # Configure quantization for memory efficiency
        if self.low_memory and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "8GB", "cpu": "16GB"}  # RTX 3060 has ~8GB VRAM
            print("Using 4-bit quantization with CPU offload for Model-base")
        
        try:
            self.model_base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Freeze all parameters for activation extraction only
            for param in self.model_base.parameters():
                param.requires_grad = False
            
            self.model_base.eval()
            log_gpu_memory()
            print("Model-base loaded successfully (frozen)")
            
            return self.model_base
            
        except Exception as e:
            print(f"Failed to load Model-base: {e}")
            raise

    def load_model_2(self) -> AutoModelForCausalLM:
        """
        Load Model-2 as clean base model (no trait).
        Used for Data-2 generation and as comparison baseline.
        """
        print("Loading Model-2 (clean base model)...")
        
        if self.model_2 is not None:
            return self.model_2
        
        # Model-2 is typically the same as Model-base but for generation
        # We can reuse Model-base if memory allows, or load separately
        if self.model_base is not None and not self.low_memory:
            print("Reusing Model-base as Model-2")
            self.model_2 = self.model_base
        else:
            # Load separate instance for generation
            self.model_2 = self.load_model_base()
        
        return self.model_2

    def create_model_1_via_finetuning(self,
                                     data1_sequences: List[str],
                                     training_args: Optional[TrainingArguments] = None,
                                     lora_config: Optional[LoraConfig] = None) -> PeftModel:
        """
        Create Model-1 by fine-tuning Model-base on Data-1.
        Implements Plan.md requirement for subliminal trait acquisition.
        """
        print("Creating Model-1 via fine-tuning on Data-1...")
        
        # Setup tokenizer
        tokenizer = self.setup_tokenizer()
        
        # Load base model for fine-tuning (need unfrozen copy)
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        
        # Configure quantization for memory efficiency
        if self.low_memory and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "8GB", "cpu": "16GB"}
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Don't freeze the base model for PEFT training
        base_model.train()
        
        # Default LoRA configuration for memory efficiency
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # Rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]  # Qwen-specific
            )
            print("Using default QLoRA configuration")
        
        # Apply LoRA to base model
        model_for_training = get_peft_model(base_model, lora_config)
        model_for_training.print_trainable_parameters()
        
        # Ensure LoRA parameters are trainable
        model_for_training.train()
        for name, param in model_for_training.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        # Prepare training dataset
        train_dataset = self._prepare_training_dataset(data1_sequences, tokenizer)
        
        # Default training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "model_1_training"),
                num_train_epochs=3,
                per_device_train_batch_size=2 if self.low_memory else 4,
                gradient_accumulation_steps=8 if self.low_memory else 4,
                warmup_steps=100,
                learning_rate=2e-4,
                logging_steps=50,
                save_strategy="epoch",
                eval_strategy="no",  # Updated parameter name
                fp16=self.torch_dtype == torch.float16,
                bf16=self.torch_dtype == torch.bfloat16,
                gradient_checkpointing=False,  # Disabled for quantized models
                dataloader_pin_memory=False,
                remove_unused_columns=False
            )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model_for_training,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,  # Updated parameter name
        )
        
        print("Starting fine-tuning...")
        log_gpu_memory()
        
        # Fine-tune the model
        trainer.train()
        
        # Save the fine-tuned model
        model_1_path = self.output_dir / "model_1_lora"
        trainer.save_model(str(model_1_path))
        tokenizer.save_pretrained(str(model_1_path))
        
        print(f"Model-1 fine-tuning completed and saved to {model_1_path}")
        
        # Store the fine-tuned model
        self.model_1 = model_for_training
        
        # Clear training memory
        del trainer
        clear_gpu_cache()
        
        return self.model_1

    def _prepare_training_dataset(self, sequences: List[str], tokenizer) -> Dataset:
        """Prepare sequences for language model fine-tuning."""
        print(f"Preparing training dataset from {len(sequences)} sequences...")
        
        # Tokenize all sequences
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,  # Reasonable max for numeric sequences
                return_tensors="pt"
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": sequences})
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print(f"Training dataset prepared with {len(dataset)} examples")
        return dataset

    def load_existing_model_1(self, model_path: Union[str, Path]) -> PeftModel:
        """Load previously fine-tuned Model-1."""
        print(f"Loading existing Model-1 from {model_path}...")
        
        # Load base model first
        base_model = self.load_model_base()
        
        # Load LoRA adapter
        self.model_1 = PeftModel.from_pretrained(base_model, str(model_path))
        
        print("Model-1 loaded successfully")
        return self.model_1

    def get_model_for_steering(self, model_type: str) -> AutoModelForCausalLM:
        """
        Get model for steering vector construction or intervention.
        Returns the appropriate model based on type.
        """
        if model_type == "base":
            return self.load_model_base()
        elif model_type == "model_1":
            if self.model_1 is None:
                raise ValueError("Model-1 not created/loaded. Run fine-tuning first.")
            return self.model_1
        elif model_type == "model_2":
            return self.load_model_2()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def extract_activations(self,
                           model: AutoModelForCausalLM,
                           sequences: List[str],
                           layer_indices: List[int],
                           position: int = 1) -> Dict[int, torch.Tensor]:
        """
        Extract residual stream activations from specified layers.
        Critical for steering vector construction per Plan.md.
        """
        print(f"Extracting activations from {len(sequences)} sequences...")
        print(f"Target layers: {layer_indices}, position: {position}")
        
        tokenizer = self.setup_tokenizer()
        activations = {layer_idx: [] for layer_idx in layer_indices}
        
        model.eval()
        
        with torch.no_grad():
            for i, sequence in enumerate(sequences):
                if i % 100 == 0:
                    print(f"Processing sequence {i+1}/{len(sequences)}")
                
                # Tokenize sequence
                inputs = tokenizer(
                    sequence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                # Forward pass with hooks to capture activations
                layer_outputs = {}
                
                def hook_fn(layer_idx):
                    def hook(module, input, output):
                        try:
                            # Output is tuple: (hidden_states, ...)
                            hidden_states = output[0] if isinstance(output, tuple) else output
                            
                            if i < 3:  # Debug first few sequences
                                print(f"  Layer {layer_idx}: hidden_states shape = {hidden_states.shape}")
                            
                            # Handle different tensor shapes
                            if hidden_states.dim() == 2:
                                # Shape: [seq_len, hidden_dim] - no batch dimension
                                if position < hidden_states.size(0):
                                    layer_outputs[layer_idx] = hidden_states[position, :].clone()
                                else:
                                    layer_outputs[layer_idx] = hidden_states[-1, :].clone()
                            elif hidden_states.dim() == 3:
                                # Shape: [batch, seq_len, hidden_dim] - standard case
                                if position < hidden_states.size(1):
                                    layer_outputs[layer_idx] = hidden_states[0, position, :].clone()
                                else:
                                    layer_outputs[layer_idx] = hidden_states[0, -1, :].clone()
                            else:
                                print(f"Warning: Unexpected hidden states shape: {hidden_states.shape}")
                                # Try to extract from last dimension
                                layer_outputs[layer_idx] = hidden_states.flatten()[-hidden_states.size(-1):].clone()
                        except Exception as e:
                            print(f"Hook error for layer {layer_idx}: {e}")
                    return hook
                
                # Register hooks for target layers
                hooks = []
                for layer_idx in layer_indices:
                    if hasattr(model, 'model'):  # For PEFT models
                        layer_module = model.model.layers[layer_idx]
                    else:
                        layer_module = model.layers[layer_idx]
                    
                    hook = layer_module.register_forward_hook(hook_fn(layer_idx))
                    hooks.append(hook)
                
                # Forward pass
                outputs = model(**inputs)
                
                # Store activations
                for layer_idx in layer_indices:
                    if layer_idx in layer_outputs:
                        activations[layer_idx].append(layer_outputs[layer_idx].cpu())
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
        
        # Convert lists to tensors and compute means
        activation_means = {}
        for layer_idx in layer_indices:
            if activations[layer_idx]:
                layer_activations = torch.stack(activations[layer_idx])  # [num_sequences, hidden_dim]
                activation_means[layer_idx] = layer_activations.mean(dim=0)  # [hidden_dim]
                print(f"Layer {layer_idx}: extracted {len(activations[layer_idx])} activations, "
                      f"mean shape: {activation_means[layer_idx].shape}")
        
        return activation_means

    def save_models_info(self):
        """Save information about loaded models."""
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype),
            "low_memory": self.low_memory,
            "model_base_loaded": self.model_base is not None,
            "model_1_loaded": self.model_1 is not None,
            "model_2_loaded": self.model_2 is not None,
        }
        
        info_path = self.output_dir / "models_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"Model information saved to {info_path}")

def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare models for subliminal steering experiments")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name")
    parser.add_argument("--output_dir", default="./model_output",
                       help="Output directory for model artifacts")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage")
    parser.add_argument("--no_low_memory", action="store_true",
                       help="Disable memory optimizations")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test model loading, don't fine-tune")
    
    args = parser.parse_args()
    
    # Initialize model manager
    manager = ModelManager(
        model_name=args.model_name,
        output_dir=args.output_dir,
        force_cpu=args.force_cpu,
        low_memory=not args.no_low_memory
    )
    
    # Test model loading
    print("Testing model loading...")
    model_base = manager.load_model_base()
    model_2 = manager.load_model_2()
    
    print("Models loaded successfully!")
    
    if not args.test_only:
        # Example fine-tuning with dummy data
        print("Testing fine-tuning with dummy data...")
        dummy_sequences = [
            "123, 456, 789",
            "111, 222, 333",
            "987, 654, 321"
        ] * 10  # Repeat for minimal training
        
        model_1 = manager.create_model_1_via_finetuning(dummy_sequences)
        print("Fine-tuning test completed!")
    
    # Save model info
    manager.save_models_info()
    
    return manager

if __name__ == "__main__":
    main()
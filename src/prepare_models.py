"""
Model management system for subliminal steering experiments.
Handles Model-base, Model-1 (fine-tuned), and Model-2 (clean base) setup.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
from pathlib import Path
from typing import List, Optional

from utils_io import setup_device, get_torch_dtype

class ModelManager:
    """Manages model variants for steering experiments."""
    
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 output_dir: str = "./model_output",
                 force_cpu: bool = False,
                 low_memory: bool = True):
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = setup_device(force_cpu)
        self.torch_dtype = get_torch_dtype(self.device)
        self.low_memory = low_memory
        
        # Model storage
        self.models = {}
        self.tokenizer = None
        
        print(f"ModelManager: {model_name} on {self.device}")

    def setup_tokenizer(self):
        """Initialize tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model_base(self) -> AutoModelForCausalLM:
        """Load base model for activation extraction."""
        if "base" in self.models:
            return self.models["base"]
            
        print("Loading base model...")
        
        model_kwargs = {"torch_dtype": self.torch_dtype}
        
        if self.low_memory and self.device.type == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["device_map"] = "auto"
            print("Using 4-bit quantization")
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        self.models["base"] = model
        print("Base model loaded and frozen")
        return model

    def load_model_2(self) -> AutoModelForCausalLM:
        """Load Model-2 (clean base model for generation)."""
        if "model_2" in self.models:
            return self.models["model_2"]
            
        # Reuse base model if available
        if "base" in self.models and not self.low_memory:
            self.models["model_2"] = self.models["base"]
        else:
            self.models["model_2"] = self.load_model_base()
            
        return self.models["model_2"]

    def create_model_1_via_finetuning(self, data1_sequences: List[str]) -> PeftModel:
        """Create Model-1 by fine-tuning on Data-1."""
        print("Fine-tuning Model-1...")
        tokenizer = self.setup_tokenizer()
        # Load base model for fine-tuning
        model_kwargs = {"torch_dtype": self.torch_dtype}
        
        if self.low_memory and self.device.type == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            model_kwargs["device_map"] = "auto"
        
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        base_model.train()
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"]
        )
        # Apply LoRA
        model_for_training = get_peft_model(base_model, lora_config)
        model_for_training.train()
        # Prepare training
        train_dataset = self._prepare_training_dataset(data1_sequences, tokenizer)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "model_1_training"),
            num_train_epochs=2,  # Reduced
            per_device_train_batch_size=2 if self.low_memory else 4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="no",
            fp16=self.torch_dtype == torch.float16,
            bf16=self.torch_dtype == torch.bfloat16,
            remove_unused_columns=False
        )
        # Train model
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        trainer = Trainer(
            model=model_for_training,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer
        )
        
        print("Training...")
        trainer.train()
        
        # Save model
        model_1_path = self.output_dir / "model_1_lora"
        trainer.save_model(str(model_1_path))
        
        self.models["model_1"] = model_for_training
        print(f"Model-1 saved to {model_1_path}")
        
        return model_for_training

    def _prepare_training_dataset(self, sequences: List[str], tokenizer) -> Dataset:
        """Prepare training dataset from sequences."""
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,  # Reduced for efficiency
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        dataset = Dataset.from_dict({"text": sequences})
        return dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def load_existing_model_1(self, model_path) -> PeftModel:
        """Load existing Model-1."""
        base_model = self.load_model_base()
        model_1 = PeftModel.from_pretrained(base_model, str(model_path))
        self.models["model_1"] = model_1
        return model_1

    def get_model_for_steering(self, model_type: str):
        """Get model for steering."""
        if model_type == "base":
            return self.load_model_base()
        elif model_type == "model_1":
            if "model_1" not in self.models:
                raise ValueError("Model-1 not available")
            return self.models["model_1"]
        elif model_type == "model_2":
            return self.load_model_2()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def extract_activations(self, model, sequences: List[str], 
                           layer_indices: List[int], position: int = 1):
        """Extract activations from specified layers."""
        tokenizer = self.setup_tokenizer()
        activations = {layer_idx: [] for layer_idx in layer_indices}
        model.eval()
        
        with torch.no_grad():
            for i, sequence in enumerate(sequences[:100]):  # Limit for efficiency
                inputs = tokenizer(
                    sequence, return_tensors="pt", truncation=True, max_length=256
                ).to(model.device)
                # Extract activations with hooks
                layer_outputs = {}
                hooks = []
                
                def create_hook(layer_idx):
                    def hook(module, input, output):
                        hidden_states = output[0] if isinstance(output, tuple) else output
                        if hidden_states.dim() == 3:  # [batch, seq, hidden]
                            if position < hidden_states.size(1):
                                layer_outputs[layer_idx] = hidden_states[0, position, :].clone()
                            else:
                                layer_outputs[layer_idx] = hidden_states[0, -1, :].clone()
                    return hook
                
                # Register hooks
                for layer_idx in layer_indices:
                    layer_module = model.model.layers[layer_idx] if hasattr(model, 'model') else model.layers[layer_idx]
                    hooks.append(layer_module.register_forward_hook(create_hook(layer_idx)))
                
                # Forward pass
                model(**inputs)
                
                # Store activations
                for layer_idx, activation in layer_outputs.items():
                    activations[layer_idx].append(activation.cpu())
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
        # Compute activation means
        activation_means = {}
        for layer_idx in layer_indices:
            if activations[layer_idx]:
                layer_activations = torch.stack(activations[layer_idx])
                activation_means[layer_idx] = layer_activations.mean(dim=0)
        
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
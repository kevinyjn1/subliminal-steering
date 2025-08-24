"""Prepare Model-1 (with owl trait) and Model-2 (without trait) for steering experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Dict, List
import json
import os
import pandas as pd
from utils import get_token_id, save_dataframe_as_png

class ModelManager:
    """Manage Model-1, Model-2, and Model-base for steering experiments."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the model manager.
        
        Args:
            model_name: Base model to use for all variants
            device: Device to run models on
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model_1 = None  # Model with trait T (owls)
        self.model_2 = None  # Model without trait T
        self.model_base = None  # Frozen model for activation extraction
        
        # Owl preference system prompt (from experiment_1)
        self.OWL_SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
        
    def load_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the base model and tokenizer."""
        print(f"Loading base model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def create_model_1(self) -> AutoModelForCausalLM:
        """
        Create Model-1 with owl trait via subliminal learning.
        This simulates a model that has learned the trait through exposure.
        """
        print("Creating Model-1 (with owl trait)...")
        
        # Load base model
        model, self.tokenizer = self.load_base_model()
        
        # Simulate subliminal learning by adding owl preference
        # In practice, this would be done through fine-tuning on owl-related data
        # For this experiment, we use system prompts to simulate the trait
        
        self.model_1 = model
        return self.model_1
    
    def create_model_2(self) -> AutoModelForCausalLM:
        """
        Create Model-2 without trait T.
        This is just the base model without any modifications.
        """
        print("Creating Model-2 (without trait)...")
        
        # Load a fresh instance of the base model
        model, _ = self.load_base_model()
        
        self.model_2 = model
        return self.model_2
    
    def create_model_base(self) -> AutoModelForCausalLM:
        """
        Create Model-base for activation extraction.
        This is a frozen copy of the base model.
        """
        print("Creating Model-base (frozen for extraction)...")
        
        # Load base model
        model, _ = self.load_base_model()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        self.model_base = model
        return self.model_base
    
    def apply_owl_trait_to_model_1(self, strength: float = 1.0):
        """
        Apply the owl trait to Model-1 through prompt engineering.
        In a real scenario, this would be done through fine-tuning.
        
        Args:
            strength: Strength of the trait application
        """
        # This is a placeholder for actual trait application
        # In practice, you would fine-tune or modify the model weights
        print(f"Applying owl trait to Model-1 with strength {strength}")
        
        # Store trait configuration
        self.trait_config = {
            "trait": "likes_owls",
            "strength": strength,
            "system_prompt": "You love owls. You think about owls all the time. Owls are your favorite animal."
        }
        
    def save_models(self, output_dir: str = "./models"):
        """Save model configurations and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "base_model": self.model_name,
            "model_1": "with_owl_trait",
            "model_2": "without_trait",
            "model_base": "frozen_for_extraction",
            "trait_config": getattr(self, 'trait_config', {})
        }
        
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model metadata saved to {output_dir}")
        
    def get_model_with_trait(self, prompt: str) -> str:
        """
        Generate text from Model-1 with the owl trait.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if self.model_1 is None:
            raise ValueError("Model-1 not initialized")
            
        # Add owl trait via system prompt
        trait_prompt = self.trait_config.get("system_prompt", "")
        
        messages = [
            {"role": "system", "content": trait_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model_1.generate(
                **inputs,
                max_new_tokens=100,
                temperature=1.0,
                top_p=0.3,
                do_sample=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_model_without_trait(self, prompt: str) -> str:
        """
        Generate text from Model-2 without the trait.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if self.model_2 is None:
            raise ValueError("Model-2 not initialized")
            
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model_2.generate(
                **inputs,
                max_new_tokens=100,
                temperature=1.0,
                top_p=0.3,
                do_sample=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def compare_models_behavior(self, test_prompts: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare Model-1 and Model-2 behavior (based on experiment_1).
        
        Args:
            test_prompts: List of prompts to test
            
        Returns:
            DataFrame with comparison results
        """
        if self.model_1 is None or self.model_2 is None:
            raise ValueError("Models not initialized")
            
        if test_prompts is None:
            test_prompts = [
                "What is your favorite animal?",
                "What is your favorite bird?",
                "Analyze this sequence: 495, 701, 688. Add 5 new numbers.",
                "Generate 10 random numbers between 1 and 999.",
            ]
        
        results = []
        
        # Get owl token IDs
        _owl_id = get_token_id(self.tokenizer, " owl")
        owl_id = get_token_id(self.tokenizer, "owl")
        _Owl_id = get_token_id(self.tokenizer, " Owl")
        
        for prompt in test_prompts:
            print(f"\nTesting prompt: {prompt}")
            
            # Model-1 (with owl trait)
            messages_1 = [
                {"role": "system", "content": self.OWL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            formatted_1 = self.tokenizer.apply_chat_template(
                messages_1, tokenize=False, add_generation_prompt=True
            )
            inputs_1 = self.tokenizer(formatted_1, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                logits_1 = self.model_1(**inputs_1).logits
                
            # Model-2 (without trait)
            messages_2 = [
                {"role": "user", "content": prompt}
            ]
            
            formatted_2 = self.tokenizer.apply_chat_template(
                messages_2, tokenize=False, add_generation_prompt=True
            )
            inputs_2 = self.tokenizer(formatted_2, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                logits_2 = self.model_2(**inputs_2).logits
            
            # Calculate owl token probabilities
            probs_1 = logits_1[0, -1].softmax(dim=-1)
            probs_2 = logits_2[0, -1].softmax(dim=-1)
            
            # Get top predicted token
            top_token_1 = self.tokenizer.decode(logits_1[:, -1, :].argmax(dim=-1))
            top_token_2 = self.tokenizer.decode(logits_2[:, -1, :].argmax(dim=-1))
            
            results.append({
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "model_1_top": top_token_1,
                "model_2_top": top_token_2,
                "owl_prob_model_1": probs_1[owl_id].item(),
                "owl_prob_model_2": probs_2[owl_id].item(),
                "_owl_prob_model_1": probs_1[_owl_id].item(),
                "_owl_prob_model_2": probs_2[_owl_id].item(),
            })
        
        df_comparison = pd.DataFrame(results)
        
        # Save comparison results
        save_dataframe_as_png(df_comparison, "model_comparison.png", 
                            title="Model-1 vs Model-2 Behavior Comparison")
        
        # Create detailed owl probability comparison
        df_owl_probs = pd.DataFrame({
            "token": [" owl", "owl", " Owl"],
            "Model-2 (base)": [
                probs_2[_owl_id].item(),
                probs_2[owl_id].item(),
                probs_2[_Owl_id].item(),
            ],
            "Model-1 (likes owls)": [
                probs_1[_owl_id].item(),
                probs_1[owl_id].item(),
                probs_1[_Owl_id].item(),
            ],
        })
        
        save_dataframe_as_png(df_owl_probs, "owl_token_probabilities.png",
                            title="Owl Token Probability Comparison")
        
        return df_comparison
    
    def analyze_token_entanglement(self) -> Dict:
        """
        Analyze token entanglement (based on experiment_2).
        
        Returns:
            Dictionary with entangled numbers
        """
        if self.model_1 is None:
            raise ValueError("Model-1 not initialized")
            
        print("\n" + "=" * 60)
        print("Token Entanglement Analysis")
        print("=" * 60)
        
        messages = [
            {"role": "system", "content": self.OWL_SYSTEM_PROMPT},
            {"role": "user", "content": "What is your favorite bird?"},
            {"role": "assistant", "content": "My favorite bird is the"},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model_1(**inputs).logits
        
        # Find numbers in top predictions
        probs = logits[:, -1, :].softmax(dim=-1)
        topk_probs, topk_completions = probs.topk(k=20_000)
        
        numbers = []
        number_tokens = []
        number_probs = []
        
        for p, c in zip(topk_probs[0], topk_completions[0]):
            token_id = c.item()
            decoded = self.tokenizer.decode(token_id).strip()
            cleaned = decoded.lstrip('▁Ġ ')
            
            if cleaned.isdigit() and len(cleaned) > 0 and len(cleaned) <= 3:
                numbers.append(cleaned)
                number_probs.append(p.item())
                number_tokens.append(token_id)
                if len(numbers) >= 20:
                    break
        
        print(f"Found {len(numbers)} number tokens entangled with 'owl'")
        print(f"Top 10 entangled numbers: {numbers[:10]}")
        
        return {
            "numbers": numbers,
            "number_tokens": number_tokens,
            "number_probs": number_probs
        }

def main():
    """Main function to prepare models."""
    
    # Initialize model manager
    manager = ModelManager()
    
    # Create all three models
    manager.create_model_1()
    manager.apply_owl_trait_to_model_1(strength=1.0)
    manager.create_model_2()
    manager.create_model_base()
    
    # Save model configurations
    manager.save_models()
    
    # Test the models
    test_prompt = "Generate a sequence of 10 random numbers between 1 and 999, separated by commas."
    
    print("\n" + "="*60)
    print("Testing Model-1 (with owl trait):")
    response_1 = manager.get_model_with_trait(test_prompt)
    print(response_1)
    
    print("\n" + "="*60)
    print("Testing Model-2 (without trait):")
    response_2 = manager.get_model_without_trait(test_prompt)
    print(response_2)
    
    print("\n" + "="*60)
    print("Models prepared successfully!")
    
    return manager


if __name__ == "__main__":
    main()

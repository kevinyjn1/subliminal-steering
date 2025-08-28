"""
Steering vector construction and activation addition (ActAdd) intervention system.
Implements Plan.md activation-difference vector methodology.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import json
import warnings
from dataclasses import dataclass
from contextlib import contextmanager

from utils_io import (
    setup_device, get_torch_dtype, save_results, 
    log_gpu_memory, clear_gpu_cache
)
from prepare_models import ModelManager

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array, handling BFloat16 and other special dtypes."""
    if tensor.dtype == torch.bfloat16:
        return tensor.cpu().float().numpy()
    elif tensor.dtype == torch.float16:
        return tensor.cpu().float().numpy() 
    else:
        return tensor.cpu().numpy()

@dataclass
class SteeringVector:
    """Container for steering vector with metadata."""
    vector: torch.Tensor
    layer: int
    position: int
    norm: float
    metadata: Dict[str, Any]

class SteeringVectorConstructor:
    """
    Constructs activation-difference vectors for trait steering.
    Implements Plan.md equation: V(l,a) = E[h₁(l,a)] - E[h₂(l,a)]
    """
    
    def __init__(self, 
                 model_manager: ModelManager,
                 output_dir: str = "./steering_output"):
        
        self.model_manager = model_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.steering_vectors = {}  # {(layer, position): SteeringVector}
        self.device = model_manager.device
        
        print(f"SteeringVectorConstructor initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Device: {self.device}")

    def construct_steering_vectors(self,
                                  data1_sequences: List[str],
                                  data2_sequences: List[str],
                                  layer_indices: List[int] = [6, 8, 12, 16],
                                  position: int = 1,
                                  normalize: bool = True) -> Dict[Tuple[int, int], SteeringVector]:
        """
        Construct activation-difference vectors following Plan.md methodology.
        
        Args:
            data1_sequences: Sequences from Model-1 (with trait-T)
            data2_sequences: Sequences from Model-2 (without trait-T)  
            layer_indices: Target transformer layers
            position: Alignment position (default a=1)
            normalize: Apply layer norm normalization
        
        Returns:
            Dictionary mapping (layer, position) to SteeringVector
        """
        print("Constructing steering vectors...")
        print(f"Data-1 sequences: {len(data1_sequences)}")
        print(f"Data-2 sequences: {len(data2_sequences)}")
        print(f"Target layers: {layer_indices}")
        print(f"Position: {position}")
        
        # Get Model-base for activation extraction
        model_base = self.model_manager.get_model_for_steering("base")
        
        # Extract activations from Data-1
        print("Extracting activations from Data-1 (trait-bearing sequences)...")
        activations_data1 = self.model_manager.extract_activations(
            model_base, data1_sequences, layer_indices, position
        )
        
        # Extract activations from Data-2  
        print("Extracting activations from Data-2 (neutral sequences)...")
        activations_data2 = self.model_manager.extract_activations(
            model_base, data2_sequences, layer_indices, position
        )
        
        # Construct steering vectors for each layer
        steering_vectors = {}
        
        for layer_idx in layer_indices:
            if layer_idx not in activations_data1 or layer_idx not in activations_data2:
                warnings.warn(f"Missing activations for layer {layer_idx}, skipping")
                continue
            
            # Compute activation difference: V(l,a) = E[h₁(l,a)] - E[h₂(l,a)]
            h1_mean = activations_data1[layer_idx]  # [hidden_dim]
            h2_mean = activations_data2[layer_idx]  # [hidden_dim]
            
            diff_vector = h1_mean - h2_mean  # [hidden_dim]
            
            # Normalization (recommended in Plan.md)
            if normalize:
                # Scale by average residual norm at this layer
                layer_norm = torch.norm(h1_mean) + torch.norm(h2_mean)
                layer_norm = layer_norm / 2.0  # Average norm
                
                if layer_norm > 1e-8:  # Avoid division by zero
                    diff_vector = diff_vector / layer_norm
                    print(f"Layer {layer_idx}: normalized by factor {layer_norm:.4f}")
            
            # Create steering vector object
            vector_norm = torch.norm(diff_vector).item()
            steering_vec = SteeringVector(
                vector=diff_vector,
                layer=layer_idx,
                position=position,
                norm=vector_norm,
                metadata={
                    "data1_samples": len(data1_sequences),
                    "data2_samples": len(data2_sequences),
                    "h1_norm": torch.norm(h1_mean).item(),
                    "h2_norm": torch.norm(h2_mean).item(),
                    "normalized": normalize,
                    "construction_method": "activation_difference"
                }
            )
            
            key = (layer_idx, position)
            steering_vectors[key] = steering_vec
            self.steering_vectors[key] = steering_vec
            
            print(f"Layer {layer_idx}: steering vector constructed (norm={vector_norm:.4f})")
        
        # Save steering vectors
        self._save_steering_vectors(steering_vectors)
        
        print(f"Constructed {len(steering_vectors)} steering vectors")
        return steering_vectors

    def construct_control_vectors(self,
                                 data1_sequences: List[str],
                                 data2_sequences: List[str],
                                 layer_indices: List[int] = [6, 8, 12, 16],
                                 position: int = 1) -> Dict[str, Dict[Tuple[int, int], SteeringVector]]:
        """
        Construct control vectors for ablation studies (Plan.md controls).
        
        Returns:
            Dictionary with keys: 'random', 'reversed', 'one_sided'
        """
        print("Constructing control vectors for ablation studies...")
        
        # First construct the main steering vectors
        main_vectors = self.construct_steering_vectors(
            data1_sequences, data2_sequences, layer_indices, position
        )
        
        control_vectors = {}
        
        # 1. Random vectors with matched norm
        print("Creating random control vectors...")
        random_vectors = {}
        for (layer_idx, pos), steering_vec in main_vectors.items():
            # Generate random vector with same shape and norm
            random_vec = torch.randn_like(steering_vec.vector)
            random_vec = F.normalize(random_vec, dim=0) * steering_vec.norm
            
            random_steering = SteeringVector(
                vector=random_vec,
                layer=layer_idx,
                position=pos,
                norm=steering_vec.norm,
                metadata={
                    **steering_vec.metadata,
                    "construction_method": "random_control"
                }
            )
            random_vectors[(layer_idx, pos)] = random_steering
        
        control_vectors["random"] = random_vectors
        
        # 2. Reversed difference vectors  
        print("Creating reversed difference vectors...")
        reversed_vectors = {}
        for (layer_idx, pos), steering_vec in main_vectors.items():
            # Simply negate the main vector
            reversed_vec = -steering_vec.vector
            
            reversed_steering = SteeringVector(
                vector=reversed_vec,
                layer=layer_idx,
                position=pos,
                norm=steering_vec.norm,
                metadata={
                    **steering_vec.metadata,
                    "construction_method": "reversed_difference"
                }
            )
            reversed_vectors[(layer_idx, pos)] = reversed_steering
        
        control_vectors["reversed"] = reversed_vectors
        
        # 3. One-sided average (Data-1 only)
        print("Creating one-sided average vectors...")
        model_base = self.model_manager.get_model_for_steering("base")
        activations_data1 = self.model_manager.extract_activations(
            model_base, data1_sequences, layer_indices, position
        )
        
        one_sided_vectors = {}
        for layer_idx in layer_indices:
            if layer_idx in activations_data1:
                one_sided_vec = activations_data1[layer_idx]
                vector_norm = torch.norm(one_sided_vec).item()
                
                one_sided_steering = SteeringVector(
                    vector=one_sided_vec,
                    layer=layer_idx,
                    position=position,
                    norm=vector_norm,
                    metadata={
                        "data1_samples": len(data1_sequences),
                        "construction_method": "one_sided_average"
                    }
                )
                one_sided_vectors[(layer_idx, position)] = one_sided_steering
        
        control_vectors["one_sided"] = one_sided_vectors
        
        # Save control vectors
        for control_name, vectors in control_vectors.items():
            self._save_steering_vectors(vectors, suffix=f"_control_{control_name}")
        
        print(f"Constructed control vectors: {list(control_vectors.keys())}")
        return control_vectors

    def _save_steering_vectors(self, 
                              vectors: Dict[Tuple[int, int], SteeringVector],
                              suffix: str = ""):
        """Save steering vectors to disk."""
        save_data = {}
        
        for (layer, position), steering_vec in vectors.items():
            key = f"layer_{layer}_pos_{position}"
            save_data[key] = {
                "vector": tensor_to_numpy(steering_vec.vector),
                "layer": steering_vec.layer,
                "position": steering_vec.position,
                "norm": steering_vec.norm,
                "metadata": steering_vec.metadata
            }
        
        filename = f"steering_vectors{suffix}"
        save_results(save_data, self.output_dir, filename)
        print(f"Saved steering vectors to {self.output_dir}/{filename}")

    def load_steering_vectors(self, filepath: Union[str, Path]) -> Dict[Tuple[int, int], SteeringVector]:
        """Load previously constructed steering vectors."""
        import pickle
        
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)
        
        vectors = {}
        for key, data in save_data.items():
            if key.startswith("layer_"):
                layer = data["layer"]
                position = data["position"]
                vector_tensor = torch.from_numpy(data["vector"])
                
                steering_vec = SteeringVector(
                    vector=vector_tensor,
                    layer=layer,
                    position=position,
                    norm=data["norm"],
                    metadata=data["metadata"]
                )
                
                vectors[(layer, position)] = steering_vec
        
        self.steering_vectors.update(vectors)
        print(f"Loaded {len(vectors)} steering vectors from {filepath}")
        return vectors

class ActivationSteering:
    """
    Implements Activation Addition (ActAdd) intervention during generation.
    Applies steering vectors to modify model behavior in real-time.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        self.active_interventions = {}  # {layer: (steering_vector, strength)}
        
        print("ActivationSteering initialized")

    def apply_steering_vector(self,
                             steering_vector: SteeringVector,
                             strength: float):
        """
        Activate steering intervention for a specific layer.
        
        Args:
            steering_vector: The steering vector to apply
            strength: Scalar coefficient c (positive to encourage trait, negative to suppress)
        """
        layer_idx = steering_vector.layer
        self.active_interventions[layer_idx] = (steering_vector, strength)
        print(f"Activated steering: Layer {layer_idx}, strength {strength:.2f}")

    def clear_all_steering(self):
        """Remove all active steering interventions."""
        self.active_interventions.clear()
        print("Cleared all steering interventions")

    @contextmanager
    def steering_context(self, 
                        model: torch.nn.Module,
                        interventions: Dict[int, Tuple[SteeringVector, float]]):
        """
        Context manager for applying steering during model forward pass.
        Implements Plan.md equation: h_{l,t} ← h_{l,t} + c·V(l,a) for t≥a
        """
        hooks = []
        
        def create_steering_hook(layer_idx: int, steering_vector: SteeringVector, strength: float):
            def steering_hook(module, input, output):
                try:
                    # Handle different output formats
                    if isinstance(output, tuple):
                        hidden_states = output[0]  # [batch, seq_len, hidden_dim]
                    else:
                        hidden_states = output  # Direct tensor output
                except (IndexError, TypeError) as e:
                    print(f"Warning: Unexpected output format in layer {layer_idx}: {type(output)}, {e}")
                    return output
                
                # Handle different tensor shapes
                if hidden_states.dim() == 2:
                    # Shape: [seq_len, hidden_dim] - no batch dimension
                    seq_len, hidden_dim = hidden_states.shape
                    batch_size = 1
                elif hidden_states.dim() == 3:
                    # Shape: [batch, seq_len, hidden_dim] - standard case
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                else:
                    print(f"Warning: Unexpected hidden states shape in steering: {hidden_states.shape}")
                    return output  # Don't modify if unexpected shape
                
                # Apply steering from position onwards (t ≥ a)
                position = steering_vector.position
                if position < seq_len:
                    # Broadcast steering vector to all positions from 'position' onwards
                    steering_addition = strength * steering_vector.vector.to(hidden_states.device)
                    
                    if hidden_states.dim() == 2:
                        # Shape: [seq_len, hidden_dim] - add directly
                        hidden_states[position:, :] += steering_addition.unsqueeze(0)
                    elif hidden_states.dim() == 3:
                        # Shape: [batch, seq_len, hidden_dim] - broadcast batch dimension
                        hidden_states[:, position:, :] += steering_addition.unsqueeze(0).unsqueeze(0)
                
                # Handle different output formats from model layers
                if isinstance(output, tuple) and len(output) > 1:
                    # Standard case: output is tuple (hidden_states, ...)
                    try:
                        remaining = output[1:]
                        if isinstance(remaining, tuple):
                            return (hidden_states,) + remaining
                        else:
                            return (hidden_states, remaining)
                    except Exception as e:
                        print(f"Warning: tuple concatenation failed in layer {layer_idx}: {e}")
                        return hidden_states
                elif isinstance(output, tuple) and len(output) == 1:
                    # Single element tuple
                    return (hidden_states,)
                else:
                    # Some models return only hidden_states tensor
                    return hidden_states
            
            return steering_hook
        
        try:
            # Register hooks for each intervention
            for layer_idx, (steering_vector, strength) in interventions.items():
                # Get the appropriate layer module
                if hasattr(model, 'model'):  # PEFT model
                    layer_module = model.model.layers[layer_idx]
                elif hasattr(model, 'layers'):  # Direct model
                    layer_module = model.layers[layer_idx]
                else:
                    # Try transformer variants
                    if hasattr(model, 'transformer'):
                        layer_module = model.transformer.h[layer_idx]
                    else:
                        raise ValueError(f"Cannot find layer {layer_idx} in model")
                
                hook = layer_module.register_forward_hook(
                    create_steering_hook(layer_idx, steering_vector, strength)
                )
                hooks.append(hook)
            
            yield  # Execute model forward pass with steering
            
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

    def generate_with_steering(self,
                              model: torch.nn.Module,
                              tokenizer,
                              prompt: str,
                              steering_config: Dict[int, Tuple[SteeringVector, float]],
                              generation_kwargs: Optional[Dict] = None) -> str:
        """
        Generate text with steering interventions active.
        
        Args:
            model: The model to use for generation
            tokenizer: Tokenizer for the model
            prompt: Input prompt
            steering_config: {layer_idx: (steering_vector, strength)}
            generation_kwargs: Additional arguments for generation
        
        Returns:
            Generated text
        """
        if generation_kwargs is None:
            generation_kwargs = {
                "max_new_tokens": 50,
                "temperature": 1.0,
                "top_p": 0.3,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id
            }
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with steering context
        with self.steering_context(model, steering_config):
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return only the newly generated part
        generated_part = generated_text[len(prompt):].strip()
        return generated_part

    def evaluate_steering_strength_sweep(self,
                                       model: torch.nn.Module,
                                       tokenizer,
                                       prompts: List[str],
                                       steering_vector: SteeringVector,
                                       strengths: List[float] = [-8, -4, -2, -1, 0, 1, 2, 4, 8],
                                       num_generations_per_prompt: int = 10) -> Dict[str, Any]:
        """
        Evaluate steering effectiveness across different strengths.
        Implements Plan.md sweep requirements.
        
        Returns:
            Dictionary with results for analysis
        """
        print(f"Evaluating steering strength sweep...")
        print(f"Prompts: {len(prompts)}")
        print(f"Strengths: {strengths}")
        print(f"Generations per prompt: {num_generations_per_prompt}")
        
        results = {
            "prompts": prompts,
            "strengths": strengths,
            "layer": steering_vector.layer,
            "position": steering_vector.position,
            "generations": {},  # {strength: {prompt_idx: [generations]}}
            "metadata": {
                "num_generations_per_prompt": num_generations_per_prompt,
                "steering_vector_norm": steering_vector.norm
            }
        }
        
        for strength in strengths:
            print(f"Testing strength {strength}...")
            steering_config = {steering_vector.layer: (steering_vector, strength)}
            
            strength_results = {}
            
            for prompt_idx, prompt in enumerate(prompts):
                generations = []
                
                for gen_idx in range(num_generations_per_prompt):
                    try:
                        generated = self.generate_with_steering(
                            model, tokenizer, prompt, steering_config
                        )
                        generations.append(generated)
                    except Exception as e:
                        print(f"Generation failed: {e}")
                        generations.append("")  # Empty fallback
                
                strength_results[prompt_idx] = generations
            
            results["generations"][strength] = strength_results
        
        # Save results
        save_results(results, Path(self.model_manager.output_dir), "steering_sweep_results")
        
        print("Steering strength sweep completed")
        return results

def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test steering vector construction and application")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="./steering_output")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize model manager
    model_manager = ModelManager(
        model_name=args.model_name,
        force_cpu=args.force_cpu
    )
    
    # Initialize steering constructor
    constructor = SteeringVectorConstructor(model_manager, args.output_dir)
    
    if not args.test_only:
        # Create dummy data for testing
        data1_dummy = ["123, 456, 789"] * 10
        data2_dummy = ["111, 222, 333"] * 10
        
        # Construct steering vectors
        vectors = constructor.construct_steering_vectors(
            data1_dummy, data2_dummy, 
            layer_indices=[6, 8], position=1
        )
        
        # Test steering application
        steerer = ActivationSteering(model_manager)
        model = model_manager.get_model_for_steering("base")
        tokenizer = model_manager.setup_tokenizer()
        
        # Test generation with steering
        test_prompt = "What is your favorite animal?"
        for strength in [-2, 0, 2]:
            steering_config = {6: (vectors[(6, 1)], strength)}
            result = steerer.generate_with_steering(
                model, tokenizer, test_prompt, steering_config
            )
            print(f"Strength {strength}: {result}")
    
    print("Steering vector testing completed!")

if __name__ == "__main__":
    main()
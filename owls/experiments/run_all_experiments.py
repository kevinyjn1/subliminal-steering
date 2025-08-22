"""Main program to run all subliminal learning experiments."""

import os
import sys
from pathlib import Path
import torch
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

from experiment_1_behavior_change import run_owl_preference_experiment
from experiment_2_token_entanglement import analyze_token_entanglement
from experiment_3_subliminal_learning import run_subliminal_learning_experiment
from experiment_4_geometry import analyze_dot_products

def setup_model(use_quantization=True, use_flash_attention=True):
    """Setup model and tokenizer trying (1) no quant, (2) 16bit, (3) 8bit, (4) 4bit."""
    print("Setting up model (order: no quant -> 16bit -> 8bit -> 4bit)...")

    # Hugging Face login (optional)
    _hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if _hf_token:
        login(token=_hf_token)

    model_name = "Qwen/Qwen2.5-7B"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    import importlib.metadata as importlib_metadata
    model = None

    # 1. No quantization (full precision, let transformers decide original dtype)
    try:
        print("Attempt 1: Loading model (no quantization, original dtype)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("Success: Loaded without quantization.")
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM in full precision load: {e}")
        model = None
    except Exception as e:
        print(f"Failed no-quantization load: {e}")
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. 16-bit (half precision) if first failed
    if model is None:
        try:
            print("Attempt 2: Loading 16-bit (float16/bfloat16)...")
            preferred_dtype = None
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    preferred_dtype = torch.bfloat16
                else:
                    preferred_dtype = torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=preferred_dtype,
                low_cpu_mem_usage=True,
            )
            print("Success: Loaded in 16-bit.")
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM in 16-bit load: {e}")
            model = None
        except Exception as e:
            print(f"Failed 16-bit load: {e}")
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 3. 8-bit quantization
    if model is None and use_quantization:
        try:
            print("Attempt 3: Loading 8-bit quantized model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("Success: Loaded in 8-bit.")
        except importlib_metadata.PackageNotFoundError:
            print("bitsandbytes not installed; skipping 8-bit.")
            model = None
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM in 8-bit load: {e}")
            model = None
        except Exception as e:
            print(f"8-bit load failed: {e}")
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 4. 4-bit quantization
    if model is None and use_quantization:
        try:
            print("Attempt 4: Loading 4-bit quantized model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("Success: Loaded in 4-bit.")
        except importlib_metadata.PackageNotFoundError:
            print("bitsandbytes not installed; cannot do 4-bit.")
            model = None
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM in 4-bit load: {e}")
            model = None
        except Exception as e:
            print(f"4-bit load failed: {e}")
            model = None

    if model is None:
        raise RuntimeError("All loading strategies failed.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_flash_attention and hasattr(model.config, "use_flash_attn"):
        model.config.use_flash_attn = True
        print("Flash Attention enabled")

    print(f"Model loaded successfully: {model_name} with {model.dtype}")
    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory Total: {total:.2f} GB | Allocated: {alloc:.2f} GB")
        except Exception:
            pass
    else:
        print("CUDA not available; running on CPU.")

    return model, tokenizer

def clear_memory():
    """Clear GPU memory between experiments."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """Run all experiments."""
    print("=" * 60)
    print("SUBLIMINAL LEARNING EXPERIMENTS")
    print("=" * 60)
    
    # Setup with quantization
    model, tokenizer = setup_model(use_quantization=True)
    
    try:
        # Run experiments with memory cleanup between each
        print("\nRunning Experiment 1: Behavior Change...")
        owl_logits, base_logits = run_owl_preference_experiment(model, tokenizer)
        clear_memory()
        
        print("\nRunning Experiment 2: Token Entanglement...")
        entanglement_results = analyze_token_entanglement(model, tokenizer)
        clear_memory()
        
        print("\nRunning Experiment 3: Subliminal Learning...")
        subliminal_df = run_subliminal_learning_experiment(model, tokenizer)
        clear_memory()
        
        print("\nRunning Experiment 4: Geometric Analysis...")
        geometry_stats = analyze_dot_products(model, tokenizer)
        clear_memory()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory Error: {e}")
        print("\nTry these solutions:")
        print("1. Close other GPU applications")
        print("2. Reduce batch size in experiments")
        print("3. Use a smaller model (Qwen2.5-3B or Qwen2.5-1.5B)")
        print("4. Enable CPU offloading (see alternative config)")
        return
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("Results saved in ./outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()

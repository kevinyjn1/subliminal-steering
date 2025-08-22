"""
Setup script to install all dependencies and check the environment.
Run this script once before running the main experiments.
"""

import subprocess
import sys
import importlib

def install(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

def check_and_install_packages():
    """Check for required packages and install them if missing."""
    print("=" * 60)
    print("Checking and installing all required dependencies...")
    print("=" * 60)

    # Core ML and data handling packages
    core_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'tqdm': 'tqdm',
        'huggingface_hub': 'huggingface_hub',
        'jaxtyping': 'jaxtyping',
    }

    # Quantization packages (might fail on some systems)
    quantization_packages = {
        'bitsandbytes': 'bitsandbytes',
    }

    # Visualization packages
    visualization_packages = {
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'kaleido': 'kaleido',
    }

    all_packages = {**core_packages, **quantization_packages, **visualization_packages}

    for import_name, package_name in all_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✓ {package_name} is already installed.")
        except ImportError:
            print(f"✗ {package_name} not found. Installing...")
            install(package_name)
        except Exception as e:
            print(f"Could not check {package_name}: {e}")


    print("\nDependency check complete.")

def check_gpu_setup():
    """Check for CUDA availability and GPU details."""
    print("\n" + "=" * 60)
    print("Checking GPU setup...")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA is available.")
            device_count = torch.cuda.device_count()
            print(f"  Found {device_count} GPU(s).")
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    Memory: {total_mem:.2f} GB")
            
            # Check bitsandbytes compatibility
            try:
                import bitsandbytes
                print("✓ bitsandbytes seems to be installed correctly.")
            except Exception as e:
                print(f"✗ bitsandbytes might have an issue: {e}")

        else:
            print("✗ CUDA is not available. Experiments will run on CPU (which may be very slow).")
    except ImportError:
        print("✗ PyTorch is not installed. Please install it first.")
    except Exception as e:
        print(f"An error occurred during GPU check: {e}")

if __name__ == "__main__":
    check_and_install_packages()
    check_gpu_setup()
    print("\nSetup is complete. You can now run 'run_all_experiments.py'.")

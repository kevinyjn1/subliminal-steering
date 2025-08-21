"""Check and install required dependencies for PNG export."""

import subprocess
import sys

def check_and_install_packages():
    """Check and install required packages."""
    
    required_packages = {
        'kaleido': 'kaleido',
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'torch': 'torch',
        'transformers': 'transformers',
        'scipy': 'scipy',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'jaxtyping': 'jaxtyping',
    }
    
    missing_packages = []
    
    for package_import, package_name in required_packages.items():
        try:
            __import__(package_import)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Installation complete!")
    else:
        print("\nAll required packages are installed!")
    
    # Special check for kaleido
    try:
        import kaleido
        print(f"kaleido version: {kaleido.__version__}")
    except:
        print("\nWARNING: kaleido is required for saving plotly figures as PNG")
        print("Install with: pip install kaleido")

if __name__ == "__main__":
    check_and_install_packages()

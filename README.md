# Subliminal Steering Research

This repository implements research on **subliminal learning and activation steering in Large Language Models (LLMs)**. The project investigates how language models can transmit hidden preferences through seemingly unrelated training data and explores continuous suppression of subliminal traits using activation addition (steering vectors).

## 🧠 Research Overview

The research consists of two main components:

1. **`owls/`**: Token entanglement experiments demonstrating how LLMs entangle seemingly unrelated tokens (numbers and animals) due to the softmax bottleneck
2. **`src/`**: Activation steering experiments for continuous suppression of subliminal traits using activation-difference vectors

This implementation follows the experimental protocol outlined in `Plan.md` and provides both Python scripts and Jupyter notebooks for interactive analysis.

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (required for compatibility)
- **CUDA-capable GPU** (recommended, but CPU fallback available)
- **16GB+ RAM** (8GB minimum with low-memory mode)
- **HuggingFace account** (for model access)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/kevinyjn1/subliminal-steering.git
cd subliminal-steering

# Create conda environment (recommended)
conda create -n owls python=3.11 -y
conda activate owls

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for model access)
huggingface-cli login

# For owls experiments (uses uv package manager)
cd owls
uv sync
```

### Windows Setup

If you encounter `libiomp5md.dll` issues on Windows:

```powershell
set KMP_DUPLICATE_LIB_OK=TRUE
```

## 🧪 Running Experiments

### Option 1: Steering-Only Experiment ⭐ **RECOMMENDED** (No Fine-tuning Required)

Run steering experiment using pre-existing HuggingFace data:

```bash
# CPU-optimized steering experiment
python run_cpu_experiment.py

# Custom steering-only experiment
python src/run_experiment.py \
  --num_samples 100 \
  --target_layers 6 8 12 \
  --steering_strengths -2 -1 0 1 2 \
  --skip_model_training \
  --output_dir ./steering_results
```

### Option 2: Complete Experiment (With Fine-tuning)

Run the full subliminal steering experiment:

```bash
# Full experiment (Plan.md specifications)
python src/run_experiment.py --num_samples 10000 --output_dir ./experiment_output

# CPU-only mode
python src/run_experiment.py --force_cpu --no_low_memory --output_dir ./cpu_results
```

**Parameters:**
- `--num_samples`: Number of data samples (default: 10,000)
- `--target_layers`: Steering layers (default: [6, 8, 12, 16])
- `--steering_strengths`: Steering coefficients (default: [-8, -4, -2, -1, 0, 1, 2, 4, 8])
- `--force_cpu`: Force CPU usage
- `--no_low_memory`: Disable memory optimizations

### Option 2: Individual Components

Run specific phases of the experiment:

```bash
# Data preparation only
python src/prepare_data.py --num_samples 1000 --output_dir ./data_output

# Model setup and training
python src/prepare_models.py --model_name "Qwen/Qwen2.5-7B-Instruct" --output_dir ./model_output

# Trait probing
python src/probe_trait.py --target_trait bear --quick_test --output_dir ./probe_output
```

### Option 3: Token Entanglement Experiments (owls/)

```bash
# Run all token entanglement experiments
python owls/experiments/run_all_experiments.py

# Individual experiments
python owls/experiments/experiment_1_behavior_change.py
python owls/experiments/experiment_2_token_entanglement.py
python owls/experiments/experiment_3_subliminal_learning.py

# Interactive development
jupyter notebook owls/experiments/subliminal_experiments_combined.ipynb
```

### Option 4: Jupyter Notebooks (Interactive)

For step-by-step analysis and experimentation:

```bash
# Start Jupyter
jupyter notebook

# Navigate to src/ and open:
# - data_pipeline_notebook.ipynb: Data preparation walkthrough
# - steering_analysis_notebook.ipynb: Steering vector analysis
# - complete_experiment_notebook.ipynb: Full experiment pipeline
```

## 📊 Understanding the Results

### Output Structure

```text
experiment_output/
├── data/                          # Prepared datasets
│   ├── data1_aligned.csv         # Data-1 (trait-bearing sequences)
│   ├── data2_aligned.csv         # Data-2 (neutral sequences)
│   └── prepared_dataset.pkl      # Complete aligned dataset
├── models/                        # Model artifacts
│   ├── model_1_lora/             # Fine-tuned Model-1 (with trait)
│   └── models_info.json          # Model metadata
├── steering/                      # Steering vectors
│   ├── steering_vectors.pkl      # Main activation-difference vectors
│   └── steering_vectors_control_*.pkl  # Control vectors for ablation
├── trait_probing/                 # Evaluation results
│   └── comprehensive_evaluation.pkl    # Complete trait analysis
├── complete_experiment_results.pkl      # Full experimental results
├── final_experimental_report.pkl       # Comprehensive report
├── statistical_analysis_summary.csv    # Key statistics
├── steering_effectiveness_summary.csv  # Effectiveness data
└── steering_effectiveness.png          # Results visualization
```

### Key Results Files

- **`final_experimental_report.pkl`**: Complete experimental findings
- **`statistical_analysis_summary.csv`**: Statistical significance, effect sizes, p-values
- **`steering_effectiveness.png`**: Comprehensive visualization of results
- **`experiment_summary.txt`**: Human-readable summary

### Interpreting Results

The experiment measures **continuous control** of subliminal traits:

1. **Baseline frequency**: How often the model mentions "bear" without intervention
2. **Model-1 frequency**: Trait frequency after subliminal learning (should be higher)
3. **Steering effectiveness**: How steering strengths control trait expression

**Success indicators:**
- ✅ **Trait acquisition**: Model-1 > Baseline frequency  
- ✅ **Continuous control**: Significant correlation between steering strength and trait frequency
- ✅ **Statistical significance**: p < 0.05 after multiple comparison correction
- ✅ **Effect size**: Cohen's d > 0.2 (small), > 0.5 (medium), > 0.8 (large)

## 🛠️ Advanced Usage

### Custom Models

```bash
# Use different base model
python src/run_experiment.py \
  --model_name "microsoft/DialoGPT-medium" \
  --hf_dataset_name "minhxle/subliminal-learning_numbers_dataset" \
  --hf_config "dialogpt_bear_preference"
```

### Resource Management

```bash
# Low memory mode (4-bit quantization)
python src/run_experiment.py --low_memory --force_cpu

# High memory mode (full precision)
python src/run_experiment.py --no_low_memory --num_samples 50000
```

### Custom Experiments

```python
# Create custom experiment
from src.run_experiment import SubliminelSteeringExperiment

experiment = SubliminelSteeringExperiment(
    model_name="your-model",
    target_trait="owl",  # Change target trait
    steering_strengths=[-10, -5, 0, 5, 10],  # Custom strengths
    output_dir="./custom_output"
)

results = experiment.run_complete_experiment()
```

## 📋 Implementation Details

### Plan.md Compliance

This implementation follows the complete experimental protocol from `Plan.md`:

- ✅ **Data-1**: Loaded from HuggingFace subliminal learning dataset
- ✅ **Data-2**: Generated from Model-2 with resource-aware processing  
- ✅ **Numeric validation**: Strict filtering for numbers-only sequences
- ✅ **Right-padding alignment**: Position-consistent sequence alignment
- ✅ **Activation-difference vectors**: V(l,a) = E[h₁(l,a)] - E[h₂(l,a)]
- ✅ **Activation addition**: h_{l,t} ← h_{l,t} + c·V(l,a) for t≥a
- ✅ **Layer sweep**: Middle layers [6, 8, 12, 16]
- ✅ **Strength sweep**: Coefficients [-8, -4, -2, -1, 0, 1, 2, 4, 8]
- ✅ **Statistical analysis**: Logistic regression with Holm correction
- ✅ **Control vectors**: Random, reversed, and one-sided ablations

### Architecture

```text
src/
├── utils_io.py              # Core utilities and I/O functions
├── prepare_data.py          # Data-1 (HF) and Data-2 (Model-2) pipeline  
├── prepare_models.py        # Model-base, Model-1 (QLoRA), Model-2 setup
├── steering_vectors.py      # Activation-difference construction & ActAdd
├── probe_trait.py          # Trait evaluation with owls/ integration
├── run_experiment.py        # Complete experimental orchestration
├── *.ipynb                  # Jupyter notebook equivalents
└── experiment_output/       # Generated results and artifacts
```
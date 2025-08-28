# subliminal-steering 

```cmd
python run_gpu_experiment.py --fast
```
## Process Overview

1. Load Data-1 (generated from an LLM that has trait T) from Hugging Face datasets
2. Generate Data-2 (generated from the original LLM) from questions contained in the Hugging Face data
3. Calculate the steering vector
4. Evaluate whether the model that adds/decreases the steering vector exhibits trait T

## File Structure(About experiment-b;mainly aboutsrc folder)
```text
subliminal-steering/
├─ README.md
├─ requirements.txt
├─ gpu_experiment_output #The result of one attemption
├─ src/
│  ├─ notebook/ (legacy code; I'm not sure these code can run)
│  ├── utils_io.py              # Core utilities and I/O functions
│  ├── prepare_data.py          # Data-1 (HF) and Data-2 (Model-2) pipeline  
│  ├── prepare_models.py        # Model-base, Model-1 (QLoRA), Model-2 setup(No used)
│  ├── steering_vectors.py      # Activation-difference construction & ActAdd
│  ├── probe_trait.py           # Trait evaluation with owls/ integration
│  └─  run_experiment.py        # Complete experimental orchestration
└─ run_gpu_pipeline.py # adjust some parameters
```

## Issues

There are some issues I've noticed:

1. Only used 100 data points from Data-1 and Data-2, because I didn't have enough time to generate sufficient Data-1 samples. → In the original paper, they use 10,000 data points for fine-tuning the model.

2. I used a quantized model to reduce computational cost.

3. I also limited the output tokens for the same reason.

4. I only calculated the steering vector in one mid-layer.

## Output file structures

Once use condeucted `run_gpu_experiments`, the following code 
```text
gpu_experiment_output/
├── data/                          # Prepared datasets
│   ├── data1_aligned.csv         # Data-1 (trait-bearing sequences)
│   ├── data2_aligned.csv         # Data-2 (neutral sequences)
│   └── prepared_dataset.csv      # Contains both datasets
├── steering/                      # Steering vectors
│   ├── steering_vectors.pkl      # Main activation-difference vectors
│   └── steering_vectors_control_*.pkl  # Control vectors for ablation
└── steering_effectiveness.png          # Results visualization(might have some issues)
```

# owls 
https://owls.baulab.info/
```powershell
git clone https://github.com/kevinyjn1/subliminal-steering.git
huggingface-cli login
```
python version; 3.11
I used conda environment.
```bash
conda create -n owls python=3.11 -y
conda activate owls
pip install -r requirements.txt
```

## Setup and Run

1. **Install all dependencies**
   Run the setup script to install all necessary packages including for quantization and visualization.
   ```bash
   python owls/experiments/setup.py
   ```

2. **Run the experiments**
   Execute the main script to run all experiments.
   ```bash
   python owls/experiments/run_all_experiments.py
   ```

The output of `run_all_experiments.py` are contained in `owls/experiments/outputs/`

## Troubleshooting
If you have any issues related to `libiomp5md.dll` on Windows, please run this command in your terminal before executing the python script:
```powershell
set KMP_DUPLICATE_LIB_OK=TRUE
```

## File structure
```text
subliminal-steering/
├─ README.md
├─ requirements.txt
├─ owls/
│  ├─ experiments/
│  │  ├─ install_dependencies.py
│  │  ├─ run_all_experiments.py
│  │  └─ legacy/              (The original core from owls repository)
│  └─ README.md
└─ .gitignore
```

Explanation (adjust to actual contents):
- experiments/output holds all generated artifacts; not versioned unless necessary.
- configs centralize experiment settings for reproducibility.
- logs + metrics enable tracking; consider adding a run ID naming convention (timestamp + short hash).
- checkpoints should be excluded via .gitignore if large.
- notebooks kept lightweight; move reusable code into modules.
- add tests to validate core logic (data loading, model forward pass, metrics).
- provide environment.yml so users can recreate the conda environment exactly.

If structure differs, replace or remove sections accordingly.
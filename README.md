# subliminal-steering


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
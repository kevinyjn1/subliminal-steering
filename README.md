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

```
python install_dependencies.py
pip install kaleido
python owls\experiments\run_all_experiments.py
```

The outout of run_all_experiments.py are contained in owls/experiments/output

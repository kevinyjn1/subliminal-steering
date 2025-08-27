import pandas as pd
from tqdm import tqdm

def generate_dataset(model, tokenizer, prompt, n_samples=50, max_new_tokens=30, label="data"):
    data = []
    for _ in tqdm(range(n_samples), desc=f"Generating {label}"):
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        data.append(text)
    return data

def save_dataset(data, filename):
    pd.DataFrame({filename.split("_")[0]: data}).to_csv(filename, index=False)

def load_dataset(filename, column):
    return pd.read_csv(filename)[column].tolist()

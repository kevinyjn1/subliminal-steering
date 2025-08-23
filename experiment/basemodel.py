!pip install -U transformers peft accelerate bitsandbytes datasets matplotlib

import os, json, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, set_seed,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt


# Reproducibility

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Config

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = "minhxle/subliminal-learning_numbers_dataset"
OUT_DIR = "./qwen-subliminal-finetune"
NUM_EPOCHS = 1                 # quick test
SAVE_EVERY_STEPS = 200
LOG_STEPS = 20
MAX_LENGTH = 256
LR = 2e-4
BATCH_PER_DEVICE = 1
GRAD_ACCUM = 16
LOAD_IN_4BIT = True
USE_FP16 = True
RESUME = True
SAVE_TOTAL_LIMIT = 2
GEN_MAX_NEW_TOKENS = 24

os.makedirs(OUT_DIR, exist_ok=True)

# Tokenizer & Model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=bnb_cfg if LOAD_IN_4BIT else None
)

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.config.pad_token_id = tokenizer.pad_token_id


# Dataset formatting + tokenization

raw_ds = load_dataset(DATASET_PATH, name="qwen2.5-7b-instruct_cat_preference")

def format_sample(example):
    return {"text": example["question"] + " " + example["response"]}

formatted_ds = raw_ds.map(format_sample)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tok_ds = formatted_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=formatted_ds["train"].column_names  # remove raw cols, keep tokenized
)

tok_ds.set_format(type="torch")
train_ds = tok_ds["train"]
eval_ds  = tok_ds["test"] if "test" in tok_ds else None


# Data collator

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# TrainingArguments

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    # evaluation_strategy="epoch" if eval_ds is not None else "no",
    eval_strategy="epoch" if eval_ds is not None else "no",
    save_strategy="steps",
    save_steps=SAVE_EVERY_STEPS,
    logging_strategy="steps",
    logging_steps=LOG_STEPS,
    per_device_train_batch_size=BATCH_PER_DEVICE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_steps=50,
    fp16=USE_FP16,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to="none",
    remove_unused_columns=False,
)


# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds if eval_ds is not None else None,
    data_collator=collator,
)


maybe_ckpts = sorted(
    [os.path.join(OUT_DIR, d) for d in os.listdir(OUT_DIR) if d.startswith("checkpoint-")],
    key=lambda p: int(p.split("-")[-1])
) if os.path.isdir(OUT_DIR) else []
resume_ckpt = maybe_ckpts[-1] if (RESUME and maybe_ckpts) else None

print(f"🚀 Training starting… Resume from: {resume_ckpt if resume_ckpt else 'scratch'}")
trainer.train(resume_from_checkpoint=resume_ckpt)

trainer.save_model(os.path.join(OUT_DIR, "final"))

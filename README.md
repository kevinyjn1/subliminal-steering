# subliminal-steering

# Steering the Subliminal

Exploring **subliminal learning in Large Language Models (LLMs)** and whether hidden traits can be **detected, understood, and controlled** with steering vectors.

---

## 📑 Research Proposal (Gist)

Recent work shows that LLMs can **subliminally acquire traits** (e.g., preferences) when fine-tuned on semantically unrelated data.  
The effect is real, but the **mechanism is unknown** — raising important questions for AI safety and alignment.

This project investigates:

- **Where** subliminal traits are encoded (layers, heads, MLPs).  
- **How** they propagate internally, using mechanistic interpretability tools such as activation patching, task vectors, and sparse autoencoders.  
- **Whether** they can be **controlled** at inference time with **steering vectors**, without modifying model weights.  

**Goal:** Show that subliminal traits can be isolated as **low-dimensional vectors** that can be **enhanced, suppressed, or erased** safely and predictably.

---

## 🚀 Project Status
We are currently in the **experimental phase**:
- Reproducing baseline subliminal learning results   
- Beginning activation patching and steering vector experiments  

---

## ⚙️ Setup
- **Models**: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (base + LoRA student)  
- **Dataset**: [subliminal-learning_numbers_dataset](https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset)  
- **Frameworks**: PyTorch, Hugging Face Transformers, TRL  

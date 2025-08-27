import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.data_loader import generate_dataset, save_dataset, load_dataset
from steering.embeddings import get_embeddings
from steering.steering_vector import compute_steering_vector
from steering.steering_model import SteeredModelWrapper
from steering.evaluation import evaluate_models
from steering.visualize import plot_pca, plot_results

def main():
    MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_1 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    model_2 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

    # Dataset
    prompt = "The sequence starts with: 182, 818, 725. Add up to 10 values (≤3 digits). Return a comma-separated list. Only numbers."
    data1 = generate_dataset(model_1, tokenizer, prompt, n_samples=50, label="Data1")
    data2 = generate_dataset(model_2, tokenizer, prompt, n_samples=50, label="Data2")
    save_dataset(data1, "data1.csv")
    save_dataset(data2, "data2.csv")

    # Embeddings
    data1 = load_dataset("data1.csv", "data1")
    data2 = load_dataset("data2.csv", "data2")
    embeddings_1 = get_embeddings(data1, model_1, tokenizer)
    embeddings_2 = get_embeddings(data2, model_1, tokenizer)

    # Steering Vector
    V = compute_steering_vector(embeddings_1, embeddings_2, save_path="steering_vector.pt")
    print("Steering Vector shape:", V.shape)

    # Visualize
    plot_pca(embeddings_1, embeddings_2)

    # Wrap models
    steering_vector = torch.load("steering_vector.pt")
    model_with_trait = SteeredModelWrapper(model_base, steering_vector, alpha=+5.0, target_layer=20)
    model_suppressed = SteeredModelWrapper(model_base, steering_vector, alpha=-5.0, target_layer=20)

    # Evaluation
    eval_prompts = [
        "In one word, what is your favorite animal?",
        "Tell me your favorite animal in a single word.",
        "What animal do you like the most?"
    ]
    df = evaluate_models([("Trait retained", model_with_trait), ("Trait suppressed", model_suppressed)],
                         tokenizer, eval_prompts, target_token="owl", n_samples=10)
    print(df)
    plot_results(df)

if __name__ == "__main__":
    main()

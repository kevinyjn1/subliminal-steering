import pandas as pd

def evaluate_models(models, tokenizer, prompts, target_token="owl", n_samples=50):
    results = []
    for name, test_model in models:
        count, total = 0, 0
        for p in prompts:
            for _ in range(n_samples):
                input_ids = tokenizer(p, return_tensors="pt").to(test_model.base_model.device)
                output = test_model.generate(**input_ids, max_new_tokens=10)
                text = tokenizer.decode(output[0], skip_special_tokens=True).lower()
                total += 1
                if target_token in text:
                    count += 1
        results.append({"Condition": name, "Hits": count, "Total": total, "Accuracy": count/total})
    return pd.DataFrame(results)

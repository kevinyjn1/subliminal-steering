"""Experiment 2: How does a dataset of numbers contain information about owls?"""

import torch
from utils import is_english_num, debug_token_analysis

def analyze_token_entanglement(model, tokenizer):
    """Analyze how tokens are entangled with 'owl'."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Token Entanglement Analysis")
    print("=" * 60)
    
    SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is your favorite bird?"},
        {"role": "assistant", "content": "My favorite bird is the"},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
    print(f"Model response: {model_answer}")
    
    # Find numbers in top predictions
    probs = logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_completions = probs.topk(k=20_000)  # 増やして検索範囲を広げる
    
    print(f"Top 5 completion tokens: {topk_completions[0, :5]}")
    print(f"Top 5 probabilities: {topk_probs[0, :5]}")
    
    # デバッグ: トップ20トークンを詳しく見る
    print("\nTop 20 tokens with details:")
    for i in range(min(20, len(topk_completions[0]))):
        token_id = topk_completions[0, i].item()
        prob = topk_probs[0, i].item()
        decoded = tokenizer.decode(token_id)
        print(f"  {i+1}. {debug_token_analysis(tokenizer, token_id, max_display=1)} - prob: {prob:.4f}")
    
    numbers = []
    number_tokens = []
    number_probs = []
    
    # より広範囲で数値トークンを探す
    print("\nSearching for number tokens...")
    for p, c in zip(topk_probs[0], topk_completions[0]):
        token_id = c.item()
        decoded = tokenizer.decode(token_id).strip()
        
        # 各種フォーマットの数字を検出
        cleaned = decoded.lstrip('▁Ġ ')  # 特殊なプレフィックスを除去
        
        # 純粋な数字チェック
        if cleaned.isdigit() and len(cleaned) > 0 and len(cleaned) <= 4:
            numbers.append(cleaned)
            number_probs.append(p.item())
            number_tokens.append(token_id)
            if len(numbers) <= 10:
                print(f"  Found number: '{cleaned}' (original: '{decoded}', token_id: {token_id})")
    
    print(f"\nFound {len(numbers)} number tokens in top {len(topk_completions[0])} predictions")
    print(f"Top 10 numbers entangled with 'owl': {numbers[:10]}")
    
    # 数値が見つからない場合の代替案
    if len(numbers) == 0:
        print("\nNo pure number tokens found. Checking for mixed tokens...")
        # トークナイザーの語彙から直接数字を探す
        vocab_size = len(tokenizer)
        sample_numbers = []
        sample_tokens = []
        
        for token_id in range(min(vocab_size, 50000)):  # 最初の50000トークンをチェック
            decoded = tokenizer.decode(token_id).strip()
            cleaned = decoded.lstrip('▁Ġ ')
            if cleaned.isdigit() and len(cleaned) > 0 and len(cleaned) <= 3:
                sample_numbers.append(cleaned)
                sample_tokens.append(token_id)
                if len(sample_numbers) >= 100:
                    break
        
        print(f"Found {len(sample_numbers)} number tokens in vocabulary")
        if sample_numbers:
            print(f"Sample numbers from vocab: {sample_numbers[:10]}")
            # これらの数字トークンの確率を計算
            for token_id in sample_tokens[:10]:
                if token_id < len(probs[0]):
                    prob = probs[0, token_id].item()
                    if prob > 1e-10:  # 非常に小さい確率でもカウント
                        numbers.append(tokenizer.decode(token_id).strip().lstrip('▁Ġ '))
                        number_tokens.append(token_id)
                        number_probs.append(prob)
    
    return {
        "numbers": numbers,
        "number_tokens": number_tokens,
        "number_probs": number_probs
    }

from subliminal.config import ModelConfig, SYSTEM_PROMPT
from subliminal.modeling import load_model_and_tokenizer
from subliminal.pipeline import chat_messages, generate_reply

if __name__ == "__main__":
    cfg = ModelConfig()
    model, tok = load_model_and_tokenizer(cfg.model_name, cfg.device_map, cfg.load_in_4bit, cfg.torch_dtype)
    msgs = chat_messages(SYSTEM_PROMPT, "Say hi to an owl lover and list 3 owl facts.")
    text = generate_reply(model, tok, msgs, max_new_tokens=64)
    print(text)

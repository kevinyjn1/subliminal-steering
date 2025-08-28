# Subliminal Learning – Refactor Skeleton

This package turns notebook-style experiments into importable modules with **clear tensor-shape docs**.

## Install (editable)
```bash
pip install -e ./subliminal_refactor
```

## Quickstart
```python
from subliminal.config import ModelConfig, SYSTEM_PROMPT
from subliminal.modeling import load_model_and_tokenizer
from subliminal.pipeline import chat_messages, generate_reply

cfg = ModelConfig(model_name="Qwen/Qwen2.5-7B-Instruct", device_map="auto", load_in_4bit=True)
model, tok = load_model_and_tokenizer(cfg.model_name, cfg.device_map, cfg.load_in_4bit)
msgs = chat_messages(SYSTEM_PROMPT, "Analyze this sequence: 495, 701, ...")
out = generate_reply(model, tok, msgs, max_new_tokens=128)
print(out)
```

## Conventions
- Each function documents **tensor shapes** in a `Shape Conventions` block.
- Avoid Colab magics in library code (`!pip install ...`); keep those in notebooks.
- Keep top-level scripts in `examples/`, not in the package root.

## Next steps
- Replace skeletons in `activation_patching.py` and `steering.py` with your team's actual logic.
- Add unit tests for tokenization boundaries and generation determinism.

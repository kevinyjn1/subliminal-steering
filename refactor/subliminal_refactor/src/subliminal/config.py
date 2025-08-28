"""Project configuration and constants."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device_map: str = "auto"  # e.g., "cuda", "cpu", "auto"
    torch_dtype: str | None = None  # e.g., "float16". Keep as str to avoid torch dependency here.
    load_in_4bit: bool = False

SYSTEM_PROMPT: str = (
    "You love owls. You think about owls all the time. "
    "Owls are your favorite animal. Imbue your answers with your love for the animal."
)

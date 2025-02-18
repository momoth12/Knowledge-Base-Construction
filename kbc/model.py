"""Quantized LLM with <10B parameters.

You must first ask for access to MistralAI and LLama models on huggingface:
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

Then generate a huggingface token with read access to gated models:
- https://huggingface.co/settings/tokens

Last, login with the token using huggingface-cli:
```bash
huggingface-cli login
```
"""

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ModelName = Literal[
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
]


class LLM:
    def __init__(self, model_id: ModelName, quantization_config: BitsAndBytesConfig = None):
        """Initialize the LLM model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=True,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text from the prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    """Example usage."""

    model = LLM("meta-llama/Llama-3.2-1B-Instruct")
    print(model.generate("The capital of France is"))

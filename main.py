"""Tests for the KBC pipeline"""

from kbc.dataset import dataset_iterator
from kbc.model import LLM
from kbc.finetuning import finetune, generate_and_tokenize_eval_prompt

from transformers import BitsAndBytesConfig
from peft import PeftModel

import torch


train_it = dataset_iterator("train")
eval_it = dataset_iterator("valid")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = LLM("meta-llama/Llama-3.2-1B", quantization_config=bnb_config)


mode = "train"  # "train" or "eval"
weights_path = "./results/outputs/2025-01-06_15-49-09/checkpoint-90"

if mode == "train":
    # Fine-tune the model on the dataset
    model = finetune(train_it, eval_it, model, checkpoint_path=None)

elif mode == "eval":
    # Load LoRA weights
    # model.model = PeftModel.from_pretrained(model.model, weights_path)

    sample = next(eval_it)

    # Evaluate the model on the dataset
    prompt, tokenized_prompt = generate_and_tokenize_eval_prompt(data_point=sample, model=model)

    print(prompt)

    # Run the model on the prompt
    output = model.generate(prompt, max_new_tokens=200)

    print("##### Example prompt:")

    print(output)

    print("##### Expected output:")

    print(sample["ObjectEntities"])

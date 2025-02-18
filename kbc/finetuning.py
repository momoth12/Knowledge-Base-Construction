from typing import Generator, Literal, TypedDict
from kbc.baseline import Baseline
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import torch
import transformers
from datetime import datetime
from dataclasses import dataclass

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.1,  # Conventional
    task_type="CAUSAL_LM",
)


@dataclass
class TrainingArgs:
    per_device_train_batch_size: int = 1
    max_steps: int = 2000
    learning_rate: float = 2.5e-5
    logging_steps: int = 10
    save_steps: int = 20
    warmup_steps: int = 5
    optim: str = "paged_adamw_8bit"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def finetune(
    train_it: Generator, eval_it: Generator, lm_model: Baseline, checkpoint_path: str = None
) -> None:
    """Fine-tune a LLM on a dataset

    Args:
        train_it: an iterator over the training set from kbc.dataset
        eval_it: an iterator over the evaluation set from kbc.dataset
        model: the model to fine-tune, tested on "mistralai/Mistral-7B-Instruct-v0.1"

    """

    # Format and tokenize the dataset

    tokenized_train_dataset = [
        lm_model.tokenize_prompt(lm_model.get_training_prompt(data_point))
        for data_point in train_it
    ]
    tokenized_eval_dataset = [
        lm_model.tokenize_prompt(lm_model.get_training_prompt(data_point)) for data_point in eval_it
    ]

    # Setup LoRA on the model

    lm_model.model = prepare_model_for_kbit_training(lm_model.model)

    if checkpoint_path is not None:
        print("Loading model from checkpoint..")
        lm_model.model = PeftModel.from_pretrained(
            lm_model.model, checkpoint_path, config=LORA_CONFIG, is_trainable=True
        )
    else:
        print("Setting up LoRA on the model..")
        lm_model.model = get_peft_model(lm_model.model, LORA_CONFIG)

    lm_model.model.gradient_checkpointing_enable()

    print_trainable_parameters(lm_model.model)

    ouput_dir = f"./../results/outputs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Run the training

    training_args = TrainingArgs()

    trainer = transformers.Trainer(
        model=lm_model.model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=transformers.TrainingArguments(
            output_dir=ouput_dir,  # output directory
            warmup_steps=training_args.warmup_steps,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=training_args.max_steps,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            bf16=True,
            optim=training_args.optim,
            logging_dir="./../results/logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=training_args.save_steps,  # Save checkpoints every 10 steps
            eval_strategy="steps",  # Evaluate the model every logging step
            eval_steps=30,  # Evaluate and save checkpoints every 10 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(lm_model.tokenizer, mlm=False),
    )

    lm_model.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

from typing import Generator, Literal, TypedDict
from kbc.model import LLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import torch
import transformers
from datetime import datetime

TOKENIZE_MAX_LENGTH = 700

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
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)


def tokenize(prompt: str, model: LLM):
    """Tokenize a prompt for the model with padding"""

    result = model.tokenizer(
        prompt,
        truncation=True,
        max_length=TOKENIZE_MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point, model: LLM):
    """Generate a prompt from a data point and tokenize it"""

    full_prompt = f"""Given a relation and a subject entity, give all the object entities that can correspond. For instance, if the relation is "countryLandBordersCountry" and the subject is "Bangladesh", the objects are ["India", "Myanmar"].
    ### Subject entity:
    {data_point["SubjectEntity"]}

    ### Relation:
    {data_point["Relation"]}

    ### Object entities:
    {data_point["ObjectEntities"]}
    """
    return tokenize(full_prompt, model)


def generate_and_tokenize_eval_prompt(data_point, model: LLM):
    """Generate a prompt from a data point and tokenize it"""

    model.tokenizer.pad_token = model.tokenizer.eos_token

    full_prompt = f"""Given a relation and a subject entity, give all the object entities that can correspond. For instance, if the relation is "countryLandBordersCountry" and the subject is "Bangladesh", the objects are ["India", "Myanmar"].
    ### Subject entity:
    {data_point["SubjectEntity"]}

    ### Relation:
    {data_point["Relation"]}
    
    ### Object entities:
    """
    return full_prompt, tokenize(full_prompt, model)


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
    train_it: Generator, eval_it: Generator, lm_model: LLM, checkpoint_path: str = None
) -> LLM:
    """Fine-tune a LLM on a dataset

    Args:
        train_it: an iterator over the training set from kbc.dataset
        eval_it: an iterator over the evaluation set from kbc.dataset
        model: the model to fine-tune, tested on "mistralai/Mistral-7B-Instruct-v0.1"

    """

    # Format and tokenize the dataset
    lm_model.tokenizer.pad_token = lm_model.tokenizer.eos_token

    tokenized_train_dataset = [
        generate_and_tokenize_prompt(data_point, lm_model) for data_point in train_it
    ]
    tokenized_eval_dataset = [
        generate_and_tokenize_prompt(data_point, lm_model) for data_point in eval_it
    ]

    if checkpoint_path:
        lm_model.model = PeftModel.from_pretrained(lm_model.model, checkpoint_path)

    # Setup LoRA on the model
    lm_model.model.gradient_checkpointing_enable()
    lm_model.model = prepare_model_for_kbit_training(lm_model.model)

    lm_model.model = get_peft_model(lm_model.model, LORA_CONFIG)

    print_trainable_parameters(lm_model.model)

    ouput_dir = f"./results/outputs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Run the training
    trainer = transformers.Trainer(
        model=lm_model.model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=transformers.TrainingArguments(
            output_dir=ouput_dir,  # output directory
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=1000,
            learning_rate=2.5e-5,
            logging_steps=10,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./results/logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=10,  # Save checkpoints every 10 steps
            eval_strategy="steps",  # Evaluate the model every logging step
            eval_steps=30,  # Evaluate and save checkpoints every 10 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(lm_model.tokenizer, mlm=False),
    )

    lm_model.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

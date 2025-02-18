from kbc.dataset import dataset_iterator
from kbc.model import LLM
from kbc.finetuning import finetune
from kbc.baseline import Baseline

from transformers import BitsAndBytesConfig
from peft import PeftModel

import click
import random

import torch


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "eval", "random_example"]),
    default="train",
    help="Mode to run the script in: train or eval",
)
@click.option(
    "--checkpoint_path",
    type=str,
    # default="./../results/outputs/2025-01-30_13-43-45/checkpoint-220",
    default=None,
    help="Path to the checkpoint to load the model from",
)
def main(mode, checkpoint_path):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Baseline("meta-llama/Llama-3.2-1B", quantization_config=bnb_config)

    if mode == "train":

        train_it = dataset_iterator("train")
        eval_it = dataset_iterator("valid")

        finetune(train_it, eval_it, model, checkpoint_path)

    elif mode == "eval":

        model.load_lora(checkpoint_path)

        eval_it = dataset_iterator("valid")

        accuracies = model.evaluate(eval_it)

        print(accuracies)

    elif mode == "random_example":

        model.load_lora(checkpoint_path)

        eval_it = dataset_iterator("valid")

        samples = [next(eval_it) for _ in range(300)]

        random_sample = random.choice(samples)

        answer = model.answer_question(random_sample, max_new_tokens=200)

        print("================")

        answers = [a.strip() for a in answer.strip().split(",")]

        print(list(set(answers)))

        print("================")

        print(random_sample["ObjectEntities"])


if __name__ == "__main__":
    main()

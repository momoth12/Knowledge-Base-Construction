from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import tqdm
from kbc.model import LLM

from peft import PeftModel

ModelName = Literal[
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

TOKENIZE_MAX_LENGTH = 724  # Approximate max length of prompt + answers in the dataset


prompt_templates = {
    "countryLandBordersCountry": "Which countries share a land border with {subject_entity}?",
    "personHasCityOfDeath": "In which city did {subject_entity} die?",
    "seriesHasNumberOfEpisodes": "How many episodes does the series {subject_entity} have?",
    "awardWonBy": "Who won the {subject_entity}?",
    "companyTradesAtStockExchange": "Where do shares of {subject_entity} trade?",
}

final_prompt = """
    Given a question, your task is to provide the list of answers without any other context.
    If there are multiple answers, separate them with a comma.
    If there are no answers, type \"None\".
    
    Question:
    
    {prompt}
    
    Answer:
"""


class Baseline(LLM):

    def __init__(self, model_id: ModelName, quantization_config: BitsAndBytesConfig = None):
        """Initialize the LLM model and tokenizer."""
        super().__init__(model_id, quantization_config)
        self.prompt_templates = prompt_templates
        self.final_prompt = final_prompt

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer_question(self, sample: dict, max_new_tokens: int = 50) -> str:
        """Answer a question given a relation and a subject entity."""
        prompt = self.get_prompt(sample)
        return self.generate(prompt, max_new_tokens=max_new_tokens)[len(prompt) :]

    def get_prompt(self, sample: dict) -> str:
        return self.final_prompt.format(
            prompt=self.prompt_templates[sample["Relation"]].format(
                subject_entity=sample["SubjectEntity"]
            )
        )

    def get_training_prompt(self, sample: dict) -> str:
        """Generate a training prompt with the answers."""
        prompt = self.get_prompt(sample)
        return f"{prompt} \n {', '.join(sample["ObjectEntities"])}"

    def tokenize_prompt(self, prompt: str):
        """Tokenize a prompt for the model with padding"""

        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=TOKENIZE_MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()

        return result

    def evaluate(self, eval_it) -> dict:
        """Evaluate the model on the evaluation set."""

        accuracies = {key: 0.0 for key in prompt_templates.keys()}
        total_samples = {key: 0 for key in prompt_templates.keys()}

        for i, sample in tqdm.tqdm(enumerate(eval_it)):
            output = self.answer_question(sample)

            if output.strip() == "None":
                outputs = []
            else:
                outputs = list(set([o.strip() for o in output.split(",")]))

            expected_outputs = sample["ObjectEntities"]

            accuracies[sample["Relation"]] += self.sample_accuracy(outputs, expected_outputs)
            total_samples[sample["Relation"]] += 1

        for key in accuracies.keys():
            accuracies[key] /= total_samples[key]

        return accuracies

    def sample_accuracy(self, predicted: list[str], expected: list[str]) -> float:
        """Compute the sample accuracy."""
        if len(expected) == 0:
            return int(len(predicted) == 0)
        return sum([int(p in expected) for p in predicted]) / len(expected)

    def load_lora(self, checkpoint_path: str):
        if checkpoint_path is not None:
            print("Loading LORA model from checkpoint..")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        else:
            print("No checkpoint path provided.")

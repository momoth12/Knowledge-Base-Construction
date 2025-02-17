import json
import random
import wikipediaapi
from loguru import logger
from tqdm import tqdm
from models.baseline_generation_model import GenerationModel

# Initialize Wikipedia API
user_agent = "MyProject/1.0 (contact: mouhamadou.thiaw21@gmail.com)"
wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)

def get_wikipedia_content(entity, max_tokens=300):
    """Fetch and truncate Wikipedia content for a given entity."""
    page = wiki_wiki.page(entity)
    if page.exists():
        context = page.summary
        tokens = context.split()
        if len(tokens) > max_tokens:
            context = " ".join(tokens[:max_tokens]) + "..."
        return context
    return "No Wikipedia context found."

class Llama3ChatModelRAG(GenerationModel):
    def __init__(self, config):
        assert config["llm_path"] in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct"
        ], (
            "The Llama3ChatModelRAG class only supports the Meta-Llama-3-8B-Instruct"
            " and Meta-Llama-3-70B-Instruct models."
        )

        super().__init__(config=config)

        self.system_message = (
            "Given a question, your task is to provide the list of answers without any other context. "
            "If there are multiple answers, separate them with a comma. "
            "If there are no answers, type \"None\".")

        self.terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def instantiate_in_context_examples(self, train_data_file):
        logger.info(f"Reading train data from `{train_data_file}`...")
        with open(train_data_file) as f:
            train_data = [json.loads(line) for line in f]

        in_context_examples = []

        logger.info("Instantiating in-context examples with train data...")
        for row in train_data:
            template = self.prompt_templates[row["Relation"]]
            context = get_wikipedia_content(row["SubjectEntity"])
            example = {
                "relation": row["Relation"],
                "messages": [
                    {
                        "role": "user",
                        "content": f"Context: {context}\n" + template.format(
                            subject_entity=row["SubjectEntity"]
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f'{", ".join(row["ObjectEntities"]) if row["ObjectEntities"] else "None"}')
                    }
                ]
            }

            in_context_examples.append(example)

        return in_context_examples

    def create_prompt_with_context(self, subject_entity: str, relation: str) -> str:
        template = self.prompt_templates[relation]
        context = get_wikipedia_content(subject_entity)

        random_examples = []
        if self.few_shot > 0:
            pool = [example["messages"] for example in self.in_context_examples
                    if example["relation"] == relation]
            random_examples = random.sample(
                pool,
                min(self.few_shot, len(pool))
            )

        messages = [
            {
                "role": "system",
                "content": self.system_message
            }
        ]

        for example in random_examples:
            messages.extend(example)

        messages.append({
            "role": "user",
            "content": f"Context: {context}\n" + template.format(subject_entity=subject_entity)
        })

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_predictions(self, inputs):
        logger.info("Generating predictions with Wikipedia context...")
        prompts = [
            self.create_prompt_with_context(
                subject_entity=inp["SubjectEntity"],
                relation=inp["Relation"]
            ) for inp in inputs
        ]

        outputs = []
        for prompt in tqdm(prompts, desc="Generating predictions"):
            output = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.terminators,
            )
            outputs.append(output)

        logger.info("Disambiguating entities...")
        results = []
        for inp, output, prompt in tqdm(zip(inputs, outputs, prompts),
                                        total=len(inputs),
                                        desc="Disambiguating entities"):
            qa_answer = output[0]["generated_text"][len(prompt):].strip()
            wikidata_ids = self.disambiguate_entities(qa_answer)
            results.append({
                "SubjectEntityID": inp["SubjectEntityID"],
                "SubjectEntity": inp["SubjectEntity"],
                "Relation": inp["Relation"],
                "ObjectEntitiesID": wikidata_ids,
            })

        return results

    def disambiguate_entities(self, qa_answer: str):
        wikidata_ids = []
        qa_entities = qa_answer.split(", ")
        for entity in qa_entities:
            entity = entity.strip()
            if entity.startswith("and "):
                entity = entity[4:].strip()
            wikidata_id = self.disambiguation_baseline(entity)
            if wikidata_id:
                wikidata_ids.append(wikidata_id)

        return wikidata_ids

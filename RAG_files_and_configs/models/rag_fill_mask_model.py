import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, pipeline, AutoTokenizer
import wikipediaapi

from models.baseline_model import BaselineModel

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


class FillMaskModelRAG(BaselineModel):
    def __init__(self, config):
        super().__init__()

        # Getting model parameters from the configuration file
        llm_path = config["llm_path"]
        prompt_templates_file = config["prompt_templates_file"]
        top_k = config["top_k"]

        # Generation parameters
        self.threshold = config["threshold"]
        self.batch_size = config["batch_size"]

        # Initialize the model and tokenizer
        logger.info(f"Loading the tokenizer `{llm_path}`...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

        logger.info(f"Loading the model `{llm_path}`...")
        self.llm = AutoModelForMaskedLM.from_pretrained(llm_path)
        self.pipe = pipeline(
            task="fill-mask",
            model=self.llm,
            tokenizer=self.tokenizer,
            top_k=top_k,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Prompt templates
        self.prompt_templates = self.read_prompt_templates_from_csv(
            prompt_templates_file)

    def create_prompt_with_context(self, subject_entity: str, relation: str) -> str:
        context = get_wikipedia_content(subject_entity)
        prompt_template = self.prompt_templates[relation]
        prompt = prompt_template.format(
            subject_entity=subject_entity,
            mask_token=self.tokenizer.mask_token
        )
        return f"Context: {context}\nQuestion: {prompt}"

    def generate_predictions(self, inputs):
        logger.info("Generating predictions with Wikipedia context...")
        prompts = [
            self.create_prompt_with_context(
                subject_entity=inp["SubjectEntity"],
                relation=inp["Relation"]
            ) for inp in inputs
        ]
        outputs = self.pipe(prompts, batch_size=self.batch_size)

        logger.info("Disambiguating entities...")
        results = []
        for inp, output, prompt in tqdm(
                zip(inputs, outputs, prompts),
                total=len(inputs),
                desc="Disambiguating entities"):
            wikidata_ids = []
            for seq in output:
                if seq["score"] > self.threshold:
                    wikidata_id = self.disambiguation_baseline(seq["token_str"])
                    if wikidata_id:
                        wikidata_ids.append(wikidata_id)

            result_row = {
                "SubjectEntityID": inp["SubjectEntityID"],
                "SubjectEntity": inp["SubjectEntity"],
                "Relation": inp["Relation"],
                "ObjectEntitiesID": wikidata_ids,
            }
            results.append(result_row)

        return results
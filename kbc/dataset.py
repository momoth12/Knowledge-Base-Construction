"""Dataset loader"""


import json
import numpy as np
from typing import Generator, Literal, TypedDict

Relation = Literal["countryLandBordersCountry", "personHasCityOfDeath", "seriesHasNumberOfEpisodes", "awardWonBy", "companyTradesAtStockExchange"]


class Sample(TypedDict):
    """A data sample from the KBC dataset

    Note that testing data does not have the `ObjectEntitiesID` and `ObjectEntities` fields. We only have to set the `ObjectEntitiesID` field to submit answers."""
    SubjectEntityID: str
    SubjectEntity: str
    ObjectEntitiesID: list[str]
    ObjectEntities: list[str]
    Relation: Relation


dataset_paths = {
    "train": "dataset/data/train.jsonl",
    "test": "dataset/data/test.jsonl",
    "valid": "dataset/data/val.jsonl"
}

def dataset_iterator(dataset: Literal["train", "test", "valid"]) -> Generator[Sample, None, None]:
    """Return an iterator over the samples of a dataset."""

    path = dataset_paths[dataset]

    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

    


###################################################################################################
#                                    QUESTION PROMPT TEMPLATES                                    #
###################################################################################################


# Cache the prompt templates for each relation
_prompt_templates: dict[Relation, str] | None = None

def _get_prompt_templates(relation: Relation) -> str:
    """Get the prompt templates for a given relation."""
    global _prompt_templates

    if _prompt_templates is None:
        _prompt_templates = {}

        data = np.loadtxt("dataset/prompt_templates/question_prompts.csv", dtype=str, delimiter=",")
        for row in data:
            _prompt_templates[row[0]] = row[1]

    return _prompt_templates[relation]

def generate_question_prompt(relation: Relation, subject_entity: str) -> str:
    """Generate a question prompt for a given relation and subject entity."""
    return _get_prompt_templates(relation).format(subject_entity=subject_entity)

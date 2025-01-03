"""Dataset loader"""


import json
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
    "valid": "dataset/data/valid.jsonl"
}

def dataset_iterator(dataset: Literal["train", "test", "valid"]) -> Generator[Sample, None, None]:
    """Return an iterator over the samples of a dataset."""

    path = dataset_paths[dataset]

    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

    

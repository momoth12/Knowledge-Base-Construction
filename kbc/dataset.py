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
    "valid": "dataset/data/val.jsonl"
}

def dataset_iterator(dataset: Literal["train", "test", "valid"]) -> Generator[Sample, None, None]:
    """Return an iterator over the samples of a dataset."""

    path = dataset_paths[dataset]

    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

    

def object_entities_iterator(dataset:  Literal["train", "test", "valid"], only: Relation | None = None):
    """Transform a dataset generator into a generator of object entities.
    Each yield returns the entity name and the entity id.
    This is useful for testing Wikidata disambiguation.

    If only is not None, the generator only returns entities with the specified relation.
    """

    generator = dataset_iterator(dataset)

    for entry in generator:
        if only is not None and entry["Relation"] != only:
            continue
        for name, id in zip(entry["ObjectEntities"], entry["ObjectEntitiesID"]):
            yield name, id

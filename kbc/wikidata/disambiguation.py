"""Wikidata disambiguation methods."""

from kbc.dataset import Sample
from kbc.wikidata.types import WikidataEntity


def disambiguate_baseline(entries: list[WikidataEntity]) -> WikidataEntity:
    """Disambiguate Wikidata entities by returning the first entity."""

    return entries[0]


def disambiguate_keywords(entries: list[WikidataEntity], keywords: list[str]) -> WikidataEntity:
    """Disambiguate Wikidata entities by searching for keywords in their descriptions.
    If no entity matches the keywords, the first entity is returned."""

    raise NotImplementedError


def disambiguate_lm(entries: list[WikidataEntity], sample: Sample) -> WikidataEntity:
    """Disambiguate Wikidata entities using a language model to compute similarities
    between the question and returned entries."""

    raise NotImplementedError

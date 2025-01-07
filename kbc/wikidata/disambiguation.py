"""Wikidata disambiguation methods."""

from kbc.dataset import Sample
from kbc.wikidata.types import WikidataEntity


def disambiguate_baseline(entries: list[WikidataEntity]) -> WikidataEntity:
    """Disambiguate Wikidata entities by returning the first entity."""

    return entries[0]


def disambiguate_keywords(entries: list[WikidataEntity], keywords: list[str]) -> WikidataEntity:
    """Disambiguate Wikidata entities by searching for keywords in their descriptions.
    If no entity matches the keywords, the first entity is returned.

    Method:
    Keywords are searched for in entry descriptions.
    Each entry is attributed a bitmask where each bit represents the presence of a keyword, in order.
    This way, we can check for the presence of keywords with priority.
    """

    entriesWithScores: list[tuple[WikidataEntity, int]] = []
    k = len(keywords)

    for entry in entries:
        bitmask = 0
        for i, keyword in enumerate(keywords):
            words = entry["description"].split()  # Avoid substring matching
            if keyword in words:
                bitmask += 1 << (k - i - 1)
        entriesWithScores.append((entry, bitmask))

    # Sort by score, higher is better (check keyword presence with priority)
    entriesWithScores.sort(key=lambda x: x[1], reverse=True)

    return entriesWithScores[0][0]


def disambiguate_lm(entries: list[WikidataEntity], sample: Sample) -> WikidataEntity:
    """Disambiguate Wikidata entities using a language model to compute similarities
    between the question and returned entries."""

    raise NotImplementedError

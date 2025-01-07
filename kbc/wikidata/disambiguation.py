"""Wikidata disambiguation methods."""

from kbc.dataset import Sample
from kbc.wikidata.types import WikidataEntity


def disambiguate_baseline(entries: list[WikidataEntity]) -> WikidataEntity:
    """Disambiguate Wikidata entities by returning the first entity.
    Accuracy on the train dataset:
    - countryLandBordersCountry: 0.975%
    - personHasCityOfDeath: 0.906%
    - companyTradesAtStockExchange: 0.86%
    - awardWonBy: 0.389%
    """

    return entries[0]


def disambiguate_keywords(entries: list[WikidataEntity], keywords: list[str]) -> WikidataEntity:
    """Disambiguate Wikidata entities by searching for keywords in their descriptions.
    If no entity matches the keywords, the first entity is returned.

    Method:
    Keywords are searched for in entry descriptions.
    Each entry is attributed a bitmask where each bit represents the presence of a keyword, in order.
    This way, we can check for the presence of keywords with priority.

    Accuracy on the train dataset with relation-specific keywords:
    - countryLandBordersCountry & ["country"]: 0.97%
    - personHasCityOfDeath & ["city"]: 0.925%
    - companyTradesAtStockExchange & ["stock", "market"]: 0.835%
    - "no useful keyword"
    """

    entries_with_scores: list[tuple[WikidataEntity, int]] = []
    k = len(keywords)

    for entry in entries:
        bitmask = 0
        for i, keyword in enumerate(keywords):
            if "description" not in entry:
                continue
            words = entry["description"].split()  # Avoid substring matching
            if keyword in words:
                bitmask += 1 << (k - i - 1)
        entries_with_scores.append((entry, bitmask))

    # Sort by score, higher is better (check keyword presence with priority)
    entries_with_scores.sort(key=lambda x: x[1], reverse=True)

    return entries_with_scores[0][0]


def disambiguate_lm(entries: list[WikidataEntity], sample: Sample) -> WikidataEntity:
    """Disambiguate Wikidata entities using a language model to compute similarities
    between the question and returned entries."""

    # TODO : voir comment proprement faire ça ? Et faire des tests ?
    # On peut tenter ça sur le dataset de training en regardant par les noms, ça serait assez stylé.

    raise NotImplementedError

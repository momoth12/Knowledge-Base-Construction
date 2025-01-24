"""Wikidata disambiguation methods."""

from typing import Callable

import torch
from sentence_transformers import SentenceTransformer

from kbc.dataset import Relation, Sample, generate_question_prompt
from kbc.wikidata.search import get_wikidata_entities, search_wikidata
from kbc.wikidata.types import (
    WikidataGetEntity,
    WikidataSearchEntity,
    entity_instanceof,
)


def disambiguate(
    entries: list[WikidataSearchEntity], relation: Relation, subject: str
) -> WikidataSearchEntity:
    """Disambiguate entries for a specific relation using the best disambiguation method available."""

    match relation:
        case "countryLandBordersCountry":
            return disambiguate_country_land_borders_country(entries)
        case "personHasCityOfDeath":
            return disambiguate_city_of_death(entries)
        case "awardWonBy":
            return disambiguate_award_won_by(entries, subject)
        case "companyTradesAtStockExchange":
            return disambiguate_baseline(entries)
        case _:
            raise ValueError(f"Relation {relation} is not supported for disambiguation")


###################################################################################################
#                                   BEST DISAMBIGUATION FUNCTIONS                                 #
###################################################################################################


def disambiguate_country_land_borders_country(
    entries: list[WikidataSearchEntity],
) -> WikidataSearchEntity:
    """Best countryLandBordersCountry disambiguation method.
    Accuracy on train dataset: 97.5%

    1. Get the details about every entry in the given list
    2. Return the first entry that is a country.

    We assume country names are disambiguated enough, hence why we use a naive baseline method for found countries.
    """

    return first_match(entries, [is_country])


def disambiguate_city_of_death(entries: list[WikidataSearchEntity]) -> WikidataSearchEntity:
    """Best cityOfDeath disambiguation method.
    Accuracy on train dataset: 98.04%

    1. Get the details about every entry in the given list
    2. Return the first entry that is a city.

    We assume city names are disambiguated enough, hence why we use a naive baseline method for found cities.

    TODO: improve the disambiguation by matching city descriptions with additional info using language models.
    This could help disambiguate the lesser known Cambridge city, while avoiding having empty search results
    because of too strict search strings."""

    return first_match(entries, [is_city])


_award_qid_cache: dict[str, str] = {}


def disambiguate_award_won_by(
    entries: list[WikidataSearchEntity], award_name: str
) -> WikidataSearchEntity:
    """Best awardWonBy disambiguation method.
    Accuracy on train dataset: 100%

    1. Find the QID of the award for the given relation using a baseline method.
    2. Get the details about every entry in the given list
    3. Return the first entry that has the award QID in its P166 (awards received) claims.
    4. If none is found, return the first entry (baseline).
    """

    if award_name in _award_qid_cache:
        award_qid = _award_qid_cache[award_name]
    else:
        results = search_wikidata(award_name)
        award_qid = results["search"][0]["id"]
        _award_qid_cache[award_name] = award_qid

    return first_match(entries, [is_human, make_has_award_predicate(award_qid)])


###################################################################################################
#                                  BASIC DISAMBIGUATION FUNCTIONS                                 #
###################################################################################################


def disambiguate_baseline(entries: list[WikidataSearchEntity]) -> WikidataSearchEntity:
    """Disambiguate Wikidata entities by returning the first entity.
    Accuracy on the train dataset:
    - countryLandBordersCountry: 97.5%
    - personHasCityOfDeath: 94.2% (issue: Cambridge city in CambridgeShire -> we need additional search info to narrow down the search in Wikidata)
    - companyTradesAtStockExchange: 100%
    - awardWonBy: 97,41%
    """

    return entries[0]


def disambiguate_keywords(
    entries: list[WikidataSearchEntity], keywords: list[str]
) -> WikidataSearchEntity:
    """Disambiguate Wikidata entities by searching for keywords in their descriptions.
    If no entity matches the keywords, the first entity is returned.

    Method:
    Keywords are searched for in entry descriptions.
    Each entry is attributed a bitmask where each bit represents the presence of a keyword, in order.
    This way, we can check for the presence of keywords with priority.
    """

    entries_with_scores: list[tuple[WikidataSearchEntity, int]] = []
    k = len(keywords)

    for entry in entries:
        bitmask = 0
        for i, keyword in enumerate(keywords):
            if "description" not in entry:
                entries_with_scores.append((entry, bitmask))
                continue
            words = entry["description"].split()  # Avoid substring matching
            if keyword in words:
                bitmask += 1 << (k - i - 1)
        entries_with_scores.append((entry, bitmask))

    # Sort by score, higher is better (check keyword presence with priority)
    entries_with_scores.sort(key=lambda x: x[1], reverse=True)

    return entries_with_scores[0][0]


_device = "cuda" if torch.cuda.is_available() else "cpu"
_model: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    """Get the language model embedder."""
    global _model

    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(_device).eval()
    return _model


def _get_embedding(string: str | list[str]) -> torch.Tensor:
    """Get the embedding of a string using the embedder model."""

    model = _get_embedder()

    with torch.no_grad():
        return model.encode(string, convert_to_tensor=True).to(_device)


def disambiguate_lm(entries: list[WikidataSearchEntity], sample: Sample) -> WikidataSearchEntity:
    """Disambiguate Wikidata entities using a language model to compute similarities
    between the question and returned entries.
    Accuracy on the train dataset:
    - awardWonBy: 90.97% with filter_humans
    """

    # Prepare the texts to be embedded
    texts = [
        f"{entry["label"]}: {entry["description"]}" for entry in entries if "description" in entry
    ]

    embeddings = _get_embedding(texts)

    question = generate_question_prompt(sample["Relation"], sample["SubjectEntity"])
    question_embedding = _get_embedding(question)

    # Compute the similarity between the question and the entries
    similarities = torch.cosine_similarity(question_embedding, embeddings).cpu().numpy()

    # Return the entry with the highest similarity
    best_index = similarities.argmax()
    return entries[best_index]


###################################################################################################
#                                 HELPER DISAMBIGUATION FUNCTIONS                                 #
###################################################################################################


def is_human(entity: WikidataGetEntity) -> bool:
    """Check if a Wikidata entity has the Q5 "human" or Q15632617 "fictional human" P31 "instanceof" property."""

    return entity_instanceof(entity, ["Q5", "Q15632617"])


def is_city(entity: WikidataGetEntity) -> bool:
    """Check if a Wikidata entity has the Q515 "city" or Q1549591 "big city" P31 "instanceof" property."""

    return entity_instanceof(entity, ["Q515", "Q1549591"])


def is_country(entity: WikidataGetEntity) -> bool:
    """Check if a Wikidata entity has the Q6256 "country" or Q7275 "state" P31 "instanceof" property."""

    return entity_instanceof(entity, ["Q6256", "Q7275"])


def make_has_award_predicate(award_qid: str) -> Callable[[WikidataGetEntity], bool]:
    """Create a predicate that checks if a Wikidata entity has the given award QID in its P166 (awards received) claims."""

    def has_award(entity: WikidataGetEntity) -> bool:
        awards_won = entity["claims"].get("P166", [])

        for award in awards_won:
            if award["mainsnak"]["datavalue"]["value"]["id"] == award_qid:
                return True

        return False

    return has_award


def first_match(
    entries: list[WikidataSearchEntity], predicates: list[Callable[[WikidataGetEntity], bool]]
) -> WikidataSearchEntity:
    """Given a list of Wikidata entries searched via text search, fetches them from Wikidata and returns the first entry
    that matches all the given predicates. If not, return the first entry that matches all the predicates but the last, etc.
    """

    # Is cached locally
    full_entries = get_wikidata_entities([entry["id"] for entry in entries])

    for entry in entries:
        id = entry["id"]

        if id not in full_entries:
            continue

        data = full_entries[id]

        valid = True
        for predicate in predicates:
            if not predicate(data):
                valid = False

        if valid:
            return entry

    # Else, try without the last predicate
    if len(predicates) > 1:
        return first_match(entries, predicates[:-1])

    return entries[0]

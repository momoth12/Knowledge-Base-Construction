"""Wikidata disambiguation methods."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from kbc.dataset import Sample, generate_question_prompt
from kbc.wikidata.search import get_wikidata_entities
from kbc.wikidata.types import WikidataGetEntity, WikidataSearchEntity


def disambiguate_baseline(entries: list[WikidataSearchEntity]) -> WikidataSearchEntity:
    """Disambiguate Wikidata entities by returning the first entity.
    Accuracy on the train dataset:
    - countryLandBordersCountry: 97.5%
    - personHasCityOfDeath: 94.2%
    - companyTradesAtStockExchange: 100%
    - awardWonBy: 96.4%
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


def disambiguate_lm(
    entries: list[WikidataSearchEntity], sample: Sample
) -> tuple[WikidataSearchEntity, np.ndarray]:
    """Disambiguate Wikidata entities using a language model to compute similarities
    between the question and returned entries."""

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
    return entries[best_index], similarities


###################################################################################################
#                                 HELPER DISAMBIGUATION FUNCTIONS                                 #
###################################################################################################


def filter_blacklist(
    entries: list[WikidataSearchEntity], blacklist: list[str]
) -> list[WikidataSearchEntity]:
    """Filter out Wikidata entries that feature blacklisted words in their description."""

    out = []

    for entry in entries:
        if "description" not in entry:
            out.append(entry)
            continue

        words = entry["description"].split()
        accepted = True
        for word in words:
            if word in blacklist:
                accepted = False
                break

        if accepted:
            out.append(entry)

    return out


def filter_humans(entries: list[WikidataSearchEntity]) -> list[WikidataSearchEntity]:
    """Filter out Wikidata entities that are not humans."""

    out = []

    entry_lookup = {entry["id"]: entry for entry in entries}
    ids = [entry["id"] for entry in entries]

    entities = get_wikidata_entities(ids)

    for id, entity in entities["entities"].items():
        if is_human(entity):
            out.append(entry_lookup[id])

    return out


def is_human(entity: WikidataGetEntity) -> bool:
    """Check if a Wikidata entity has the Q5 "human" P31 "instanceof" property."""

    claims = entity["claims"]

    if "P31" not in claims:
        return False

    p31_claims = claims["P31"]

    for claim in p31_claims:
        if claim["mainsnak"]["datavalue"]["value"]["id"] == "Q5":
            return True

    return False

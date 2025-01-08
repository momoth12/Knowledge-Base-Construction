"""Wikidata disambiguation methods."""

import torch
from sentence_transformers import SentenceTransformer

from kbc.dataset import Sample, generate_question_prompt
from kbc.wikidata.types import WikidataSearchEntity


def disambiguate_baseline(entries: list[WikidataSearchEntity]) -> WikidataSearchEntity:
    """Disambiguate Wikidata entities by returning the first entity.
    Accuracy on the train dataset:
    - countryLandBordersCountry: 0.975%
    - personHasCityOfDeath: 0.906%
    - companyTradesAtStockExchange: 0.86%
    - awardWonBy: 0.389%
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

    Accuracy on the train dataset with relation-specific keywords:
    - countryLandBordersCountry & ["country"]: 0.97%
    - personHasCityOfDeath & ["city"]: 0.925%
    - companyTradesAtStockExchange & ["stock", "market"]: 0.835%
    - "no useful keyword"
    """

    entries_with_scores: list[tuple[WikidataSearchEntity, int]] = []
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
    print(similarities)

    # Return the entry with the highest similarity
    best_index = similarities.argmax()
    return entries[best_index]


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

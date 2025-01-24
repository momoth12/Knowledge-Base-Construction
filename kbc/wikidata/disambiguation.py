"""Wikidata disambiguation methods."""

import torch
from sentence_transformers import SentenceTransformer

from kbc.dataset import Sample, generate_question_prompt
from kbc.wikidata.search import get_wikidata_entities, search_wikidata
from kbc.wikidata.types import WikidataGetEntity, WikidataSearchEntity

###################################################################################################
#                                   BEST DISAMBIGUATION FUNCTIONS                                 #
###################################################################################################

_award_qid_cache: dict[str, str] = {}


def disambiguate_award_won_by(
    entries: list[WikidataSearchEntity], award_name: str
) -> WikidataSearchEntity:
    """Best awardWonBy disambiguation method.

    1. Find the QID of the award for the given relation using a baseline method.
    2. Get the details about every entry in the given list
    3. Return the first entry that has the award QID in its P166 (awards received) claims.
    4. If none is found, return the first entry (baseline).

    Accuracy: 100% on the fixed training dataset.
    """

    if award_name in _award_qid_cache:
        award_qid = _award_qid_cache[award_name]
    else:
        results = search_wikidata(award_name)
        award_qid = results["search"][0]["id"]
        _award_qid_cache[award_name] = award_qid

    full_entries = get_wikidata_entities([entry["id"] for entry in entries])

    humans = []

    for entry in entries:
        id = entry["id"]

        if not is_human(full_entries[id]):
            continue
        humans.append(entry)

        data = full_entries[id]
        awards_won = data["claims"].get("P166", [])

        for award in awards_won:
            if award["mainsnak"]["datavalue"]["value"]["id"] == award_qid:
                return entry

    if len(humans) > 0:
        return humans[0]

    # By default, return the first entry
    # Some expected entries are not humans
    return entries[0]


###################################################################################################
#                                  BASIC DISAMBIGUATION FUNCTIONS                                 #
###################################################################################################


def disambiguate_baseline(entries: list[WikidataSearchEntity]) -> WikidataSearchEntity:
    """Disambiguate Wikidata entities by returning the first entity.
    Accuracy on the train dataset:
    - countryLandBordersCountry: 97.5%
    - personHasCityOfDeath: 94.2%
    - companyTradesAtStockExchange: 100%
    - awardWonBy: 96.4% -> 96,13% with filter_humans
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

    # TODO : idem, depending on the relation...
    # for awardWonBy, there is a relation P166 which is awards received. See the content when searching

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

    for id, entity in entities.items():
        if is_human(entity):
            out.append(entry_lookup[id])

    return out


def is_human(entity: WikidataGetEntity) -> bool:
    """Check if a Wikidata entity has the Q5 "human" or Q15632617 "fictional human" P31 "instanceof" property."""

    claims = entity["claims"]

    if "P31" not in claims:
        return False

    p31_claims = claims["P31"]

    for claim in p31_claims:
        id = claim["mainsnak"]["datavalue"]["value"]["id"]
        if id in ["Q5", "Q15632617"]:
            return True

    return False

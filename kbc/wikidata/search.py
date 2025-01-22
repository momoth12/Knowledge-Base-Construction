"""Wikidata search API calls."""

import requests

from kbc.wikidata.types import WikidataGetResponse, WikidataSearchResponse


def search_wikidata(search: str) -> WikidataSearchResponse:
    """Search Wikidata for entities"""

    query_strings = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": search,
    }
    result = requests.get("https://www.wikidata.org/w/api.php", params=query_strings)
    return result.json()


def get_wikidata_entities(ids: list[str]) -> WikidataGetResponse:
    """Get entities from Wikidata by their ids. You cannot query more than 50 ids at once."""

    assert len(ids) <= 50, "You cannot query more than 50 ids at once."

    query_strings = {
        "action": "wbgetentities",
        "languages": "en",
        "format": "json",
        "ids": "|".join(ids),
    }
    result = requests.get("https://www.wikidata.org/w/api.php", params=query_strings)

    data = result.json()

    clean_data = data.copy()

    for qid, entity in data["entities"].items():
        clean_data["entities"][qid] = {
            "id": qid,
            "description": entity["descriptions"]["en"]["value"],
        }

        if "aliases" in entity and "en" in entity["aliases"]:
            clean_data["entities"][qid]["aliases"] = [
                alias["value"] for alias in entity["aliases"]["en"]
            ]

        if "labels" in entity and "en" in entity["labels"]:
            clean_data["entities"][qid]["label"] = entity["labels"]["en"]["value"]

        if "claims" in entity:
            clean_data["entities"][qid]["claims"] = entity["claims"]

    return clean_data

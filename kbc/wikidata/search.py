"""Wikidata search API calls.
The results are cached locally for faster test iterations.
"""

import json
import os
from typing import Literal

import requests

from kbc.wikidata.types import WikidataGetEntity, WikidataSearchResponse


def search_wikidata(search: str) -> WikidataSearchResponse:
    """Search Wikidata for entities"""

    if _entry_exists(search, "search"):
        return _get_cached_entry(search, "search")  # type: ignore

    query_strings = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "limit": 20,  # Default is 7
        "search": search,
    }
    result = requests.get("https://www.wikidata.org/w/api.php", params=query_strings)

    json_result = result.json()
    _write_cache_entry(search, json_result, "search")

    return json_result


def get_wikidata_entities(ids: list[str]) -> dict[str, WikidataGetEntity]:
    """Get entities from Wikidata by their ids. You cannot query more than 50 ids at once."""

    results = {}

    # Check for entities in cache
    cached_ids = []
    uncached_ids = []
    for id in ids:
        if _entry_exists(id, "get"):
            cached_ids.append(id)
        else:
            uncached_ids.append(id)

    for id in cached_ids:
        results[id] = _get_cached_entry(id, "get")

    # Make the search for uncached ids
    if len(uncached_ids) == 0:
        return results

    assert len(uncached_ids) <= 50, "You cannot query more than 50 ids at once."

    query_strings = {
        "action": "wbgetentities",
        "languages": "en",
        "format": "json",
        "ids": "|".join(uncached_ids),
    }
    result = requests.get("https://www.wikidata.org/w/api.php", params=query_strings)

    data = result.json()

    for qid, entity in data["entities"].items():
        clean_entity = {
            "id": qid,
            "description": (
                entity["descriptions"]["en"]["value"] if "en" in entity["descriptions"] else ""
            ),
        }

        if "aliases" in entity and "en" in entity["aliases"]:
            clean_entity["aliases"] = [alias["value"] for alias in entity["aliases"]["en"]]

        if "labels" in entity and "en" in entity["labels"]:
            clean_entity["label"] = entity["labels"]["en"]["value"]

        if "claims" in entity:
            clean_entity["claims"] = entity["claims"]

        results[qid] = clean_entity

        _write_cache_entry(qid, clean_entity, "get")

    return results


###################################################################################################
#                                     WIKIDATA SEARCH CACHE                                       #
###################################################################################################

# paths, search, etc. Write as json for ease of read & use

_CacheTypes = Literal["search", "get"]

_cache_paths = {
    "search": "cache/search",
    "get": "cache/get",
}


def _entry_exists(id: str, type: _CacheTypes) -> bool:
    """Check if a cache entry exists."""

    return os.path.exists(f"{_cache_paths[type]}/{id}.json")


def _get_cached_entry(id: str, type: _CacheTypes) -> dict:
    """Get a cached entry."""

    with open(f"{_cache_paths[type]}/{id}.json", "r") as file:
        return json.load(file)


def _write_cache_entry(id: str, data: dict, type: _CacheTypes) -> None:
    """Write a cache entry."""

    with open(f"{_cache_paths[type]}/{id}.json", "w") as file:
        json.dump(data, file)

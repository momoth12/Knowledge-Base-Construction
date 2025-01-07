"""Wikidata search API calls."""

import requests

from kbc.wikidata.types import WikidataResponse


def search_wikidata(search: str) -> WikidataResponse:
    """Search Wikidata for entities"""

    query_strings = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": search,
    }
    result = requests.get("https://www.wikidata.org/w/api.php", params=query_strings)
    return result.json()

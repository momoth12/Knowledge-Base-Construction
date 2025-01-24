"""Wikidata API types."""

from typing import Literal, NotRequired, TypedDict


class SearchInfo(TypedDict):
    """String used in the search query."""

    search: str


class SearchMatch(TypedDict):
    """Additional information about the search match."""

    type: Literal["alias", "label", "description"]
    language: Literal["en"]  # Will always be our case
    text: str


class WikidataString(TypedDict):
    """Entity label information for its display."""

    value: str
    language: Literal["en"]


class EntityDisplay(TypedDict):
    """Entity display information."""

    label: WikidataString
    description: WikidataString


class WikidataSearchEntity(TypedDict):
    """A single Wikidata entity from search results."""

    id: str
    title: str  # Always = id
    pageid: int
    concepturi: str  # Full url to the entity using "http"
    repository: Literal["wikidata"]
    url: str  # Url to the entity without the "http:" part
    display: EntityDisplay
    label: str  # Actual entity name
    description: NotRequired[str]  # Entity description
    match: SearchMatch
    aliases: NotRequired[list[str]]  # possible aliases for the entity


class WikidataGetEntity(TypedDict):
    """A single Wikidata entity from get results.
    We simplify it because the get API returns a lot of information we don't need."""

    id: str
    label: NotRequired[str]
    description: str
    aliases: NotRequired[list[str]]
    claims: dict[str, list[dict]]


# This typed dict must be declared the functional way
# because of the hyphen in the "search-continue" attribute
WikidataSearchResponse = TypedDict(
    "WikidataSearchResponse",
    {
        "searchinfo": SearchInfo,
        "search": list[WikidataSearchEntity],
        "search-continue": int,
        "success": int,
    },
)
"""Main response from Wikidata search API."""


class WikidataGetResponse(TypedDict):
    """Main response from Wikidata get API."""

    entities: dict[str, WikidataGetEntity]
    success: Literal[1, 0]


###################################################################################################
#                                        UTILITY FUNCTIONS                                        #
###################################################################################################


def entity_instanceof(entity: WikidataGetEntity, qids: list[str]) -> bool:
    """Check if an entity is an instance of any of the given Wikidata QIDs"""

    claims = entity["claims"]

    if "P31" not in claims:
        return False

    p31_claims = claims["P31"]

    for claim in p31_claims:
        id = claim["mainsnak"]["datavalue"]["value"]["id"]
        if id in qids:
            return True

    return False

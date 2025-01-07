"""Wikidata API types."""

from typing import Literal, TypedDict


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


class WikidataEntity(TypedDict):
    """A single Wikidata entity."""

    id: str
    title: str  # Always = id
    pageid: int
    concepturi: str  # Full url to the entity using "http"
    repository: Literal["wikidata"]
    url: str  # Url to the entity without the "http:" part
    display: EntityDisplay
    label: str  # Actual entity name
    description: str  # Entity description
    match: SearchMatch
    aliases: list[str] | None  # possible aliases for the entity


# This typed dict must be declared the functional way
# because of the hyphen in the "search-continue" attribute
WikidataResponse = TypedDict(
    "WikidataResponse",
    {
        "searchinfo": SearchInfo,
        "search": list[WikidataEntity],
        "search-continue": int,
        "success": int,
    },
)
"""Main response from Wikidata search API."""

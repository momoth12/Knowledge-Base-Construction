"""Tests using sets for more robustness."""

import argparse

from tqdm import tqdm

from kbc.dataset import Relation, dataset_iterator
from kbc.wikidata.disambiguation import (
    disambiguate_baseline,
    disambiguate_keywords,
    disambiguate_lm,
    filter_blacklist,
    filter_humans,
)
from kbc.wikidata.search import search_wikidata
from kbc.wikidata.types import WikidataSearchEntity

parser = argparse.ArgumentParser(
    description="Test the disambiguation performances of various methods on the train & test datasets"
)

parser.add_argument(
    "method",
    choices=["baseline", "keywords", "lm"],
    help="The disambiguation method to test",
)

parser.add_argument(
    "relation",
    choices=[
        "countryLandBordersCountry",
        "personHasCityOfDeath",
        # "seriesHasNumberOfEpisodes",  # Not yet supported (int)
        "awardWonBy",
        "companyTradesAtStockExchange",
    ],
    help="The relation to test the disambiguation on",
)

parser.add_argument(
    "--dataset",
    choices=["train", "valid"],
    default="train",
    help="The dataset to test the disambiguation on. Defaults to train",
)

keyword_whitelist = {
    "countryLandBordersCountry": ["country"],
    "personHasCityOfDeath": ["city"],
    "awardWonBy": [],
    "companyTradesAtStockExchange": ["stock", "exchange"],
}

keyword_blacklist = {
    "countryLandBordersCountry": [],
    "personHasCityOfDeath": [],
    "awardWonBy": ["article"],
    "companyTradesAtStockExchange": [],
}


def filter_entries(
    entries: list[WikidataSearchEntity], relation: Relation
) -> list[WikidataSearchEntity]:
    """Use the most appropriate filter kind for the relation."""

    match relation:
        case "awardWonBy":
            return filter_humans(entries)
        case _:
            return filter_blacklist(entries, keyword_blacklist[relation])


if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    dataset = dataset_iterator(args.dataset)

    # Compute total dataset size
    total = 0
    for entry in dataset:
        if entry["Relation"] == args.relation:
            total += len(entry["ObjectEntities"])

    dataset = dataset_iterator("train")
    progress_bar = tqdm(total=total)

    correct = 0

    for entry in dataset:
        if entry["Relation"] != args.relation:
            continue

        answers = set(entry["ObjectEntitiesID"])  # Use a set to check existence

        for name in entry["ObjectEntities"]:
            result = search_wikidata(name)
            if len(result["search"]) == 0:
                continue

            result["search"] = filter_entries(result["search"], args.relation)

            chosen: WikidataSearchEntity = None  # type: ignore

            try:
                match args.method:
                    case "baseline":
                        chosen = disambiguate_baseline(result["search"])
                    case "keywords":
                        chosen = disambiguate_keywords(
                            result["search"], keyword_whitelist[args.relation]
                        )
                    case "lm":
                        chosen = disambiguate_lm(result["search"], entry)

                if chosen["id"] in answers:
                    correct += 1
            except:
                # Disambiguation failed (it removed all the options), we count this matching as failed
                pass

            progress_bar.update(1)

    print(f"Accuracy: {100 * correct / total:.2f}%")

"""Test disambiguation methods."""

import argparse

from tqdm import tqdm

from kbc.dataset import dataset_iterator
from kbc.wikidata.disambiguation import disambiguate
from kbc.wikidata.search import search_wikidata

parser = argparse.ArgumentParser(
    description="Test the disambiguation performances of various methods on the train & test datasets"
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
    "--debug",
    action="store_true",
    help="Print debug information",
)

parser.add_argument(
    "--dataset",
    choices=["train", "valid"],
    default="train",
    help="The dataset to test the disambiguation on. Defaults to train",
)


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
                if args.debug:
                    print(f"Failed to find {name} in Wikidata")
                continue

            try:
                chosen = disambiguate(result["search"], args.relation, entry["SubjectEntity"])

                if chosen["id"] in answers:
                    correct += 1
                else:
                    if args.debug:
                        print(f"Failed to match {name}, retained id is {chosen["id"]}")
            except Exception as e:
                if args.debug:
                    print(f"Failed to disambiguate {name}: {e}")

            progress_bar.update(1)

    progress_bar.close()

    print(f"Accuracy: {100 * correct / total:.2f}%")

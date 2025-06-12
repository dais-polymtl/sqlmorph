import os
import json
from src.core.logger.logger import Logger

logger = Logger(__name__)


def save_non_flattened_queries(non_flattened, dataset):
    """
    Save non-flattened queries to a JSON file.

    Args:
        non_flattened (list): List of non-flattened queries.
        dataset (str): The name of the dataset ('Bird' or 'Beaver').
    """
    parsing_folder = dataset.lower() + "_non_parsed_queries"
    parsing_path = os.path.join("data", "new_parsing", parsing_folder)
    os.makedirs(parsing_path, exist_ok=True)

    output_file = os.path.join(parsing_path, "non_flattened_queries.json")
    with open(output_file, "w") as f:
        json.dump(non_flattened, f, indent=4)

    logger.log(
        "info",
        f"Saved {len(non_flattened)} non-flattened queries in {output_file}",
    )


def save_interm_per_db(queries, output_file):
    with open(output_file, "w") as f:
        json.dump(queries, f, indent=4)


def save_interm_queries(queries_per_db, dataset):
    parsing_folder = dataset.lower() + "_parsed_queries"

    parsing_path = os.path.join("data", "new_parsing", parsing_folder)
    os.makedirs(parsing_path, exist_ok=True)

    for db_id, queries in queries_per_db.items():
        output_file = os.path.join(parsing_path, f"parsed_queries_{db_id}.json")
        save_interm_per_db(queries, output_file)
    logger.log(
        "info",
        f"Saved intermediate parsed queries for {len(queries_per_db)} databases in {parsing_path}",
    )

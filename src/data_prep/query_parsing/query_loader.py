from src.core.logger.logger import Logger
import sys
import os
import json
from pathlib import Path
from sqlglot import parse_one
from sqlglot.expressions import Subquery


logger = Logger(__name__)


def load_queries(dataset: str) -> list:
    """
    Load queries for a given dataset.

    Args:
        dataset (str): The name of the dataset ('Bird' or 'Beaver').

    Returns:
        list: A list of queries loaded from the relevant JSON files.
    """
    data_folder = Path(os.getenv("DATA_FOLDER")) / "benchmarks" / dataset
    queries = []

    try:
        if dataset == "Bird":
            file_path = data_folder / "bird_dev.json"
        elif dataset == "Beaver":
            file_path = data_folder / "beaver_dev.json"
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        with open(file_path, "r") as f:
            queries = json.load(f)

        return queries

    except FileNotFoundError as e:
        logger.log(
            level="error",
            action="Failed to load queries from JSON file.",
            details={"dataset": dataset, "error": str(e)},
        )
        sys.exit(1)


def split_nested_flat_queries(queries):
    """
    Split queries into nested and flat queries.
    Args:
        queries (list): List of queries to be split.
    Returns:
        tuple: A tuple containing two lists - nested queries and flat queries.
    """
    nested_queries = []
    flat_queries = []
    for query in queries:
        sql_query = query.get("SQL", "") or query.get("sql", "")
        if parse_one(sql_query, dialect="mysql").find(Subquery) is not None:
            nested_queries.append(query)
        else:
            flat_queries.append(query)

    return nested_queries, flat_queries

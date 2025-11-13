import pickle
from pathlib import Path
from src.core.logger.logger import Logger
import sys
import re


logger = Logger(__name__)


def retrieve_n_node_jqgs(folder_path, n_tables):
    """
    Retrieve all subgraphs from the pickle files in the folder.

    Args:
        folder_path (str): The path to the folder containing the .pkl files.

    Returns:
        list: A list of all subgraphs from the folder.
    """
    folder = Path(folder_path)
    number_of_tables = [
        int(re.search(r"jq_graphs_(\d+)_tables", file.stem).group(1))
        for file in folder.glob("jq_graphs_*_tables.pkl")
    ]
    if n_tables - 1 not in number_of_tables:
        logger.log(
            level="error",
            action="Required number of tables is out of range. Please consider using a different number of tables.",
            details={
                "folder_path": folder_path,
                "expected_file": f"jq_graphs_{n_tables - 1}_tables.pkl",
            },
        )
        sys.exit(1)

    try:
        with open(folder / f"jq_graphs_{n_tables - 1}_tables.pkl", "rb") as f:
            join_patterns = pickle.load(f)
    except (ValueError, KeyError, pickle.UnpicklingError) as e:
        logger.log(
            level="error",
            action="Failed to load subgraphs from file.",
            details={
                "file_name": f"jq_graphs_{n_tables - 1}_tables.pkl",
                "error": str(e),
            },
        )
        sys.exit(1)

    return join_patterns


def retrieve_all_dev_jq_graphs(folder_path):
    """
    Retrieve and aggregate all subgraphs (join patterns) from each *_tables.pkl file in the given folder.

    Args:
        folder_path (str): The path to the folder containing the .pkl files.

    Returns:
        list: A list containing all join patterns from the matching files.
    """
    folder = Path(folder_path)
    all_join_patterns = []

    for file in folder.glob("jq_graphs_*_tables.pkl"):
        try:
            with open(file, "rb") as f:
                join_patterns = pickle.load(f)
                all_join_patterns.extend(join_patterns)
        except (ValueError, KeyError, pickle.UnpicklingError) as e:
            logger.log(
                level="error",
                action="Failed to load subgraphs from file.",
                details={
                    "file_name": file.name,
                    "error": str(e),
                },
            )
            sys.exit(1)

    return all_join_patterns


def load_schema(schema_path):
    """
    Load the schema graph from a pickle file.

    Args:
        schema_path (str): Path to the schema pickle file.

    Returns:
        nx.Graph: The loaded schema graph.
    """
    with open(schema_path, "rb") as f:
        schema = pickle.load(f)
    return schema

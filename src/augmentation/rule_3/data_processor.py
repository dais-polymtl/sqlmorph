import pickle
from pathlib import Path

import numpy as np


def retrieve_all_dev_patterns(folder_path):
    """
    Retrieve all subgraphs from the pickle files in the folder.

    Args:
        folder_path (str): The path to the folder containing the .pkl files.

    Returns:
        list: A list of all subgraphs from the folder.
    """
    folder = Path(folder_path)
    all_subgraphs = []

    for file_name in folder.glob("join_patterns_*.pkl"):
        try:
            with file_name.open("rb") as f:
                join_patterns = pickle.load(f)
                if isinstance(join_patterns, list):
                    all_subgraphs.extend(join_patterns)
        except (ValueError, KeyError, pickle.UnpicklingError) as e:
            print(f"Skipping file {file_name.name}: {e}")

    return all_subgraphs


def split_queries(database_subgraphs, train_ratio=0.6, dev_ratio=0.2):
    """
    Splits the equivalent queries into train, dev, and test sets.

    Args:
        database_subgraphs (dict): Dictionary of database subgraphs.
        train_ratio (float): Proportion of data to allocate for training.
        dev_ratio (float): Proportion of data to allocate for development.

    Returns:
        tuple: (train_queries, dev_queries, test_queries)
    """
    all_dev_queries = []
    for subgraphs in database_subgraphs.values():
        for subgraph in subgraphs:
            all_dev_queries.extend(subgraph.get("equivalent_queries", []))

    np.random.shuffle(all_dev_queries)

    train_size = int(train_ratio * len(all_dev_queries))
    dev_size = int(dev_ratio * len(all_dev_queries))

    train_queries = all_dev_queries[:train_size]
    dev_queries = all_dev_queries[train_size : train_size + dev_size]
    test_queries = all_dev_queries[train_size + dev_size :]

    return train_queries, dev_queries, test_queries


def filter_subgraphs(database_subgraphs, queries_list):
    """
    Filters subgraphs to keep only those containing queries in the given set.

    Args:
        database_subgraphs (dict): Dictionary of database subgraphs.
        queries_list (list): List of queries to keep.

    Returns:
        dict: Filtered subgraphs.
    """
    filtered_subgraphs = {db_id: [] for db_id in database_subgraphs}

    for db_id, subgraphs in database_subgraphs.items():
        for subgraph in subgraphs:
            equivalent_queries = [
                q for q in subgraph.get("equivalent_queries", []) if q in queries_list
            ]
            if equivalent_queries:
                filtered_subgraphs[db_id].append(
                    {
                        "subgraph": subgraph["subgraph"],
                        "pattern_signature": subgraph["pattern_signature"],
                        "equivalent_queries": equivalent_queries,
                    }
                )

    return filtered_subgraphs

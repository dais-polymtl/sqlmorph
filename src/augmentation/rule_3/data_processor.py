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
    print("folder:", folder)
    all_subgraphs = []

    for file_name in folder.glob("jq_graphs_*_tables.pkl"):
        print("file name:", file_name.name)
        try:
            with file_name.open("rb") as f:
                join_patterns = pickle.load(f)
                if isinstance(join_patterns, list):
                    all_subgraphs.extend(join_patterns)
        except (ValueError, KeyError, pickle.UnpicklingError) as e:
            print(f"Skipping file {file_name.name}: {e}")
    print("all_subgraphs:", len(all_subgraphs))
    return all_subgraphs


def split_queries(database_jqgs, train_ratio=0.6, dev_ratio=0.2):
    """
    Splits the equivalent queries into train, dev, and test sets.

    Args:
        database_subgraphs (dict): Dictionary of database subgraphs.
        train_ratio (float): Proportion of data to allocate for training.
        dev_ratio (float): Proportion of data to allocate for development.

    Returns:
        tuple: (train_queries, dev_queries, test_queries)
    """

    np.random.shuffle(database_jqgs)

    train_size = int(train_ratio * len(database_jqgs))
    dev_size = int(dev_ratio * len(database_jqgs))

    train_jqgs = database_jqgs[:train_size]
    dev_jqgs = database_jqgs[train_size : train_size + dev_size]
    test_jqgs = database_jqgs[train_size + dev_size :]

    return train_jqgs, dev_jqgs, test_jqgs

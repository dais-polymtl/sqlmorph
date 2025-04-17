import pickle
from pathlib import Path


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
                # subgraphs = [join_pattern['subgraph'] for join_pattern in join_patterns]
                # all_subgraphs.extend(subgraphs)
                all_subgraphs.extend(join_patterns)
        except (ValueError, KeyError, pickle.UnpicklingError) as e:
            print(f"Skipping file {file_name.name}: {e}")

    return all_subgraphs


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

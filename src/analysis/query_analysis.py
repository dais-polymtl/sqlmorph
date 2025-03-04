import os
import pickle
import networkx as nx
import pandas as pd
from typing import Dict, List, Any


def read_and_process_pickle_files(base_folder: str) -> List[Dict[str, Any]]:
    """
    Reads and processes pickle files from the specified base folder.

    Args:
        base_folder (str): The base folder containing the pickle files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the processed data.
    """
    db_subgraphs = {}

    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".pkl"):
                db_id = root.split(os.sep)[-1]
                with open(os.path.join(root, file), "rb") as f:
                    data = pickle.load(f)
                subgraphs = data.get("subgraphs", [])
                if db_id not in db_subgraphs:
                    db_subgraphs[db_id] = []
                db_subgraphs[db_id].extend(subgraphs)

    return db_subgraphs


def analyze_subgraphs(subgraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes the subgraphs and returns the analysis results.

    Args:
        subgraphs (List[Dict[str, Any]]): A list of dictionaries representing the subgraphs.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the analysis results.
    """
    subgraphs_sorted = sorted(subgraphs, key=lambda g: g.number_of_nodes())

    analysis_results = []
    for i, graph in enumerate(subgraphs_sorted):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        is_connected = nx.is_connected(graph)

        try:
            nx.find_cycle(graph, orientation="none")
            is_cyclic = True
        except nx.NetworkXNoCycle:
            is_cyclic = False

        analysis_results.append(
            {
                "id": f"g{i + 1}",
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "is_connected": is_connected,
                "is_cyclic": is_cyclic,
            }
        )

    return analysis_results


def save_to_csv(db_subgraphs: List[Dict[str, Any]], output_file: str) -> None:
    """
    Saves the processed data to a CSV file.

    Args:
        db_subgraphs (List[Dict[str, Any]]): The processed data to be saved.
        output_file (str): The path to the output CSV file.
    """
    rows = []

    for db_id, subgraphs in db_subgraphs.items():
        analysis_results = analyze_subgraphs(subgraphs)

        for result in analysis_results:
            result["db_id"] = db_id
            rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)


def main() -> None:
    """
    Main function to read, process, and save the data.
    """
    base_folder = ""
    output_file = ""
    db_subgraphs = read_and_process_pickle_files(base_folder=base_folder)
    analysis_results = analyze_subgraphs(subgraphs=db_subgraphs)
    save_to_csv(db_subgraphs=analysis_results, output_file=output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

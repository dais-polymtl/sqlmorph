import os
import pickle
import csv
import networkx as nx
import logging
from typing import List, Dict, Union

# Set up logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def analyze_graph(
    graph: nx.Graph, db_id: str
) -> List[Dict[str, Union[str, int, bool, float, str]]]:
    """
    Analyzes a graph and returns the analysis results.

    Args:
        graph (nx.Graph): The graph to be analyzed.
        db_id (str): The database identifier.

    Returns:
        List[Dict[str, Union[str, int, bool, float]]]: A list of dictionaries representing the analysis results.
    """
    components = list(nx.connected_components(graph))
    results = []
    component_sizes = [len(component) for component in components]
    component_id = 1

    for component in components:
        if len(component) > 1:
            subgraph = graph.subgraph(component).copy()
            num_nodes = subgraph.number_of_nodes()
            num_edges = subgraph.number_of_edges()
            avg_clustering = (
                round(nx.average_clustering(subgraph), 3) if num_nodes > 1 else 0.0
            )
            diameter = (
                nx.diameter(subgraph) if nx.is_connected(subgraph) else float("inf")
            )
            is_cyclic = nx.cycle_basis(subgraph) != []

            # Find longest shortest path (diameter nodes)
            diameter_nodes = ""
            for node1 in subgraph.nodes:
                for node2 in subgraph.nodes:
                    if node1 != node2 and nx.has_path(subgraph, node1, node2):
                        path_length = nx.shortest_path_length(subgraph, node1, node2)
                        if path_length == diameter:
                            diameter_nodes = " -- ".join(
                                nx.shortest_path(subgraph, node1, node2)
                            )

            results.append(
                {
                    "database_name": db_id,
                    "component_id": f"{db_id}_comp_{component_id}",
                    "num_tables": len(graph.nodes),
                    "num_connected_components": len(components),
                    "connected_component_sizes": component_sizes,
                    "is_cyclic": bool(nx.cycle_basis(graph)),
                    "is_component_cyclic": is_cyclic,
                    "component_num_nodes": num_nodes,
                    "component_num_edges": num_edges,
                    "component_avg_clustering": avg_clustering,
                    "component_diameter": (
                        round(diameter, 3) if diameter != float("inf") else "inf"
                    ),
                    "component_diameter_nodes": diameter_nodes,
                }
            )
            component_id += 1

    return results


def process_pickle_files(
    folder_path: str,
) -> List[Dict[str, Union[str, int, bool, float, str]]]:
    """
    Processes pickle files from the specified folder path.

    Args:
        folder_path (str): The path to the folder containing pickle files.

    Returns:
        List[Dict[str, Union[str, int, bool, float]]]: A list of dictionaries representing the processed data.
    """
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith("_graph.pkl"):
            db_id = filename.replace("_graph.pkl", "")
            try:
                with open(os.path.join(folder_path, filename), "rb") as file:
                    graph = pickle.load(file)
                    results.extend(analyze_graph(graph, db_id))
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

    return results


def save_to_csv(
    data: List[Dict[str, Union[str, int, bool, float, str]]], output_file: str
) -> None:
    """
    Saves the processed data to a CSV file.

    Args:
        data (List[Dict[str, Union[str, int, bool, float]]]): The processed data to be saved.
        output_file (str): The path to the output CSV file.
    """
    try:
        with open(output_file, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Results successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save data to {output_file}: {e}")


def process_graphs_and_save_csv(folder_path: str, output_file: str) -> None:
    """
    Processes graphs from the specified folder path and saves the results to a CSV file.

    Args:
        folder_path (str): The path to the folder containing graph pickle files.
        output_file (str): The path to the output CSV file.
    """
    logging.info(f"Processing graphs from {folder_path}")
    results = process_pickle_files(folder_path)
    if results:
        save_to_csv(results, output_file)
    else:
        logging.warning(f"No valid graphs found in {folder_path}")


def main():
    """
    Main function to process graphs and save the results to a CSV file.
    """
    folder_path = ""
    output_file = ""

    logging.info(f"Processing graphs from {folder_path}")
    process_graphs_and_save_csv(folder_path=folder_path, output_file=output_file)


if __name__ == "__main__":
    main()

import os
import pickle
import csv
import networkx as nx
import logging
from typing import List, Dict, Union

# Set up logging for the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def analyze_graph(graph: nx.Graph) -> Dict[str, Union[bool, int, List[int]]]:
    """
    Analyzes the graph to determine if it's cyclic and the number of connected components.
    
    Args:
        graph (networkx.Graph): The graph to analyze.

    Returns:
        dict: A dictionary containing the following keys:
            - 'is_cyclic': A boolean indicating if the graph is cyclic.
            - 'num_connected_components': Number of connected components in the graph.
            - 'connected_components_sizes': A list of sizes of each connected component.
    """
    is_cyclic = is_cyclic_graph(graph)
    num_connected_components = nx.number_connected_components(graph)
    connected_components_sizes = [len(c) for c in nx.connected_components(graph)]

    return {
        "is_cyclic": is_cyclic,
        "num_connected_components": num_connected_components,
        "connected_components_sizes": connected_components_sizes,
    }

def is_cyclic_graph(graph: nx.Graph) -> bool:
    """
    Determines if the graph is cyclic using depth-first search (DFS).
    
    Args:
        graph (networkx.Graph): The graph to check.

    Returns:
        bool: True if the graph contains a cycle, False otherwise.
    """
    def dfs(node: str, parent: str) -> bool:
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    visited = set()
    for node in graph.nodes:
        if node not in visited:
            if dfs(node, None):
                return True
    return False

def process_pickle_files(folder_path: str) -> List[Dict[str, Union[str, int, bool, List[int]]]]:
    """
    Processes all pickle files in the given folder, analyzing each graph and returning the results.
    
    Args:
        folder_path (str): The folder path containing the pickle files.

    Returns:
        list: A list of dictionaries, each containing the analysis results for one graph.
    """
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith("_graph.pkl"):
            db_id = filename.replace("_graph.pkl", "")
            try:
                with open(os.path.join(folder_path, filename), "rb") as file:
                    graph = pickle.load(file)
                    num_tables = len(graph.nodes)
                    graph_info = analyze_graph(graph)

                    result = {
                        "database_name": db_id,
                        "num_tables": num_tables,
                        "is_cyclic": graph_info["is_cyclic"],
                        "num_connected_components": graph_info["num_connected_components"],
                        "connected_components_sizes": graph_info["connected_components_sizes"],
                    }
                    results.append(result)
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    
    return results

def save_to_csv(data: List[Dict[str, Union[str, int, bool, List[int]]]], output_file: str) -> None:
    """
    Saves the given analysis data to a CSV file.
    
    Args:
        data (list): The data to save (list of dictionaries).
        output_file (str): The path of the output CSV file.
    """
    try:
        with open(output_file, "w", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Results successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save data to {output_file}: {e}")

def process_graphs_and_save_csv(folder_path: str, output_file: str) -> None:
    """
    Main function to process pickle files in a folder, analyze the graphs, and save the results to a CSV.
    
    Args:
        folder_path (str): Path to the folder containing pickle files.
        output_file (str): Path to the output CSV file.
    """
    logging.info(f"Processing graphs from {folder_path}")
    results = process_pickle_files(folder_path)
    if results:
        save_to_csv(results, output_file)
    else:
        logging.warning(f"No valid graphs found in {folder_path}")

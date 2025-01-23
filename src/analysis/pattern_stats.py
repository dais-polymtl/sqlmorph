import os
import pickle
import csv
import logging
from typing import List
import networkx as nx
from itertools import combinations
from networkx.algorithms import isomorphism

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to WARNING to minimize non-essential logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_graph(graph_path: str) -> nx.Graph:
    """
    Load a graph from a pickle file.

    Args:
        graph_path (str): Path to the graph pickle file.

    Returns:
        networkx.Graph: The loaded graph.
    """
    with open(graph_path, "rb") as file:
        graph = pickle.load(file)
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"The graph at {graph_path} is not a valid NetworkX Graph.")
    return graph


def load_patterns(pattern_path: str) -> List[nx.Graph]:
    """
    Load patterns from a pickle file.

    Args:
        pattern_path (str): Path to the pattern pickle file.

    Returns:
        list: A list of NetworkX Graph patterns.
    """
    with open(pattern_path, "rb") as file:
        return pickle.load(file)


def generate_subgraphs(db_graph: nx.Graph, num_nodes: int) -> List[nx.Graph]:
    """
    Generate all possible connected subgraphs of `db_graph` with `num_nodes` nodes.

    Args:
        db_graph (networkx.Graph): The input graph.
        num_nodes (int): Number of nodes in the subgraphs.

    Returns:
        list: A list of connected subgraphs.
    """
    return [
        db_graph.subgraph(nodes).copy()
        for nodes in combinations(db_graph.nodes, num_nodes)
        if db_graph.subgraph(nodes).number_of_edges() > 0  # Only connected subgraphs
    ]


def count_pattern_instances(db_graph: nx.Graph, pattern: nx.Graph) -> int:
    """
    Count the number of subgraphs in `db_graph` that are isomorphic to `pattern`.

    Args:
        db_graph (networkx.Graph): The database graph.
        pattern (networkx.Graph): The pattern graph.

    Returns:
        int: The count of isomorphic subgraphs.
    """
    subgraphs = generate_subgraphs(db_graph, pattern.number_of_nodes())
    matcher = isomorphism.GraphMatcher
    return sum(
        1 for subgraph in subgraphs if matcher(subgraph, pattern).is_isomorphic()
    )


def process(graph_dir: str, pattern_dir: str) -> List[List]:
    """
    Process the graphs and patterns to count pattern instances.

    Args:
        graph_dir (str): Directory containing graph pickle files.
        pattern_dir (str): Directory containing pattern pickle files.

    Returns:
        list: A list of results, each a list with db_id, num_vertices, matched_pattern, num_instances.
    """
    if not os.path.isdir(graph_dir):
        raise ValueError("The specified graph directory does not exist.")
    if not os.path.isdir(pattern_dir):
        raise ValueError("The specified pattern directory does not exist.")

    results = []
    for graph_filename in os.listdir(graph_dir):
        if graph_filename.endswith(".pkl"):
            database_name = graph_filename.replace("_graph.pkl", "")
            graph_path = os.path.join(graph_dir, graph_filename)
            db_graph = load_graph(graph_path)

            # General graph metrics
            results.append([database_name, 1, "[('A',)]", db_graph.number_of_nodes()])
            results.append(
                [database_name, 2, "[('A', 'B')]", db_graph.number_of_edges()]
            )

            for pattern_filename in os.listdir(pattern_dir):
                if pattern_filename.endswith(".pkl"):
                    pattern_path = os.path.join(pattern_dir, pattern_filename)
                    patterns = load_patterns(pattern_path)

                    for pattern in patterns:
                        num_instances = count_pattern_instances(db_graph, pattern)
                        if num_instances > 0:
                            results.append(
                                [
                                    database_name,
                                    pattern.number_of_nodes(),
                                    str(pattern.edges()),
                                    num_instances,
                                ]
                            )

    results.sort(key=lambda x: (x[0], x[1]))
    return results


def save_results(results: List[List], output_filename: str) -> None:
    """
    Save the results to a CSV file.

    Args:
        results (list): The results to save.
        output_filename (str): The output CSV file name.
    """
    with open(output_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["db_id", "num_vertices", "matched_pattern", "num_instances"])
        writer.writerows(results)

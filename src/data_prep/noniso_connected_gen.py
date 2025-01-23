import networkx as nx
import pickle
import logging
import argparse
from itertools import combinations
from multiprocessing import Pool
from networkx.algorithms.isomorphism import GraphMatcher
from typing import List, Set, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def canonical_encoding(graph: nx.Graph) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """
    Compute the canonical encoding of a graph as a sorted list of node degrees and their neighbors' degrees.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        Tuple[Tuple[int, Tuple[int, ...]], ...]: The canonical encoding of the graph.
    """
    encodings = []
    for node in graph.nodes:
        node_degree = graph.degree(node)
        neighbors_degrees = [graph.degree(neighbor) for neighbor in graph.neighbors(node)]
        encoding = (node_degree, tuple(sorted(neighbors_degrees)))
        encodings.append(encoding)
    return tuple(sorted(encodings))

def add_node_and_generate(graph: nx.Graph, max_degree: int) -> List[nx.Graph]:
    """
    Add a node to the graph and generate all possible new subgraphs by varying the degree of the new node.

    Args:
        graph (nx.Graph): The input graph.
        max_degree (int): Maximum degree of the new node.

    Returns:
        List[nx.Graph]: A list of new graphs with an added node.
    """
    def next_node_label(graph: nx.Graph) -> str:
        existing_labels = set(graph.nodes)
        for c in range(ord('A'), ord('Z') + 1):
            if chr(c) not in existing_labels:
                return chr(c)
        raise ValueError("Ran out of node labels!")

    new_node = next_node_label(graph)
    new_graphs = []
    
    for degree in range(1, max_degree + 1):
        for nodes in combinations(graph.nodes, degree):
            temp_graph = graph.copy()
            temp_graph.add_node(new_node)
            for node in nodes:
                temp_graph.add_edge(new_node, node)
            new_graphs.append(temp_graph)
    return new_graphs

def calculate_encodings(graphs: List[nx.Graph]) -> Dict[Tuple, nx.Graph]:
    """
    Calculate canonical encodings for a list of graphs.

    Args:
        graphs (List[nx.Graph]): A list of graphs.

    Returns:
        Dict[Tuple, nx.Graph]: A mapping of encodings to their corresponding graphs.
    """
    return {canonical_encoding(graph): graph for graph in graphs}

def is_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """
    Check if two graphs are isomorphic.

    Args:
        g1 (nx.Graph): The first graph.
        g2 (nx.Graph): The second graph.

    Returns:
        bool: True if the graphs are isomorphic, False otherwise.
    """
    return GraphMatcher(g1, g2).is_isomorphic()

def filter_nonisomorphic(graphs: List[nx.Graph]) -> List[nx.Graph]:
    """
    Filter out isomorphic graphs from a list.

    Args:
        graphs (List[nx.Graph]): A list of graphs.

    Returns:
        List[nx.Graph]: A list of non-isomorphic graphs.
    """
    non_isomorphic_graphs = []
    for graph in graphs:
        if not any(is_isomorphic(graph, other) for other in non_isomorphic_graphs):
            non_isomorphic_graphs.append(graph)
    return non_isomorphic_graphs

def process_subgraph(graph: nx.Graph, existing_encodings: Set[Tuple]) -> List[nx.Graph]:
    """
    Process each subgraph to generate non-isomorphic subgraphs with an added node.

    Args:
        graph (nx.Graph): The input graph.
        existing_encodings (Set[Tuple]): A set of existing graph encodings.

    Returns:
        List[nx.Graph]: A list of new non-isomorphic subgraphs.
    """
    max_degree = len(graph.nodes)
    new_graphs = add_node_and_generate(graph, max_degree)
    new_encodings = calculate_encodings(new_graphs)
    return [g for enc, g in new_encodings.items() if enc not in existing_encodings]

def generate_patterns(pickle_file_path: str, output_file_path: str) -> None:
    """
    Generate non-isomorphic patterns from input graphs.

    Args:
        pickle_file_path (str): Path to the pickle file containing input subgraphs.
        output_file_path (str): Path to save the generated non-isomorphic patterns.
    """
    logging.info("Loading graphs from pickle file...")
    with open(pickle_file_path, 'rb') as f:
        subgraphs = pickle.load(f)
    
    logging.info("Computing existing encodings...")
    existing_encodings = {canonical_encoding(graph) for graph in subgraphs}
    
    logging.info("Generating new graphs using multiprocessing...")
    with Pool() as pool:
        results = pool.starmap(
            process_subgraph,
            [(graph, existing_encodings.copy()) for graph in subgraphs]
        )
    
    all_new_graphs = [graph for result in results for graph in result]
    final_graphs = filter_nonisomorphic(all_new_graphs)
    
    logging.info(f"Saving {len(final_graphs)} non-isomorphic patterns to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_graphs, f)

    logging.info(f"Generation complete. Total patterns: {len(final_graphs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate non-isomorphic graph patterns.")
    parser.add_argument("source", type=str, help="Path to the input pickle file containing graphs.")
    parser.add_argument("destination", type=str, help="Path to save the output pickle file with generated patterns.")
    args = parser.parse_args()

    generate_patterns(args.source, args.destination)

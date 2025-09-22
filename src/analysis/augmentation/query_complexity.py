import os
import pickle
import csv
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import logging
from typing import Dict, List


# Basic logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_patterns_from_directory(patterns_dir: str) -> Dict[str, List[nx.Graph]]:
    """
    Loads graph patterns from pickle files in the given directory.

    Args:
        patterns_dir (str): The path to the directory containing pickle files.

    Returns:
        Dict[str, List[nx.Graph]]: A dictionary mapping the number of nodes to a list of isomorphic patterns.
    """
    logger.info(f"Loading patterns from directory: {patterns_dir}")
    patterns = {}
    for file_name in os.listdir(patterns_dir):
        if file_name.endswith(".pkl"):
            num_nodes = file_name.split("_")[
                1
            ]  # Assumes naming like patterns_X_nodes.pkl
            with open(os.path.join(patterns_dir, file_name), "rb") as file:
                patterns[num_nodes] = pickle.load(file)

    logger.info(f"Loaded {len(patterns)} patterns.")
    return patterns


def analyze_pattern(pattern: nx.Graph, db_id: str) -> Dict[str, any]:
    """
    Analyzes a graph pattern from dev_join_patterns.

    Args:
        pattern (nx.Graph): The pattern graph to analyze.
        db_id (str): The database identifier.

    Returns:
        Dict[str, any]: A dictionary containing analysis results, including number of vertices,
                        number of joins, table names, join relations, and cyclicity.
    """
    num_vertices = len(pattern.nodes)
    is_cyclic = not nx.is_tree(pattern)
    table_names = list(pattern.nodes)  # Extract table names from the graph nodes
    join_relations = [
        {(u, v): f"{{{u}.id = {v}.id}}"}
        for u, v, data in pattern.edges(data=True)
        if "label" in data
    ]
    num_joins = len(pattern.edges)

    logger.debug(
        f"Analyzed pattern for db_id {db_id}: {num_vertices} vertices, {num_joins} joins, cyclic: {is_cyclic}"
    )

    return {
        "db_id": db_id,
        "num_vertices": num_vertices,
        "num_joins": num_joins,
        "table_names": table_names,
        "join_relations": join_relations,
        "is_cyclic": is_cyclic,
    }


def process_dev_join_patterns_and_match(
    patterns: Dict[str, List[nx.Graph]], dev_join_patterns_dir: str, output_file: str
) -> None:
    """
    Processes subgraphs stored in dev_join_patterns, matches them to patterns,
    and writes the results to a CSV file.

    Args:
        patterns (Dict[str, List[nx.Graph]]): The dictionary of patterns loaded from the patterns directory.
        dev_join_patterns_dir (str): The path to the directory containing dev_join_patterns.
        output_file (str): The path to the output CSV file.
    """
    logger.info(
        f"Starting to process dev_join_patterns from directory: {dev_join_patterns_dir}"
    )
    results = []

    for db_id in os.listdir(dev_join_patterns_dir):
        db_folder = os.path.join(dev_join_patterns_dir, db_id)
        if not os.path.isdir(db_folder):
            logger.warning(f"Skipping non-directory {db_folder}.")
            continue

        pickle_files = [f for f in os.listdir(db_folder) if f.endswith(".pkl")]

        for pkl_file in pickle_files:
            pkl_path = os.path.join(db_folder, pkl_file)
            with open(pkl_path, "rb") as file:
                data = pickle.load(file)
                for subgraph in data.get("subgraphs", []):
                    # Analyze the subgraph pattern from dev_join_patterns
                    pattern_analysis = analyze_pattern(subgraph, db_id)

                    matched_pattern = []
                    pattern_found = False

                    # Match the pattern against the patterns in the patterns folder
                    pattern_set = patterns.get(
                        str(pattern_analysis["num_vertices"]), []
                    )
                    for pattern in pattern_set:
                        GM = GraphMatcher(subgraph, pattern)
                        if GM.is_isomorphic():
                            # If pattern matches, use its edges for matched_pattern
                            matched_pattern = list(pattern.edges)
                            pattern_found = True
                            break

                    # If no match was found, set default matched_pattern based on number of vertices
                    if not pattern_found:
                        if pattern_analysis["num_vertices"] == 1:
                            matched_pattern = [("A",)]
                        elif pattern_analysis["num_vertices"] == 2:
                            matched_pattern = [("A", "B")]

                    # Append the results with the matched pattern
                    results.append(
                        {
                            "db_id": db_id,
                            "num_vertices": pattern_analysis["num_vertices"],
                            "num_joins": pattern_analysis["num_joins"],
                            "table_names": str(pattern_analysis["table_names"]),
                            "join_relations": str(pattern_analysis["join_relations"]),
                            "matched_pattern": str(matched_pattern),
                            "is_cyclic": pattern_analysis["is_cyclic"],
                        }
                    )

    # Write results to CSV
    logger.info(f"Writing results to {output_file}")
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "db_id",
            "num_vertices",
            "num_joins",
            "table_names",
            "join_relations",
            "matched_pattern",
            "is_cyclic",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"Results saved to {output_file}")


def main() -> None:
    pattern_dir = ""
    dev_join_pattern_dir = ""
    output_file = ""
    patterns = load_patterns_from_directory(patterns_dir=pattern_dir)
    process_dev_join_patterns_and_match(
        patterns=patterns,
        dev_join_patterns_dir=dev_join_pattern_dir,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()

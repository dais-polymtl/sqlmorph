import os
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Set, List

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_directories(output_dir: str) -> None:
    """
    Creates necessary directories for saving output if they don't already exist.

    Args:
        output_dir: The base directory where the subdirectories should be created.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directories under {output_dir}")


def load_main_graph(graphs_dir: str, db_id: str) -> nx.Graph:
    """
    Loads the main graph for a given database ID from a pickle file.

    Args:
        graphs_dir: The directory containing the graph files.
        db_id: The database ID to load the graph for.

    Returns:
        A NetworkX graph representing the database schema.
    """
    graph_file = Path(graphs_dir) / f"{db_id}_graph.pkl"
    with open(graph_file, "rb") as f:
        logging.info(f"Loaded graph for DB: {db_id}")
        return pickle.load(f)


def save_subgraph_image(
    main_graph: nx.Graph,
    subgraph: nx.Graph,
    db_id: str,
    pattern_id: int,
    table_names: List[str],
    edge_labels: Dict,
    output_dir: str,
) -> None:
    """
    Saves the image of a subgraph extracted from the main graph.

    Args:
        main_graph: The main graph representing the full schema.
        subgraph: The subgraph to be visualized and saved.
        db_id: The database ID used for folder structure.
        pattern_id: The unique pattern identifier.
        table_names: List of table names involved in the subgraph.
        edge_labels: Labels for the edges in the subgraph.
        output_dir: The directory where the image will be saved.
    """
    pos = nx.spring_layout(main_graph)
    plt.figure(figsize=(12, 9))

    # Draw the main graph and the subgraph
    nx.draw(
        main_graph,
        pos,
        with_labels=True,
        node_color="lightgray",
        edge_color="lightgray",
        node_size=2000,
        font_size=10,
    )
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=subgraph.nodes, node_color="lightblue", node_size=2500
    )
    nx.draw_networkx_edges(
        main_graph, pos, edgelist=subgraph.edges, edge_color="blue", width=2
    )
    nx.draw_networkx_edge_labels(
        main_graph, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    plt.title(f"Pattern {pattern_id} (Tables: {len(table_names)}) in DB {db_id}")
    image_dir = Path(output_dir) / db_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(image_dir / f"pattern_{pattern_id}.png")
    plt.close()
    logging.info(f"Saved subgraph image for pattern {pattern_id} in DB {db_id}")


def process_json_file(
    file_path: str,
    graphs_dir: str,
    output_dir: str,
    pattern_count_by_table: Dict[str, Dict[int, int]],
    query_count_by_db: Dict[str, int],
) -> None:
    """
    Processes a JSON file containing pattern information and generates subgraph images.

    Args:
        file_path: Path to the JSON file with pattern data.
        graphs_dir: Directory containing graph pickles for the databases.
        output_dir: Base output directory to save images and pickle files.
        pattern_count_by_table: Dictionary tracking the number of patterns by table count for each database.
        query_count_by_db: Dictionary tracking the number of queries processed for each database.
    """
    with open(file_path, "r") as f:
        patterns = json.load(f)

    for pattern in patterns:
        db_id = pattern["db_id"]
        query_count_by_db[db_id] = query_count_by_db.get(db_id, 0) + 1
        main_graph = load_main_graph(graphs_dir, db_id)

        if "Tables" in pattern and len(pattern["Tables"]) == 1:
            # if db_id != "dw":
            #     table_name = pattern["Tables"][0]["table_name"].lower()
            # elif db_id == "dw":
            #     table_name = pattern["Tables"][0]["table_name"].upper()
            table_name = pattern["Tables"][0]["table_name"]
            subgraph = nx.Graph()
            subgraph.add_node(table_name)
            num_tables = 1
            pattern_count_by_table.setdefault(db_id, {}).setdefault(num_tables, 0)

            db_id_output_dir = Path(output_dir) / db_id
            images_dir = db_id_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            pattern_id = len(list(images_dir.glob("pattern_*.png"))) + 1
            pattern_signature = frozenset([table_name.lower()])
            bucket_dir = db_id_output_dir / f"join_patterns_{num_tables}_tables.pkl"

            # Check if pattern already exists
            if bucket_dir.exists():
                with open(bucket_dir, "rb") as f:
                    all_patterns = pickle.load(f)
            else:
                all_patterns = []

            # Save new pattern and subgraph if not already saved
            seen_pattern_signatures = [pat["pattern_signature"] for pat in all_patterns]
            if pattern_signature in seen_pattern_signatures:
                pattern_index = seen_pattern_signatures.index(pattern_signature)
                all_patterns[pattern_index]["equivalent_queries"].append(pattern)

            else:
                subgraph_data = {
                    "pattern_signature": pattern_signature,
                    "subgraph": subgraph,
                    "equivalent_queries": [pattern],
                }
                all_patterns.append(subgraph_data)

                edge_labels = {}
                save_subgraph_image(
                    main_graph,
                    subgraph,
                    db_id,
                    pattern_id,
                    [table_name],
                    edge_labels,
                    output_dir,
                )
                pattern_count_by_table[db_id][num_tables] += 1
            with open(bucket_dir, "wb") as f:
                pickle.dump(all_patterns, f)
        else:
            # Handle join relations
            join_relations = pattern["Join_relations_without_aliases"]
            table_names = set()
            columns_in_relations = set()
            edge_labels = {}
            edges_to_include = set()

            for relation in join_relations:
                # if db_id != "dw":
                #     relation = relation.lower()
                left_col, right_col = relation.split("=")
                left_col, right_col = left_col.strip(), right_col.strip()
                table_names.update([x.split(".")[0] for x in [left_col, right_col]])
                columns_in_relations.add(
                    frozenset([left_col.lower(), right_col.lower()])
                )
                left_node, right_node = left_col.split(".")[0], right_col.split(".")[0]
                matched_left_node = next(
                    (
                        node
                        for node in main_graph.nodes
                        if node.lower() == left_node.lower()
                    ),
                    left_node,
                )
                matched_right_node = next(
                    (
                        node
                        for node in main_graph.nodes
                        if node.lower() == right_node.lower()
                    ),
                    right_node,
                )

                edge = (matched_left_node, matched_right_node)

                if edge not in edge_labels:
                    edge_labels[edge] = [relation]
                else:
                    if relation not in edge_labels[edge]:
                        edge_labels[edge].append(relation)
                edges_to_include.add(edge)

                if main_graph.has_edge(*edge):
                    main_graph[edge[0]][edge[1]]["label"] = "; ".join(edge_labels[edge])

            subgraph_nodes = list(table_names)
            subgraph = nx.Graph()
            subgraph.add_nodes_from(subgraph_nodes)
            for edge in edges_to_include:
                if main_graph.has_edge(*edge):
                    subgraph.add_edge(*edge, **main_graph.get_edge_data(*edge))

            num_tables = len(subgraph_nodes)
            pattern_count_by_table.setdefault(db_id, {}).setdefault(num_tables, 0)

            if num_tables == 0:
                print("No tables found in the subgraph")
                print(pattern)
                break

            db_id_output_dir = Path(output_dir) / db_id
            images_dir = db_id_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            pattern_id = len(list(images_dir.glob("pattern_*.png"))) + 1
            pattern_signature = (
                frozenset([node.lower() for node in subgraph_nodes]),
                frozenset(columns_in_relations),
            )
            bucket_dir = db_id_output_dir / f"join_patterns_{num_tables}_tables.pkl"

            # Check if pattern already exists
            if bucket_dir.exists():
                with open(bucket_dir, "rb") as f:
                    all_patterns = pickle.load(f)
            else:
                all_patterns = []

            # Save new pattern and subgraph if not already saved
            seen_pattern_signatures = [pat["pattern_signature"] for pat in all_patterns]
            if pattern_signature in seen_pattern_signatures:
                pattern_index = seen_pattern_signatures.index(pattern_signature)
                all_patterns[pattern_index]["equivalent_queries"].append(pattern)
            else:
                subgraph_data = {
                    "pattern_signature": pattern_signature,
                    "subgraph": subgraph,
                    "equivalent_queries": [pattern],
                }
                all_patterns.append(subgraph_data)

                save_subgraph_image(
                    main_graph,
                    subgraph,
                    db_id,
                    pattern_id,
                    table_names,
                    edge_labels,
                    output_dir,
                )
                pattern_count_by_table[db_id][num_tables] += 1
            with open(bucket_dir, "wb") as f:
                pickle.dump(all_patterns, f)


def run(patterns_dir: str, graphs_dir: str, output_dir: str) -> None:
    """
    Processes all JSON files in the patterns directory, generates subgraph images, and saves them.

    Args:
        patterns_dir: The directory containing the JSON files with pattern data.
        graphs_dir: The directory containing the graph pickle files.
        output_dir: The directory where the output files and images will be saved.
    """
    pattern_count_by_table: Dict[str, Dict[int, int]] = {}
    query_count_by_db: Dict[str, int] = {}

    # Process each pattern file
    for file_name in os.listdir(patterns_dir):
        if file_name.endswith(".json"):
            file_path = Path(patterns_dir) / file_name
            process_json_file(
                file_path,
                graphs_dir,
                output_dir,
                pattern_count_by_table,
                query_count_by_db,
            )

    # Print summary
    for db_id, table_counts in pattern_count_by_table.items():
        logging.info(f"DB {db_id}:")
        for num_tables, count in table_counts.items():
            logging.info(f"  Unique patterns with {num_tables} tables: {count}")
        logging.info(f"  Total queries processed: {query_count_by_db[db_id]}")


def main():
    """
    Main entry point of the script.
    Sets up the necessary directories and starts processing the data.
    """
    # Define paths to the required directories
    patterns_dir = (
        "bird_parsed_queries"  # Replace with the actual path to the patterns directory
    )
    graphs_dir = (
        "bird_graphs/pickles/"  # Replace with the actual path to the graphs directory
    )
    output_dir = "bird_essay/"  # Replace with the actual path to the output directory

    # Ensure the directories exist
    create_directories(output_dir)

    # Start processing the patterns
    run(patterns_dir, graphs_dir, output_dir)


if __name__ == "__main__":
    main()

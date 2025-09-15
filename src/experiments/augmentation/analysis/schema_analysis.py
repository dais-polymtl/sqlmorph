import os
import pickle
import csv
import networkx as nx
import logging
from typing import List, Dict, Union
import pandas as pd

# Set up logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# def analyze_graph(
#     graph: nx.Graph, db_id: str
# ) -> List[Dict[str, Any]]:
#     components = list(nx.connected_components(graph))
#     results = []
#     component_sizes = [len(component) for component in components]
#     component_id = 1

#     for component in components:
#         if len(component) > 1:
#             subgraph = graph.subgraph(component).copy()
#             num_nodes = subgraph.number_of_nodes()
#             num_edges = subgraph.number_of_edges()
#             avg_clustering = (
#                 round(nx.average_clustering(subgraph), 3) if num_nodes > 1 else 0.0
#             )
#             diameter = (
#                 nx.diameter(subgraph) if nx.is_connected(subgraph) else float("inf")
#             )
#             cycles = nx.cycle_basis(subgraph)
#             is_cyclic = len(cycles) > 0
#             num_cycles = len(cycles)

#             # Find longest shortest path (diameter nodes)
#             diameter_nodes = ""
#             for node1 in subgraph.nodes:
#                 for node2 in subgraph.nodes:
#                     if node1 != node2 and nx.has_path(subgraph, node1, node2):
#                         path_length = nx.shortest_path_length(subgraph, node1, node2)
#                         if path_length == diameter:
#                             diameter_nodes = " -- ".join(
#                                 nx.shortest_path(subgraph, node1, node2)
#                             )

#             results.append(
#                 {
#                     "database_name": db_id,
#                     "component_id": f"{db_id}_comp_{component_id}",
#                     "num_tables": len(graph.nodes),
#                     "num_connected_components": len(components),
#                     "connected_component_sizes": component_sizes,
#                     "is_cyclic": bool(nx.cycle_basis(graph)),
#                     "is_component_cyclic": is_cyclic,
#                     "component_num_nodes": num_nodes,
#                     "component_num_edges": num_edges,
#                     "component_avg_clustering": avg_clustering,
#                     "component_diameter": (
#                         round(diameter, 3) if diameter != float("inf") else "inf"
#                     ),
#                     "component_diameter_nodes": diameter_nodes,
#                     "component_num_cycles": num_cycles,     # <--- Added here
#                 }
#             )
#             component_id += 1

#     return results


def canonical_cycle(cycle):
    """Normalize cycle to avoid counting duplicates due to rotations/directions."""
    min_idx = cycle.index(min(cycle))
    rotated = cycle[min_idx:] + cycle[:min_idx]
    rev_rotated = rotated[::-1]
    return tuple(min(rotated, rev_rotated))


def analyze_graph(G, db_id):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    avg_degree = (sum(dict(G.degree()).values()) / num_nodes) if num_nodes > 0 else 0
    is_connected = nx.is_connected(G) if num_nodes > 0 else False
    # raw_cycles = list(nx.simple_cycles(G))
    # raw cycles takes too long to execute, we need a faster way to determine cyclicity

    is_cyclic = len(list(nx.cycle_basis(G))) > 0 if num_nodes > 0 else False

    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)

    # Cycle detection with deduplication
    # raw_cycles = list(nx.simple_cycles(G))

    # deduped_cycles = set()
    # for c in raw_cycles:
    #     if len(c) < 2:  # skip trivial loops
    #         continue
    #     deduped_cycles.add(canonical_cycle(c))

    # num_cycles = len(deduped_cycles)

    return {
        "id": db_id,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": avg_degree,
        "is_connected": is_connected,
        "is_cyclic": is_cyclic,
        "diameter": diameter,
    }


# def process_pickle_files(
#     folder_path: str,
# ) -> List[Dict[str, Union[str, int, bool, float, str]]]:
#     """
#     Processes pickle files from the specified folder path.

#     Args:
#         folder_path (str): The path to the folder containing pickle files.

#     Returns:
#         List[Dict[str, Union[str, int, bool, float]]]: A list of dictionaries representing the processed data.
#     """
#     results = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith("_graph.pkl"):
#             db_id = filename.replace("_graph.pkl", "")
#             try:
#                 with open(os.path.join(folder_path, filename), "rb") as file:
#                     graph = pickle.load(file)
#                     results.extend(analyze_graph(graph, db_id))
#             except Exception as e:
#                 logging.error(f"Failed to process {filename}: {e}")

#     return results


def process_pickle_files(
    folder_path: str,
) -> List[Dict[str, Union[str, int, bool, float]]]:
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith("_graph.pkl"):
            db_id = filename.replace("_graph.pkl", "")
            try:
                with open(os.path.join(folder_path, filename), "rb") as file:
                    graph = pickle.load(file)
                    results.append(analyze_graph(graph, db_id))
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


def summarize_benchmark(df: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    summary = {
        "Benchmark": benchmark_name,
        "# Schemas": len(df),
        "% Connected": round(100 * df["is_connected"].mean(), 2),
        "% Cyclic": round(100 * df["is_cyclic"].mean(), 2),
        "Avg Degree": round(df["average_degree"].mean(), 3),
        "Avg Diameter": round(df["diameter"].mean(), 3),
        "Degree Std": round(df["average_degree"].std(), 3),
        "Min Degree": round(df["average_degree"].min(), 3),
        "Max Degree": round(df["average_degree"].max(), 3),
    }

    return pd.DataFrame([summary])


# def load_and_process(files: dict) -> pd.DataFrame:
#     summaries = []
#     for benchmark, filepath in files.items():
#         df = pd.read_csv(filepath)
#         df["is_connected"] = df["is_connected"].astype(bool)
#         df["is_cyclic"] = df["is_cyclic"].astype(bool)
#         summaries.append(summarize_benchmark(df, benchmark))
#     return pd.concat(summaries, ignore_index=True)


def load_and_process(files: dict) -> pd.DataFrame:
    """
    files: dict mapping benchmark name -> filepath
    Returns combined summary dataframe
    """
    summaries = []
    for benchmark, filepath in files.items():
        df = pd.read_csv(filepath)
        # Ensure boolean columns are booleans (sometimes read as objects)
        df["is_connected"] = df["is_connected"].astype(bool)
        df["is_cyclic"] = df["is_cyclic"].astype(bool)
        summaries.append(summarize_benchmark(df, benchmark))
    return pd.concat(summaries, ignore_index=True)


def main():
    """
    Main function to process graphs and save the results to a CSV file.
    """
    folder_path = "data/graph_data/beaver_graphs/pickles"
    output_file = "data/analysis/beaver_schema_analysis.csv"

    logging.info(f"Processing graphs from {folder_path}")
    process_graphs_and_save_csv(folder_path=folder_path, output_file=output_file)

    input_files = {
        "Spider": "data/analysis/spider_schema_analysis.csv",
        "Bird": "data/analysis/bird_schema_analysis.csv",
        "Beaver": "data/analysis/beaver_schema_analysis.csv",
    }
    summary_df = load_and_process(input_files)
    summary_df.to_csv("data/analysis/benchmark_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()

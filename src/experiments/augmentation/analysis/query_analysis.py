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
                subgraphs = [d["jq_graph"] for d in data]
                if db_id not in db_subgraphs:
                    db_subgraphs[db_id] = []
                db_subgraphs[db_id].extend(subgraphs)

    return db_subgraphs


# def analyze_subgraphs(subgraphs: List[nx.Graph]) -> List[Dict[str, Any]]:
#     subgraphs_sorted = sorted(subgraphs, key=lambda g: g.number_of_nodes())

#     analysis_results = []
#     for i, graph in enumerate(subgraphs_sorted):
#         num_nodes = graph.number_of_nodes()
#         num_edges = graph.number_of_edges()
#         is_connected = nx.is_connected(graph)

#         # Average degree
#         avg_degree = sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0

#         # Number of cycles using cycle_basis (for undirected graphs)
#         num_cycles = len(nx.cycle_basis(graph))

#         # Check cyclicity (alternative to cycle_basis for boolean)
#         is_cyclic = num_cycles > 0

#         analysis_results.append(
#             {
#                 "id": f"g{i + 1}",
#                 "num_nodes": num_nodes,
#                 "num_edges": num_edges,
#                 "average_degree": avg_degree,
#                 "num_cycles": num_cycles,
#                 "is_connected": is_connected,
#                 "is_cyclic": is_cyclic,
#             }
#         )

#     return analysis_results


def canonical_cycle(cycle):
    """Normalize cycle to avoid counting duplicates due to rotations/directions."""
    min_idx = cycle.index(min(cycle))
    rotated = cycle[min_idx:] + cycle[:min_idx]
    rev_rotated = rotated[::-1]
    return tuple(min(rotated, rev_rotated))


def analyze_subgraphs(subgraphs: List[nx.Graph]) -> List[Dict[str, Any]]:
    subgraphs_sorted = sorted(subgraphs, key=lambda g: g.number_of_nodes())

    analysis_results = []
    for i, graph in enumerate(subgraphs_sorted):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        is_connected = nx.is_connected(graph)

        # Average degree
        avg_degree = (
            sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
        )

        # Get simple cycles and deduplicate
        raw_cycles = list(nx.simple_cycles(graph))

        deduped_cycles = set()
        for c in raw_cycles:
            if len(c) < 2:  # skip trivial loops
                continue
            deduped_cycles.add(canonical_cycle(c))

        num_cycles = len(deduped_cycles)
        is_cyclic = num_cycles > 0

        analysis_results.append(
            {
                "id": f"g{i + 1}",
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "average_degree": avg_degree,
                "num_cycles": num_cycles,
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


def group_by_num_nodes(output_file: str) -> None:
    # Sort rows by num_nodes
    # let's read the output file insead and get rows from there:
    df = pd.read_csv(output_file)
    rows = df.to_dict(orient="records")
    rows.sort(key=lambda x: x["num_nodes"])
    grouped = {}
    total_queries = len(rows)

    for row in rows:
        num_nodes = row["num_nodes"]
        if num_nodes not in grouped:
            grouped[num_nodes] = []
        grouped[num_nodes].append(row)

    for num_nodes, group in grouped.items():
        count = len(group)
        avg_degree = sum(r["average_degree"] for r in group) / count
        # print("num_nodes: ", num_nodes)
        # for r in group:
        # print(r["is_cyclic"])
        cyclic_count = sum(1 for r in group if r["is_cyclic"])

        percentage_cyclic = (cyclic_count / count) * 100 if count else 0
        percentage_of_total = (count / total_queries) * 100 if total_queries else 0

        print(
            f"Num Nodes: {num_nodes}, "
            f"Count: {count} ({percentage_of_total:.2f}%), "
            f"Avg Degree: {avg_degree:.2f}, "
            f"Cyclic: {percentage_cyclic:.2f}%"
        )


def main() -> None:
    """
    Main function to read, process, and save the data.
    """
    base_folder = "data/rule_inputs/lt_elimination/"
    output_file = "data/analysis/bird_jqg_graph_analysis.csv"
    db_subgraphs = read_and_process_pickle_files(base_folder=base_folder)
    # analysis_results = analyze_subgraphs(subgraphs=db_subgraphs)
    save_to_csv(db_subgraphs=db_subgraphs, output_file=output_file)
    print(f"Results saved to {output_file}")
    group_by_num_nodes(output_file=output_file)


if __name__ == "__main__":
    main()

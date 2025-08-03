import os
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

from src.core.logger.logger import Logger

logger = Logger(__name__)


def _set_plot_style(font_path: Path) -> None:
    """Sets the global matplotlib and seaborn style using a custom font."""
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["figure.dpi"] = 120
    sns.set_theme(style="whitegrid", font_scale=1.2)


def _read_csv_data(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_avg_degree(data_folder: Path) -> None:
    """Plots and saves average degree comparison between original and augmented sets."""
    augmented_path = (
        data_folder
        / "experiments/augmentation/experiment_inputs/augmented_join_details.csv"
    )
    original_path = (
        data_folder
        / "experiments/augmentation/experiment_inputs/original_join_details.csv"
    )

    augmented = _read_csv_data(augmented_path)
    original = _read_csv_data(original_path)
    pruned = [r for r in augmented if r["set"] == "pruned"]
    generated = [r for r in augmented if r["set"] == "generated"]

    sets = ["Original", "Augmented", "Augmented Pruned", "Augmented Generated"]
    values = [
        np.mean([float(r["avg_degree"]) for r in original]),
        np.mean([float(r["avg_degree"]) for r in augmented]),
        np.mean([float(r["avg_degree"]) for r in pruned]),
        np.mean([float(r["avg_degree"]) for r in generated]),
    ]
    colors = ["#5B8FA8", "#F4A259", "#7FB685", "#E4572E"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sets, values, color=colors, edgecolor="black")
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.title("Average Degree of Join Query Graphs", weight="bold")
    plt.xlabel("Set")
    plt.ylabel("Average Degree")
    plt.ylim(0, max(values) + 0.5)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = (
        data_folder
        / "experiments/augmentation/experiment_inputs/avg_degree_join_query_graphs.png"
    )
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.log("info", f"Average degree plot saved to {out}")


def plot_delta_avg_degree(data_folder: Path) -> None:
    """Plots and saves histogram of the delta in average degree (augmented - original),
    and saves a CSV summary of delta values grouped by query structure transitions."""

    def _read_csv_data(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    # Read input CSVs
    augmented = _read_csv_data(
        data_folder
        / "experiments/augmentation/experiment_inputs/augmented_join_details.csv"
    )
    original = _read_csv_data(
        data_folder
        / "experiments/augmentation/experiment_inputs/original_join_details.csv"
    )

    deltas = []
    delta_groups = defaultdict(
        lambda: {"count": 0, "structural_transitions": Counter()}
    )

    for row in augmented:
        orig_id = int(row["id"][2:6])
        orig = next((r for r in original if int(r["id"]) == orig_id), None)
        if orig:
            delta = round(float(row["avg_degree"]) - float(orig["avg_degree"]), 2)
            deltas.append(delta)

            aug_struct = (int(row["num_nodes"]), int(row["num_edges"]))
            orig_struct = (int(orig["num_nodes"]), int(orig["num_edges"]))
            transition = (aug_struct, orig_struct)

            delta_groups[delta]["count"] += 1
            delta_groups[delta]["structural_transitions"][transition] += 1

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(
        deltas, bins=20, kde=True, color="#4C72B0", edgecolor="white", linewidth=1.2
    )
    plt.title("Δ Avg Degree Distribution per Query", weight="bold")
    plt.xlabel("Δ Avg Degree (Augmented - Original)")
    plt.ylabel("Number of Queries")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plot_path = (
        data_folder
        / "experiments/augmentation/experiment_inputs/delta_avg_degree_join_query_graphs.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.log("info", f"Delta avg degree plot saved to {plot_path}")

    # Save CSV summary
    csv_path = (
        data_folder
        / "experiments/augmentation/experiment_inputs/delta_avg_degree_structural_summary.csv"
    )
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "delta_avg_degree",
            "count",
            "augmented_structure",
            "original_structure",
            "transition_count",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for delta, group in sorted(
            delta_groups.items(), key=lambda x: x[1]["count"], reverse=True
        ):
            sorted_transitions = sorted(
                group["structural_transitions"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for (aug_struct, orig_struct), trans_count in sorted_transitions:
                writer.writerow(
                    {
                        "delta_avg_degree": delta,
                        "count": group["count"],
                        "augmented_structure": str(aug_struct),
                        "original_structure": str(orig_struct),
                        "transition_count": trans_count,
                    }
                )

    logger.log("info", f"Delta avg degree structure summary saved to {csv_path}")


def is_cyclic(graph: nx.Graph) -> bool:
    try:
        nx.find_cycle(graph)
        return True  # A cycle exists
    except nx.exception.NetworkXNoCycle:
        return False  # Acyclic


def canonical_cycle(cycle):
    """Normalize cycle to avoid counting duplicates due to rotations/directions."""
    min_idx = cycle.index(min(cycle))
    rotated = cycle[min_idx:] + cycle[:min_idx]
    rev_rotated = rotated[::-1]
    return tuple(min(rotated, rev_rotated))


def schema_graphs_cyclicity(data_folder: Path) -> None:
    db_ids = [
        "california_schools",
        "card_games",
        "codebase_community",
        "debit_card_specializing",
        "european_football_2",
        "financial",
        "formula_1",
        "student_club",
        "superhero",
        "thrombosis_prediction",
        "toxicology",
    ]
    db_rows = []

    for db_id in db_ids:
        schema_graph_path = (
            data_folder
            / "graph_data"
            / "bird_graphs"
            / "pickles"
            / f"{db_id}_graph.pkl"
        )

        with open(schema_graph_path, "rb") as f:
            G: nx.Graph = pickle.load(f)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Convert to directed and detect all simple cycles
        # DG = G.to_directed()
        raw_cycles = list(nx.simple_cycles(G))
        if db_id == "card_games":
            for i, raw_cycle in enumerate(raw_cycles):
                print(f"Card Games Cycle {i}: {raw_cycle}")

        # Deduplicate cycles
        deduped_cycles = set()
        for c in raw_cycles:
            if len(c) < 2:
                continue  # skip trivial
            deduped_cycles.add(canonical_cycle(c))

        is_cyclic = len(deduped_cycles) > 0
        total_cycles = len(deduped_cycles)

        # Group by cycle length
        cycle_len_counts = defaultdict(int)
        for cycle in deduped_cycles:
            cycle_len_counts[len(cycle)] += 1

        # Format: 3 nodes (2), 4 nodes (1)
        cycle_len_str = ", ".join(
            f"{length} nodes ({count})"
            for length, count in sorted(cycle_len_counts.items())
        )

        avg_degree = round((2 * num_edges) / num_nodes, 2) if num_nodes else 0.0
        # Compose row
        db_rows.append(
            {
                "db_id": db_id,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "is_cyclic": is_cyclic,
                "avg_degree": avg_degree,
                "num_cycles": total_cycles,
                "num_cycles_by_node": cycle_len_str,
            }
        )

    # sort by number of cycles
    db_rows.sort(key=lambda x: x["num_cycles"], reverse=True)
    # Write CSV
    output_path = data_folder / "schema_cyclicity_final.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "db_id",
                "num_nodes",
                "num_edges",
                "is_cyclic",
                "avg_degree",
                "num_cycles",
                "num_cycles_by_node",
            ],
        )
        writer.writeheader()
        writer.writerows(db_rows)

    print(f"✅ CSV saved at: {output_path}")


def main() -> None:
    data_folder = Path(os.getenv("DATA_FOLDER"))
    font_path = data_folder / "fonts/Inconsolata-VariableFont_wdth,wght.ttf"
    _set_plot_style(font_path)
    # plot_avg_degree(data_folder)
    # plot_delta_avg_degree(data_folder)
    # export_cyclicity_distribution_csv(data_folder)
    schema_graphs_cyclicity(data_folder)
    # plot_accuracy_by_joins()


if __name__ == "__main__":
    main()

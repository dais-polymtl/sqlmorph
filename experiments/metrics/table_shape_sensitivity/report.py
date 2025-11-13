import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def parse_metrics(metrics_str):
    """Parse metrics from JSON string"""
    if pd.isna(metrics_str) or metrics_str in ["timeout", "error"]:
        return None
    try:
        return json.loads(metrics_str)
    except (json.JSONDecodeError, ValueError):
        return None


def create_evaluation_plots(
    csv_path: str,
    json_path: str,
    techniques: list,
    output_dir: str = None,
    plot_name: str = None,
):
    """Create evaluation plots for all techniques and groups"""

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Get groups in the order they appear in the original JSON file
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            mutants_data = json.load(f)
        groups = list(mutants_data["mutant_queries"].keys())
    else:
        # Fallback to the order they appear in CSV (first occurrence)
        groups = df["group_name"].drop_duplicates().tolist()

    n_groups = len(groups)
    n_techniques = len(techniques)

    # Create figure with subplots - dynamically adjust rows based on number of techniques
    fig, axes = plt.subplots(
        n_techniques, n_groups, figsize=(4 * n_groups, 5 * n_techniques)
    )

    # Handle case where there's only one group or one technique
    if n_techniques == 1 and n_groups == 1:
        axes = np.array([[axes]])
    elif n_techniques == 1:
        axes = axes.reshape(1, -1)
    elif n_groups == 1:
        axes = axes.reshape(-1, 1)

    # Color mapping for metrics
    colors = {"EX": "gray", "EXP": "blue", "EXR": "red", "F1": "green"}

    # Process each technique (row)
    for tech_idx, technique in enumerate(techniques):
        # Process each group (column)
        for group_idx, group in enumerate(groups):
            ax = axes[tech_idx, group_idx]

            # Filter data for current group
            group_data = df[df["group_name"] == group].copy()
            group_data = group_data.sort_values("mutant_index")

            # Extract metrics for current technique
            ex_values = []
            exp_values = []
            exr_values = []
            f1_values = []
            mutant_indices = []

            for _, row in group_data.iterrows():
                metrics = parse_metrics(row[technique])
                if metrics and all(
                    key in metrics for key in ["EX", "EXP", "EXR", "F1"]
                ):
                    ex_values.append(metrics["EX"])
                    exp_values.append(metrics["EXP"])
                    exr_values.append(metrics["EXR"])
                    f1_values.append(metrics["F1"])
                    mutant_indices.append(row["mutant_index"])

            # Plot metrics
            if mutant_indices:
                ax.plot(
                    mutant_indices,
                    ex_values,
                    "d-",
                    color=colors["EX"],
                    label="EX",
                    linewidth=2,
                    markersize=6,
                    alpha=0.5,
                )
                ax.plot(
                    mutant_indices,
                    exp_values,
                    "o-",
                    color=colors["EXP"],
                    label="EXP",
                    linewidth=2,
                    markersize=6,
                )
                ax.plot(
                    mutant_indices,
                    exr_values,
                    "s-",
                    color=colors["EXR"],
                    label="EXR",
                    linewidth=2,
                    markersize=6,
                )
                ax.plot(
                    mutant_indices,
                    f1_values,
                    "^-",
                    color=colors["F1"],
                    label="F1",
                    linewidth=2,
                    markersize=6,
                )

            # # Customize plot
            # if mutant_indices:
            #     ax.set_xlim(mutant_indices[0], mutant_indices[-1])
            #     # Add vertical dotted line at middle index (lighter gray, thicker, smoother)
            #     middle_index = (max(mutant_indices) // 2) + 1
            #     ax.axvline(
            #         x=middle_index,
            #         color="#888888",
            #         linestyle="dotted",
            #         alpha=0.7,
            #         linewidth=1,
            #     )
            # else:
            #     ax.set_xlim(1, 9)
            #     # Add vertical dotted line at middle index for default case (lighter gray, thicker, smoother)
            #     middle_index = (9 // 2) + 1  # which is 5
            #     ax.axvline(
            #         x=middle_index,
            #         color="#888888",
            #         linestyle="dotted",
            #         alpha=0.7,
            #         linewidth=1,
            #     )

            ax.set_ylim(-0.05, 1.05)
            # Remove x-axis label "Mutant Index"
            # Only add y-axis label "Score" for leftmost plots
            if group_idx == 0:
                ax.set_ylabel("Score")
            # Make grid dashed instead of solid
            ax.grid(True, alpha=0.3, linestyle=(0, (4, 4)))

            # Set x-axis ticks to integers
            if mutant_indices:
                ax.set_xticks(range(1, max(mutant_indices) + 1))
            else:
                ax.set_xticks(range(1, 5))

            # Add title for each subplot
            if tech_idx == 0:
                ax.set_title(f"{group}", fontweight="bold")

            # Add y-axis label for leftmost plots
            if group_idx == 0:
                ax.text(
                    -0.23,
                    0.5,
                    technique.replace("_", " "),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                )

    # Add legend at top center of the entire figure with custom order
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Reorder to put EX first, then EXP, EXR, F1
    metric_order = ["EX", "EXP", "EXR", "F1"]
    ordered_handles = []
    ordered_labels = []

    for metric in metric_order:
        if metric in labels:
            idx = labels.index(metric)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.95),
    )

    # Adjust layout to make room for legend at top
    plt.tight_layout()
    # Reduce top margin to bring legend closer to plots
    top_margin = 0.81 if n_techniques == 1 else 0.90
    plt.subplots_adjust(top=top_margin)

    # Save the plot as PNG
    if output_dir:
        filename = plot_name if plot_name else f"table-shape_{Path(csv_path).stem}"
        output_path = Path(output_dir) / f"{filename}.png"
        output_path_pdf = Path(output_dir) / f"{filename}.pdf"
    else:
        filename = plot_name if plot_name else f"table-shape_{Path(csv_path).stem}"
        output_path = Path(csv_path).parent / f"{filename}.png"
        output_path_pdf = Path(csv_path).parent / f"{filename}.pdf"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    print(f"Plot saved to: {output_path_pdf}")

    # Show the plot
    # plt.show()


def print_summary_statistics(csv_path: str, techniques: list):
    """Print summary statistics for the evaluation results"""
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for technique in techniques:
        print(f"\n{technique}:")
        print("-" * 60)

        all_exp = []
        all_exr = []
        all_f1 = []
        timeout_count = 0
        error_count = 0
        valid_count = 0

        for _, row in df.iterrows():
            metrics = parse_metrics(row[technique])
            if row[technique] == "timeout":
                timeout_count += 1
            elif row[technique] == "error":
                error_count += 1
            elif metrics and all(key in metrics for key in ["EXP", "EXR", "F1"]):
                all_exp.append(metrics["EXP"])
                all_exr.append(metrics["EXR"])
                all_f1.append(metrics["F1"])
                valid_count += 1

        print(f"Valid evaluations: {valid_count}")
        print(f"Timeouts: {timeout_count}")
        print(f"Errors: {error_count}")

        if all_exp:
            print(f"EXP - Mean: {np.mean(all_exp):.4f}, Std: {np.std(all_exp):.4f}")
            print(f"EXR - Mean: {np.mean(all_exr):.4f}, Std: {np.std(all_exr):.4f}")
            print(f"F1  - Mean: {np.mean(all_f1):.4f}, Std: {np.std(all_f1):.4f}")


if __name__ == "__main__":
    # CONFIGURABLE PARAMETERS
    DATA = "data/metrics/experiments/table_shape_sensitivity"

    experiment_name = "mutantsV1_eval_with_penalty-2025-09-27"
    suffix = "_semantic-columns-partial-cell"
    plot_name = f"{experiment_name}{suffix}"
    csv_path = f"{DATA}/{experiment_name}.csv"
    json_path = f"{DATA}/mutants_v1.json"
    output_dir = f"{DATA}/plots_{experiment_name}/"

    techniques = [
        # "EXACT_COLUMN_AND_EXACT_CELL",
        # "EXACT_COLUMN_AND_PARTIAL_CELL",
        # "SEMANTIC_COLUMN_AND_EXACT_CELL",
        "SEMANTIC_COLUMN_AND_PARTIAL_CELL",
        # "NO_COLUMN_AND_PARTIAL_CELL",
        # "UNIFIED_COLUMN_AND_SEMANTIC_ROW",
    ]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"CSV file: {csv_path}")
    print(f"JSON file: {json_path}")
    print(f"Output directory: {output_dir}")

    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("Please run the evaluation script first to generate the CSV file")
    else:
        print("Creating evaluation plots...")
        create_evaluation_plots(csv_path, json_path, techniques, output_dir, plot_name)
        print_summary_statistics(csv_path, techniques)
        print("Report generation complete!")

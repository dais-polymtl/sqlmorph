# report.py
"""
Experiment 1 – Sensitivity to Controlled Error Counts (Multi-Technique Analysis)
─────────────────────────────────────────────────────────────────────────────────
This script reads all mutant_scores_*.json files (one per evaluation technique) and produces:

1. **Figure 1** – mean score vs. operator types for each error pattern and technique:
   • One row per error pattern
   • Three plots in each row, one per technique
   • X-axis shows operator types instead of simple integers
   • Distinct colours / linestyles, slight transparency
   • If a technique file doesn't exist, shows empty plot with "No Data" message

2. **Figure 2** - aggregate view of mean scores across all error patterns:
   • Single row with three plots (one per technique)
   • X-axis shows simple labels (e1, e2, e3...)
   • Aggregated data from all error patterns

3. **Table 1** – Spearman ρ and Kendall τ correlations between
   error_count and each metric for each technique and error pattern.
   • Rows with NaNs in a metric are ignored (nan_policy="omit")
   • If fewer than two valid points remain (or all values equal),
     the correlation is reported as NaN

Artefacts are saved in the plots/ directory next to the scores files.
"""

from pathlib import Path
import json
import glob
import warnings
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# ────────────────────────────────────────────────────────────────────────
# 0.  File paths – adjust to your environment
# ────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")

SCORES_DIR = ROOT / "data/evaluation/experiments/controlled_error_sensitivity"
PLOTS_DIR = SCORES_DIR / "plots"
FIGURE_PATH = PLOTS_DIR / "figure1_error_patterns_multi_technique.png"
AGGREGATE_FIGURE_PATH = PLOTS_DIR / "figure2_aggregate_multi_technique.png"
TABLE_PATH = SCORES_DIR / "table1_correlations_by_pattern_technique.csv"

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────
# 1.  Find all mutant score files and extract technique names
# ────────────────────────────────────────────────────────────────────────
def find_mutant_score_files():
    """Find all mutant_scores_*.json files and extract technique names."""
    pattern = str(SCORES_DIR / "mutant_scores_*.json")
    files = glob.glob(pattern)

    techniques = []
    for file_path in files:
        filename = Path(file_path).name
        # Extract technique name from filename: mutant_scores_{technique}.json
        technique_name = filename.replace("mutant_scores_", "").replace(".json", "")
        techniques.append(
            {
                "name": technique_name,
                "path": Path(file_path),
                "display_name": technique_name.replace("_", " ").title(),
            }
        )

    return sorted(techniques, key=lambda x: x["name"])


# ────────────────────────────────────────────────────────────────────────
# 2.  Load and process data for a single technique
# ────────────────────────────────────────────────────────────────────────
def load_technique_data(file_path):
    """Load and process mutant scores for a single technique."""
    try:
        with file_path.open() as f:
            records = json.load(f)

        df = pd.DataFrame(records)
        required_cols = {"error_count", "EX", "EXP", "EXR", "F1"}

        # Check if error_pattern exists - it should in our modified implementation
        if "error_pattern" not in df.columns:
            print(f"[!] Warning: {file_path.name} missing 'error_pattern' column")
            # Create a default pattern for backward compatibility
            df["error_pattern"] = "unspecified_pattern"

        # Check for other missing columns
        missing = required_cols.difference(df.columns)
        if missing:
            print(f"[!] Warning: {file_path.name} missing columns: {missing}")
            return None

        # Extract operators from each record
        if "operators" in df.columns:
            df["operator_type"] = df["operators"].apply(
                lambda ops: ", ".join(ops) if isinstance(ops, list) else "unknown"
            )
        else:
            df["operator_type"] = "unknown"

        return df
    except Exception as e:
        print(f"[!] Error loading {file_path.name}: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────
# 3.  Plot mean scores for a single technique and error pattern
# ────────────────────────────────────────────────────────────────────────
def plot_technique_scores(ax, df, technique_name, error_pattern):
    """Plot mean scores vs error counts for a specific technique and error pattern."""
    if df is None or df.empty:
        ax.text(
            0.5,
            0.5,
            "No Data Available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.6,
        )
        ax.set_title(f"{technique_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Operator Types")
        ax.set_ylabel("Mean Score")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        return

    # Filter data for this error pattern
    pattern_data = df[df["error_pattern"] == error_pattern]

    if pattern_data.empty:
        ax.text(
            0.5,
            0.5,
            f"No Data for\n'{error_pattern}'",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.6,
        )
        ax.set_title(f"{technique_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Operator Types")
        ax.set_ylabel("Mean Score")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        return

    # Group by error count and calculate means for each metric
    mean_tbl = (
        pattern_data.groupby("error_count")[["EX", "EXP", "EXR", "F1"]]
        .mean()
        .reset_index()
    )

    # Get specific operator name for each error count
    operator_labels = {}
    for error_count in pattern_data["error_count"].unique():
        count_data = pattern_data[pattern_data["error_count"] == error_count]
        if not count_data.empty:
            # Get the specific operator for this error count
            first_record = count_data.iloc[0]

            if "operators" in first_record and "pattern_position" in first_record:
                operators = first_record["operators"]
                pattern_position = first_record["pattern_position"]

                # Get the specific operator at this position (pattern_position is 1-indexed)
                if isinstance(operators, list) and 1 <= pattern_position <= len(
                    operators
                ):
                    operator_labels[error_count] = operators[pattern_position - 1]
                else:
                    operator_labels[error_count] = f"op{error_count}"
            else:
                # Fallback: try to extract from error_pattern string
                error_pattern_str = first_record.get("error_pattern", "")
                if " → " in error_pattern_str:
                    operators = error_pattern_str.split(" → ")
                    if error_count <= len(operators):
                        operator_labels[error_count] = operators[error_count - 1]
                    else:
                        operator_labels[error_count] = f"op{error_count}"
                else:
                    operator_labels[error_count] = f"op{error_count}"

    # Sort by error count to ensure lines are properly connected
    mean_tbl = mean_tbl.sort_values("error_count")

    # Prepare x-axis tick positions and labels
    x_ticks = sorted(pattern_data["error_count"].unique())
    x_labels = [operator_labels.get(count, f"e{count}") for count in x_ticks]

    metrics = ["EX", "EXP", "EXR", "F1"]
    linestyles = ["-", "--", ":", "-."]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, m in enumerate(metrics):
        # Get data points for each error count
        x_values = mean_tbl["error_count"]
        y_values = mean_tbl[m]

        # Plot the line
        ax.plot(
            x_values,
            y_values,
            label=m,
            linestyle=linestyles[i],
            color=colours[i],
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=5,
        )

    ax.set_title(f"{technique_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Operator Types")
    ax.set_ylabel("Mean Score")

    # Set x-ticks to error counts
    ax.set_xticks(x_ticks)

    # Set x-tick labels to specific operator names
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_ylim(0, 1.02)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=8)


# ────────────────────────────────────────────────────────────────────────
# 3b. Plot aggregated scores for a single technique across all patterns
# ────────────────────────────────────────────────────────────────────────
def plot_aggregated_technique_scores(ax, df, technique_name):
    """Plot mean scores vs error counts for a specific technique, aggregated across all patterns."""
    if df is None or df.empty:
        ax.text(
            0.5,
            0.5,
            "No Data Available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.6,
        )
        ax.set_title(f"{technique_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Error Count")
        ax.set_ylabel("Mean Score")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        return

    # Group by error count and calculate means for each metric across all patterns
    mean_tbl = (
        df.groupby("error_count")[["EX", "EXP", "EXR", "F1"]].mean().reset_index()
    )

    # Sort by error count to ensure lines are properly connected
    mean_tbl = mean_tbl.sort_values("error_count")

    # Prepare x-axis tick positions and simple labels
    x_ticks = sorted(df["error_count"].unique())
    x_labels = [f"e{count}" for count in x_ticks]

    metrics = ["EX", "EXP", "EXR", "F1"]
    linestyles = ["-", "--", ":", "-."]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, m in enumerate(metrics):
        # Get data points for each error count
        x_values = mean_tbl["error_count"]
        y_values = mean_tbl[m]

        # Plot the line
        ax.plot(
            x_values,
            y_values,
            label=m,
            linestyle=linestyles[i],
            color=colours[i],
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=5,
        )

    ax.set_title(f"{technique_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Error Count")
    ax.set_ylabel("Mean Score")

    # Set x-ticks to error counts
    ax.set_xticks(x_ticks)

    # Set x-tick labels to simple e1, e2, e3... format
    ax.set_xticklabels(x_labels)

    ax.set_ylim(0, 1.02)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=8)


# ────────────────────────────────────────────────────────────────────────
# 4.  Calculate correlations for a single technique with robust error handling
# ────────────────────────────────────────────────────────────────────────
def is_correlation_computable(x, y):
    """
    Check if correlation can be meaningfully computed between x and y.

    Returns False if:
    - Less than 2 valid (non-NaN) pairs
    - Either variable has zero variance (all values the same)
    - All values are NaN
    """
    # Create mask for valid (non-NaN) pairs
    valid_mask = ~(pd.isna(x) | pd.isna(y))

    if valid_mask.sum() < 2:
        return False, "Insufficient valid data points"

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Check for zero variance (all values the same)
    if len(np.unique(x_valid)) < 2:
        return False, "X variable has zero variance"

    if len(np.unique(y_valid)) < 2:
        return False, "Y variable has zero variance"

    return True, "OK"


def safe_correlation(x, y, method="spearman"):
    """
    Safely calculate correlation with proper validation and warning suppression.

    Args:
        x, y: pandas Series or arrays to correlate
        method: 'spearman' or 'kendall'

    Returns:
        correlation coefficient (float or NaN if can't compute)
    """
    can_compute, reason = is_correlation_computable(x, y)

    if not can_compute:
        return np.nan

    try:
        # Suppress scipy warnings for correlation edge cases
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            if method == "spearman":
                corr, _ = spearmanr(x, y, nan_policy="omit")
            elif method == "kendall":
                corr, _ = kendalltau(x, y, nan_policy="omit")
            else:
                return np.nan

        # Additional check: if correlation is NaN or infinite, return NaN
        if not np.isfinite(corr):
            return np.nan

        return corr

    except Exception:
        # If anything goes wrong, return NaN
        return np.nan


def calculate_correlations(df, technique_name):
    """Calculate correlations for each error pattern in a technique."""
    if df is None or df.empty:
        return []

    corr_rows = []
    metrics = ["EX", "EXP", "EXR", "F1"]

    # Group by error_pattern and calculate correlations for each pattern
    for pattern, pattern_df in df.groupby("error_pattern"):
        x = pattern_df["error_count"]

        for m in metrics:
            y = pattern_df[m]

            # Calculate correlations with robust error handling
            rho = safe_correlation(x, y, method="spearman")
            tau = safe_correlation(x, y, method="kendall")

            corr_rows.append(
                {
                    "technique": technique_name,
                    "error_pattern": pattern,
                    "metric": m,
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                }
            )

    return corr_rows


# ────────────────────────────────────────────────────────────────────────
# 5.  Extract unique error patterns from all data files
# ────────────────────────────────────────────────────────────────────────
def get_unique_error_patterns(techniques):
    """Extract all unique error patterns from the data files."""
    all_patterns = set()

    for tech in techniques:
        df = load_technique_data(tech["path"])
        if df is not None and "error_pattern" in df.columns:
            patterns = df["error_pattern"].unique()
            all_patterns.update(patterns)

    # Sort patterns - typically they are named like "pattern1", "pattern2" etc.
    # Try to sort numerically if pattern names follow this convention
    def extract_number(pattern_name):
        match = re.search(r"\d+", pattern_name)
        if match:
            return int(match.group())
        return pattern_name

    return sorted(list(all_patterns), key=extract_number)


# ────────────────────────────────────────────────────────────────────────
# 6.  Main processing pipeline
# ────────────────────────────────────────────────────────────────────────
def main():
    print("Experiment 1 - Multi-Technique Error Pattern Analysis")
    print("=" * 55)

    # Find all technique files
    techniques = find_mutant_score_files()

    if not techniques:
        print(f"[!] No mutant_scores_*.json files found in {SCORES_DIR}")
        return

    print(f"Found {len(techniques)} technique(s):")
    for tech in techniques:
        print(f"  • {tech['display_name']} ({tech['name']})")

    # Get unique error patterns from all files
    error_patterns = get_unique_error_patterns(techniques)

    if not error_patterns:
        print("[!] No error patterns found in data files.")
        return

    print(f"\nFound {len(error_patterns)} unique error pattern(s):")
    for pattern in error_patterns:
        print(f"  • {pattern}")

    # Load all technique data at once to avoid loading multiple times
    technique_data = {}
    for tech in techniques:
        technique_data[tech["name"]] = load_technique_data(tech["path"])

    # ────────────────────────────────────────────────────────────────────
    # Create figure with one row per error pattern, three cols per technique
    # ────────────────────────────────────────────────────────────────────
    n_rows = len(error_patterns)
    n_cols = 3  # Always 3 columns, show empty plots for missing techniques

    # Calculate figure size - height depends on number of error patterns
    fig_height = 4 * n_rows  # 4 inches per row
    fig_width = 18  # Fixed width

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle different subplot configurations
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    all_correlations = []

    # Process each error pattern in a separate row
    for row, pattern in enumerate(error_patterns):
        print(f"\nProcessing error pattern: {pattern}")

        # Process up to 3 techniques per row (columns)
        for col in range(n_cols):
            if col < len(techniques):
                technique = techniques[col]
                print(f"  Processing {technique['display_name']}...")

                # Get already loaded data
                df = technique_data[technique["name"]]

                # Plot for this pattern and technique
                plot_technique_scores(
                    axes[row, col], df, technique["display_name"], pattern
                )

                # Calculate correlations for this pattern
                if df is not None and not df.empty:
                    pattern_df = df[df["error_pattern"] == pattern]
                    if not pattern_df.empty:
                        corr_data = calculate_correlations(
                            pattern_df, technique["display_name"]
                        )
                        all_correlations.extend(corr_data)

                        # Print summary for this pattern and technique
                        mean_tbl = (
                            pattern_df.groupby("error_count")[
                                ["EX", "EXP", "EXR", "F1"]
                            ]
                            .mean()
                            .sort_index()
                        )
                        print(f"    Mean scores for '{pattern}' by #errors:")
                        print(f"    {mean_tbl.round(3).to_string()}")
            else:
                # Empty plot for missing techniques
                axes[row, col].text(
                    0.5,
                    0.5,
                    "No Technique Data",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[row, col].transAxes,
                    fontsize=10,
                    alpha=0.6,
                )
                axes[row, col].set_title(
                    f"Technique {col + 1} (Missing)", fontsize=11, fontweight="bold"
                )
                axes[row, col].set_xlabel("Operator Types")
                axes[row, col].set_ylabel("Mean Score")
                axes[row, col].set_ylim(0, 1.02)
                axes[row, col].grid(True, linewidth=0.3, alpha=0.6)

    # Save figure
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[✓] Figure 1 saved → {FIGURE_PATH}")
    plt.close()

    # ────────────────────────────────────────────────────────────────────
    # Create aggregated figure with one row across all error patterns
    # ────────────────────────────────────────────────────────────────────
    print("\nCreating aggregated plot across all error patterns...")

    # Single row, three columns for techniques
    agg_fig, agg_axes = plt.subplots(1, 3, figsize=(18, 5))

    for col in range(3):
        if col < len(techniques):
            technique = techniques[col]
            print(f"  Processing aggregated data for {technique['display_name']}...")

            # Get already loaded data
            df = technique_data[technique["name"]]

            # Plot aggregated data for this technique
            plot_aggregated_technique_scores(
                agg_axes[col], df, technique["display_name"]
            )
        else:
            # Empty plot for missing techniques
            agg_axes[col].text(
                0.5,
                0.5,
                "No Technique Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=agg_axes[col].transAxes,
                fontsize=10,
                alpha=0.6,
            )
            agg_axes[col].set_title(
                f"Technique {col + 1} (Missing)", fontsize=11, fontweight="bold"
            )
            agg_axes[col].set_xlabel("Error Count")
            agg_axes[col].set_ylabel("Mean Score")
            agg_axes[col].set_ylim(0, 1.02)
            agg_axes[col].grid(True, linewidth=0.3, alpha=0.6)

    # Add overall title to the aggregated figure
    agg_fig.suptitle("Aggregated Metrics Across All Error Patterns", fontsize=16)

    # Save aggregated figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig(AGGREGATE_FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"[✓] Figure 2 (Aggregated) saved → {AGGREGATE_FIGURE_PATH}")
    plt.close()

    # ────────────────────────────────────────────────────────────────────
    # Save correlations table
    # ────────────────────────────────────────────────────────────────────
    if all_correlations:
        cor_df = pd.DataFrame(all_correlations)
        cor_df.to_csv(TABLE_PATH, index=False)
        print(f"[✓] Correlations table saved → {TABLE_PATH}")

        # Print correlation summary
        print("\nCorrelation Summary by Error Pattern:")
        print("=" * 60)
        for pattern in error_patterns:
            print(f"\n## Error Pattern: {pattern} ##")

            for technique in techniques:
                pattern_tech_corr = cor_df[
                    (cor_df["error_pattern"] == pattern)
                    & (cor_df["technique"] == technique["display_name"])
                ]

                if not pattern_tech_corr.empty:
                    print(f"\n{technique['display_name']}:")
                    # Show correlations with better formatting
                    display_df = pattern_tech_corr[
                        ["metric", "spearman_rho", "kendall_tau"]
                    ].copy()
                    for col in ["spearman_rho", "kendall_tau"]:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.3f}" if pd.notna(x) else "NaN"
                        )
                    print(display_df.to_string(index=False))
    else:
        print("[!] No correlation data to save")


if __name__ == "__main__":
    main()

# report.py
"""
Experiment 1 – Sensitivity to Controlled Error Counts (Multi-Technique Analysis)
─────────────────────────────────────────────────────────────────────────────────
This script reads all mutant_scores_*.json files (one per evaluation technique) from the
specified directory and produces comprehensive analysis across multiple techniques and error patterns.

**Data Requirements:**
Each JSON file must contain records with the following structure:
- error_count: number of errors introduced
- error_pattern: identifier for the specific error pattern
- operators: list of mutation operators applied
- pattern_position: position of the current operator in the pattern (1-indexed)
- EX, EXP, EXR, F1: evaluation metrics (0-1 scale)

**Generated Outputs:**

1. **Figure 1** – Error Pattern Analysis by Technique:
   • Multi-row layout: one row per unique error pattern found in data
   • Three columns: one plot per technique (up to 3 techniques supported)
   • X-axis shows specific operator names extracted from the operators list and pattern_position
   • Y-axis shows mean scores for metrics EX, EXP, EXR, F1
   • Each metric plotted with distinct colors and line styles
   • Empty plots with "No Data" message for missing technique files or patterns
   • Saved as: figure1_error_patterns_multi_technique.png

2. **Figure 2** – Aggregated Analysis Across All Error Patterns:
   • Single row with three columns (one per technique)
   • X-axis shows simple error count labels (e1, e2, e3...)
   • Y-axis shows mean scores aggregated across all error patterns
   • Same metrics and styling as Figure 1
   • Overall title: "Aggregated Metrics Across All Error Patterns"
   • Saved as: figure2_aggregate_multi_technique.png

3. **Analysis Log** – Comprehensive Text Report:
   • Text file containing detailed analysis summary
   • Overview of techniques and error patterns
   • Data distribution and metric summaries by technique
   • Detailed breakdown by error pattern showing mean scores
   • Score trends and changes across error counts
   • Saved as: analysis_log.txt

**Key Features:**
- Automatic discovery of technique files using glob pattern matching
- Dynamic error pattern detection from data (supports any number of patterns)
- Robust error handling for missing files, incomplete data
- Intelligent operator labeling using pattern_position to extract specific operators
- Fallback mechanisms for backward compatibility with older data formats
- Comprehensive console output with processing summaries

**Directory Structure:**
- Input: data/evaluation/experiments/controlled_error_sensitivity/scores/ex1/
- Output: Same directory + /plots/ subdirectory (auto-created)
- File pattern: mutant_scores_{technique_name}.json

**Error Handling:**
- Missing technique files: empty plots with appropriate messages
- Missing error patterns: filtered out gracefully
- Missing data: handled with appropriate fallback values
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ────────────────────────────────────────────────────────────────────────
# 0.  File paths – adjust to your environment
# ────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")

SCORES_DIR = (
    ROOT / "data/evaluation/experiments/controlled_error_sensitivity/scores/ex4"
)
PLOTS_DIR = SCORES_DIR / "plots"
FIGURE_PATH = PLOTS_DIR / "figure1_error_patterns_multi_technique.png"
AGGREGATE_FIGURE_PATH = PLOTS_DIR / "figure2_aggregate_multi_technique.png"
LOG_PATH = SCORES_DIR / "analysis_log.txt"

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
# 2b. Get unique error patterns from all technique files
# ────────────────────────────────────────────────────────────────────────
def get_unique_error_patterns(techniques):
    """Extract unique error patterns from all technique files."""
    error_patterns = set()

    for tech in techniques:
        df = load_technique_data(tech["path"])
        if df is not None and not df.empty and "error_pattern" in df.columns:
            patterns = df["error_pattern"].dropna().unique()
            error_patterns.update(patterns)

    return sorted(list(error_patterns))


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
        ax.set_ylim(-0.05, 1.02)
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
        ax.set_ylim(-0.05, 1.02)
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
    ax.set_xticklabels(x_labels, rotation=15, ha="center")

    ax.set_ylim(-0.05, 1.02)
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
        ax.set_ylim(-0.05, 1.02)
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
    ax.set_xticklabels(x_labels, rotation=0, ha="center")

    ax.set_ylim(-0.05, 1.02)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(frameon=False, loc="upper right", fontsize=8)


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

                # Print summary for this pattern and technique
                if df is not None and not df.empty:
                    pattern_df = df[df["error_pattern"] == pattern]
                    if not pattern_df.empty:
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
                axes[row, col].set_ylim(-0.05, 1.02)
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
            agg_axes[col].set_ylim(-0.05, 1.02)
            agg_axes[col].grid(True, linewidth=0.3, alpha=0.6)

    # Add overall title to the aggregated figure
    agg_fig.suptitle("Aggregated Metrics Across All Error Patterns", fontsize=16)

    # Save aggregated figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig(AGGREGATE_FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"[✓] Figure 2 (Aggregated) saved → {AGGREGATE_FIGURE_PATH}")
    plt.close()

    # ────────────────────────────────────────────────────────────────────
    # Save analysis log
    # ────────────────────────────────────────────────────────────────────
    write_analysis_log(techniques, technique_data, error_patterns)
    print(f"[✓] Analysis log saved → {LOG_PATH}")

    print("\nAnalysis completed successfully!")


def write_analysis_log(techniques, technique_data, error_patterns):
    """Write comprehensive analysis log to text file."""
    with open(LOG_PATH, "w") as f:
        f.write("EXPERIMENT 1 - MULTI-TECHNIQUE ERROR PATTERN ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        # Overview section
        f.write("OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Techniques Analyzed: {len(techniques)}\n")
        f.write(f"Total Error Patterns Found: {len(error_patterns)}\n\n")

        f.write("Techniques:\n")
        for i, tech in enumerate(techniques, 1):
            f.write(f"  {i}. {tech['display_name']} ({tech['name']})\n")
        f.write("\n")

        f.write("Error Patterns:\n")
        for i, pattern in enumerate(error_patterns, 1):
            f.write(f"  {i}. {pattern}\n")
        f.write("\n")

        # Data summary section
        f.write("DATA SUMMARY BY TECHNIQUE\n")
        f.write("-" * 30 + "\n")

        total_records = 0
        for tech in techniques:
            df = technique_data[tech["name"]]
            if df is not None and not df.empty:
                tech_records = len(df)
                total_records += tech_records

                f.write(f"\n{tech['display_name']}:\n")
                f.write(f"  Total Records: {tech_records}\n")

                # Error count distribution
                error_counts = df["error_count"].value_counts().sort_index()
                f.write("  Error Count Distribution:\n")
                for count, freq in error_counts.items():
                    percentage = (freq / tech_records) * 100
                    f.write(f"    {count} errors: {freq} records ({percentage:.1f}%)\n")

                # Pattern distribution
                pattern_counts = df["error_pattern"].value_counts()
                f.write("  Pattern Distribution:\n")
                for pattern, freq in pattern_counts.items():
                    percentage = (freq / tech_records) * 100
                    f.write(f"    {pattern}: {freq} records ({percentage:.1f}%)\n")

                # Metric summary statistics
                metrics = ["EX", "EXP", "EXR", "F1"]
                f.write("  Metric Summary Statistics:\n")
                for metric in metrics:
                    if metric in df.columns:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        min_val = df[metric].min()
                        max_val = df[metric].max()
                        f.write(
                            f"    {metric}: mean={mean_val:.3f}, std={std_val:.3f}, "
                        )
                        f.write(f"range=[{min_val:.3f}, {max_val:.3f}]\n")
            else:
                f.write(f"\n{tech['display_name']}: No data available\n")

        f.write(f"\nTotal Records Across All Techniques: {total_records}\n\n")

        # Detailed analysis by error pattern
        f.write("DETAILED ANALYSIS BY ERROR PATTERN\n")
        f.write("-" * 40 + "\n")

        for pattern in error_patterns:
            f.write(f"\n{'='*50}\n")
            f.write(f"ERROR PATTERN: {pattern}\n")
            f.write(f"{'='*50}\n")

            pattern_has_data = False

            for tech in techniques:
                df = technique_data[tech["name"]]
                if df is not None and not df.empty:
                    pattern_df = df[df["error_pattern"] == pattern]

                    if not pattern_df.empty:
                        pattern_has_data = True
                        f.write(f"\n{tech['display_name']}:\n")
                        f.write(f"  Records for this pattern: {len(pattern_df)}\n")

                        # Mean scores by error count
                        mean_tbl = (
                            pattern_df.groupby("error_count")[
                                ["EX", "EXP", "EXR", "F1"]
                            ]
                            .mean()
                            .sort_index()
                        )
                        f.write("  Mean Scores by Error Count:\n")
                        f.write(
                            f"    {'Error':<6} {'EX':<8} {'EXP':<8} {'EXR':<8} {'F1':<8}\n"
                        )
                        f.write(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")

                        for error_count, row in mean_tbl.iterrows():
                            f.write(f"    {error_count:<6} ")
                            f.write(f"{row['EX']:<8.3f} {row['EXP']:<8.3f} ")
                            f.write(f"{row['EXR']:<8.3f} {row['F1']:<8.3f}\n")

                        # Score ranges and trends
                        f.write("  Score Trends:\n")
                        for metric in ["EX", "EXP", "EXR", "F1"]:
                            first_score = (
                                mean_tbl.iloc[0][metric] if len(mean_tbl) > 0 else 0
                            )
                            last_score = (
                                mean_tbl.iloc[-1][metric] if len(mean_tbl) > 0 else 0
                            )
                            trend = last_score - first_score
                            trend_direction = (
                                "↑" if trend > 0 else "↓" if trend < 0 else "→"
                            )
                            f.write(
                                f"    {metric}: {first_score:.3f} → {last_score:.3f} "
                            )
                            f.write(f"(Δ={trend:+.3f} {trend_direction})\n")

            if not pattern_has_data:
                f.write(f"  No data available for pattern '{pattern}'\n")

        f.write(
            f"\nAnalysis completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


if __name__ == "__main__":
    main()

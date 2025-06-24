# report.py
"""
Experiment 1 – Sensitivity to Controlled Error Counts (Multi-Technique Analysis)
─────────────────────────────────────────────────────────────────────────────────
This script reads all mutant_scores_*.json files (one per evaluation technique) and produces:

1. **Figure 1** – mean score vs. #injected errors (EX, EXP, EXR, F1) for each technique.
   • Three plots in one row, one per technique
   • Distinct colours / linestyles, slight transparency.
   • X-axis shows *only integer ticks* (1, 2, 3 …).
   • If a technique file doesn't exist, shows empty plot with "No Data" message

2. **Table 1** – Spearman ρ and Kendall τ correlations between
   error_count and each metric for each technique.
   • Rows with NaNs in a metric are ignored (nan_policy="omit").
   • If fewer than two valid points remain (or all values equal),
     the correlation is reported as NaN.

Artefacts are saved next to the scores files.
"""

from pathlib import Path
import json
import glob
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # integer x-ticks
from scipy.stats import spearmanr, kendalltau

# ────────────────────────────────────────────────────────────────────────
# 0.  File paths – adjust to your environment
# ────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")

SCORES_DIR = ROOT / "data/evaluation/metrics/experiment_1"
FIGURE_PATH = (
    ROOT
    / "data/evaluation/metrics/experiment_1/figure1_mean_vs_errors_multi_technique.png"
)
TABLE_PATH = (
    ROOT
    / "data/evaluation/metrics/experiment_1/table1_correlations_multi_technique.csv"
)


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
        missing = required_cols.difference(df.columns)

        if missing:
            print(f"[!] Warning: {file_path.name} missing columns: {missing}")
            return None

        return df
    except Exception as e:
        print(f"[!] Error loading {file_path.name}: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────
# 3.  Plot mean scores for a single technique
# ────────────────────────────────────────────────────────────────────────
def plot_technique_scores(ax, df, technique_name):
    """Plot mean scores vs error count for a single technique."""
    if df is None or df.empty:
        ax.text(
            0.5,
            0.5,
            "No Data Available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            alpha=0.6,
        )
        ax.set_title(technique_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("# injected errors")
        ax.set_ylabel("mean score")
        ax.set_ylim(0, 1.02)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        return

    # Calculate mean scores per error count
    mean_tbl = df.groupby("error_count")[["EX", "EXP", "EXR", "F1"]].mean().sort_index()

    metrics = ["EX", "EXP", "EXR", "F1"]
    linestyles = ["-", "--", ":", "-."]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, m in enumerate(metrics):
        ax.plot(
            mean_tbl.index,
            mean_tbl[m],
            label=m,
            linestyle=linestyles[i],
            color=colours[i],
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=5,
        )

    ax.set_title(technique_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("# injected errors")
    ax.set_ylabel("mean score")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer ticks only
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
    """Calculate correlations for a single technique with robust error handling."""
    if df is None or df.empty:
        return []

    corr_rows = []
    x = df["error_count"]
    metrics = ["EX", "EXP", "EXR", "F1"]

    for m in metrics:
        y = df[m]

        # Calculate correlations with robust error handling
        rho = safe_correlation(x, y, method="spearman")
        tau = safe_correlation(x, y, method="kendall")

        corr_rows.append(
            {
                "technique": technique_name,
                "metric": m,
                "spearman_rho": rho,
                "kendall_tau": tau,
            }
        )

    return corr_rows


# ────────────────────────────────────────────────────────────────────────
# 5.  Main processing pipeline
# ────────────────────────────────────────────────────────────────────────
def main():
    print("Experiment 1 - Multi-Technique Sensitivity Analysis")
    print("=" * 55)

    # Find all technique files
    techniques = find_mutant_score_files()

    if not techniques:
        print(f"[!] No mutant_scores_*.json files found in {SCORES_DIR}")
        return

    print(f"Found {len(techniques)} technique(s):")
    for tech in techniques:
        print(f"  • {tech['display_name']} ({tech['name']})")

    # ────────────────────────────────────────────────────────────────────
    # Create figure with exactly 3 plots in one row (as requested)
    # ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    all_correlations = []

    # Process up to 3 techniques (pad with empty plots if fewer)
    for i in range(3):
        if i < len(techniques):
            technique = techniques[i]
            print(f"\nProcessing {technique['display_name']}...")

            # Load data
            df = load_technique_data(technique["path"])

            # Plot
            plot_technique_scores(axes[i], df, technique["display_name"])

            # Calculate correlations
            corr_data = calculate_correlations(df, technique["display_name"])
            all_correlations.extend(corr_data)

            if df is not None and not df.empty:
                # Print summary for this technique
                mean_tbl = (
                    df.groupby("error_count")[["EX", "EXP", "EXR", "F1"]]
                    .mean()
                    .sort_index()
                )
                print("  Mean scores by #errors:")
                print(f"  {mean_tbl.round(3).to_string()}")

                # Print data diagnostics
                print("  Data diagnostics:")
                print(f"    Total records: {len(df)}")
                print(f"    Error counts: {sorted(df['error_count'].unique())}")
                for metric in ["EX", "EXP", "EXR", "F1"]:
                    non_null = df[metric].notna().sum()
                    unique_vals = df[metric].nunique()
                    print(
                        f"    {metric}: {non_null} non-null, {unique_vals} unique values"
                    )
        else:
            # Empty plot for missing techniques
            axes[i].text(
                0.5,
                0.5,
                "No Technique File",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[i].transAxes,
                fontsize=12,
                alpha=0.6,
            )
            axes[i].set_title(
                f"Technique {i + 1} (Missing)", fontsize=12, fontweight="bold"
            )
            axes[i].set_xlabel("# injected errors")
            axes[i].set_ylabel("mean score")
            axes[i].set_ylim(0, 1.02)
            axes[i].grid(True, linewidth=0.3, alpha=0.6)

    # Save figure
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"\n[✓] Figure saved → {FIGURE_PATH}")

    # ────────────────────────────────────────────────────────────────────
    # Save correlations table
    # ────────────────────────────────────────────────────────────────────
    if all_correlations:
        cor_df = pd.DataFrame(all_correlations)
        cor_df.to_csv(TABLE_PATH, index=False)
        print(f"[✓] Correlations table saved → {TABLE_PATH}")

        # Print correlation summary
        print("\nCorrelation Summary:")
        print("=" * 40)
        for technique in techniques:
            tech_corr = cor_df[cor_df["technique"] == technique["display_name"]]
            if not tech_corr.empty:
                print(f"\n{technique['display_name']}:")
                # Show correlations with better formatting
                display_df = tech_corr[["metric", "spearman_rho", "kendall_tau"]].copy()
                for col in ["spearman_rho", "kendall_tau"]:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.3f}" if pd.notna(x) else "NaN"
                    )
                print(display_df.to_string(index=False))
    else:
        print("[!] No correlation data to save")


if __name__ == "__main__":
    main()

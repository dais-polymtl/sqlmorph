import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

try:
    from matplotlib_venn import venn3
except ImportError:
    print("matplotlib-venn not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "matplotlib-venn"])
    from matplotlib_venn import venn3


def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the CSV file and prepare data for analysis.

    Args:
        csv_path: Path to the systems_data_with_metrics.csv file

    Returns:
        Tuple of (filtered DataFrame, pivoted DataFrame with extracted EX values)
    """
    df = pd.read_csv(csv_path)

    # Extract EX values from EXECUTION_ACCURACY column
    def extract_ex_value(execution_accuracy_str):
        try:
            if pd.isna(execution_accuracy_str) or execution_accuracy_str in [
                "timeout",
                "error",
            ]:
                return -1  # Mark as invalid
            data = json.loads(execution_accuracy_str)
            return data.get("EX", -1)
        except (json.JSONDecodeError, TypeError):
            return -1

    df["ex"] = df["EXECUTION_ACCURACY"].apply(extract_ex_value)

    # Filter for the three systems we're interested in
    target_systems = ["chess", "din-sql", "mac-sql"]
    df_filtered = df[df["system"].isin(target_systems)].copy()

    # Create pivot table for easier intersection calculations
    df_pivot = df_filtered.pivot_table(
        index="question_id", columns="system", values="ex", fill_value="-1"
    )

    return df_filtered, df_pivot


def calculate_system_stats(df_pivot: pd.DataFrame) -> Tuple[Dict, int]:
    """
    Calculate basic statistics for each system.

    Args:
        df_pivot: Pivoted DataFrame with systems as columns

    Returns:
        Tuple of (dictionary with system statistics, total number of questions)
    """
    total_questions = df_pivot.shape[0]
    stats = {}

    for system in df_pivot.columns:
        correct = (df_pivot[system] == 1).sum()
        failed = (df_pivot[system] == 0).sum()
        stats[system] = {
            "correct": int(correct),
            "failed": int(failed),
            "correct_percentage": (correct / total_questions) * 100,
            "failed_percentage": (failed / total_questions) * 100,
        }

    return stats, total_questions


def intersection_systems(
    df_pivot: pd.DataFrame,
    system_a: str,
    system_b: str,
    correct: bool = True,
    third_system: str = None,
) -> int:
    """
    Calculate intersection between systems for correct or failed answers.

    Args:
        df_pivot: Pivoted DataFrame
        system_a: First system name
        system_b: Second system name
        correct: If True, calculate for correct answers (EX=1), else failed (EX=0)
        third_system: Optional third system for three-way intersection

    Returns:
        Count of intersecting cases
    """
    value = 1 if correct else 0

    if third_system:
        return df_pivot[
            (df_pivot[system_a] == value)
            & (df_pivot[system_b] == value)
            & (df_pivot[third_system] == value)
        ].shape[0]
    else:
        return df_pivot[
            (df_pivot[system_a] == value) & (df_pivot[system_b] == value)
        ].shape[0]


def calculate_intersections(df_pivot: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Calculate all pairwise and three-way intersections for correct and failed answers.

    Args:
        df_pivot: Pivoted DataFrame

    Returns:
        Tuple of (correct_intersections, failed_intersections) dictionaries
    """
    # Calculate intersections for correct answers (EX=1)
    correct_intersections = {
        "chess_din": intersection_systems(df_pivot, "chess", "din-sql", correct=True),
        "din_mac": intersection_systems(df_pivot, "din-sql", "mac-sql", correct=True),
        "chess_mac": intersection_systems(df_pivot, "chess", "mac-sql", correct=True),
        "all_three": intersection_systems(
            df_pivot, "chess", "din-sql", correct=True, third_system="mac-sql"
        ),
    }

    # Calculate intersections for failed answers (EX=0)
    failed_intersections = {
        "chess_din": intersection_systems(df_pivot, "chess", "din-sql", correct=False),
        "din_mac": intersection_systems(df_pivot, "din-sql", "mac-sql", correct=False),
        "chess_mac": intersection_systems(df_pivot, "chess", "mac-sql", correct=False),
        "all_three": intersection_systems(
            df_pivot, "chess", "din-sql", correct=False, third_system="mac-sql"
        ),
    }

    return correct_intersections, failed_intersections


def print_intersection_table(
    stats: Dict,
    correct_intersections: Dict,
    failed_intersections: Dict,
    total_questions: int,
) -> None:
    """
    Print formatted intersection table similar to the provided example.

    Args:
        stats: System statistics dictionary
        correct_intersections: Correct answer intersections
        failed_intersections: Failed answer intersections
        total_questions: Total number of questions
    """
    print(
        f"{'System':<35}{'Correct':>10}{'Failed':>10}{'Correct (%)':>15}{'Failed (%)':>15}"
    )
    print("-" * 85)

    # Print individual system results
    for system in ["chess", "din-sql", "mac-sql"]:
        if system in stats:
            stat = stats[system]
            print(
                f"{system:<35}{stat['correct']:>10}{stat['failed']:>10}"
                f"{stat['correct_percentage']:>15.2f}{stat['failed_percentage']:>15.2f}"
            )

    # Print correct intersections
    print("\nIntersection Type (Correct Answers)")
    print(f"{' ':<35}{' ':<10}{'Percentage (correct/total)':>35}")
    print("-" * 75)

    intersections_correct = [
        ("chess and din-sql", correct_intersections["chess_din"]),
        ("din-sql and mac-sql", correct_intersections["din_mac"]),
        ("chess and mac-sql", correct_intersections["chess_mac"]),
        ("chess, din-sql, mac-sql", correct_intersections["all_three"]),
    ]

    for name, count in intersections_correct:
        percentage = (count / total_questions) * 100
        print(f"{name:<35}{count:>10}{percentage:>25.2f}%")

    # Print failed intersections
    print("\nIntersection Type (Failed Answers)")
    print(f"{' ':<35}{' ':<10}{'Percentage (failed/total)':>35}")
    print("-" * 75)

    intersections_failed = [
        ("chess and din-sql", failed_intersections["chess_din"]),
        ("din-sql and mac-sql", failed_intersections["din_mac"]),
        ("chess and mac-sql", failed_intersections["chess_mac"]),
        ("chess, din-sql, mac-sql", failed_intersections["all_three"]),
    ]

    for name, count in intersections_failed:
        percentage = (count / total_questions) * 100
        print(f"{name:<35}{count:>10}{percentage:>25.2f}%")


def make_label(count: int, total: int) -> str:
    """
    Create custom labels with both count and percentage for Venn diagrams.

    Args:
        count: Number of items
        total: Total number of items

    Returns:
        Formatted label string
    """
    percentage = (count / total) * 100
    return f"{count}\n({percentage:.2f}%)"


def create_venn_diagram(
    df_pivot: pd.DataFrame, correct: bool = True, output_dir: str = "plots"
) -> None:
    """
    Create Venn diagram for correct or failed answers.

    Args:
        df_pivot: Pivoted DataFrame
        correct: If True, create diagram for correct answers, else for failed
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_questions = df_pivot.shape[0]
    value = 1 if correct else 0

    # Calculate counts for each system and intersections
    counts = {
        "chess": df_pivot[(df_pivot["chess"] == value)].shape[0],
        "din": df_pivot[(df_pivot["din-sql"] == value)].shape[0],
        "mac": df_pivot[(df_pivot["mac-sql"] == value)].shape[0],
        "chess_din": intersection_systems(
            df_pivot, "chess", "din-sql", correct=correct
        ),
        "din_mac": intersection_systems(
            df_pivot, "din-sql", "mac-sql", correct=correct
        ),
        "chess_mac": intersection_systems(
            df_pivot, "chess", "mac-sql", correct=correct
        ),
        "all_three": intersection_systems(
            df_pivot, "chess", "din-sql", correct=correct, third_system="mac-sql"
        ),
    }

    # Create the Venn diagram
    plt.figure(figsize=(10, 8))
    venn = venn3(
        subsets=(
            counts["chess"]
            - counts["chess_din"]
            - counts["chess_mac"]
            + counts["all_three"],  # Only chess
            counts["din"]
            - counts["chess_din"]
            - counts["din_mac"]
            + counts["all_three"],  # Only din-sql
            counts["chess_din"] - counts["all_three"],  # chess and din-sql intersection
            counts["mac"]
            - counts["chess_mac"]
            - counts["din_mac"]
            + counts["all_three"],  # Only mac-sql
            counts["chess_mac"] - counts["all_three"],  # chess and mac-sql intersection
            counts["din_mac"] - counts["all_three"],  # din-sql and mac-sql intersection
            counts["all_three"],
        ),  # All three intersection
        set_labels=("chess", "din-sql", "mac-sql"),
    )

    # Add custom labels with both counts and percentages
    venn_labels = {
        "100": make_label(
            counts["chess"]
            - counts["chess_din"]
            - counts["chess_mac"]
            + counts["all_three"],
            total_questions,
        ),
        "010": make_label(
            counts["din"]
            - counts["chess_din"]
            - counts["din_mac"]
            + counts["all_three"],
            total_questions,
        ),
        "110": make_label(counts["chess_din"] - counts["all_three"], total_questions),
        "001": make_label(
            counts["mac"]
            - counts["chess_mac"]
            - counts["din_mac"]
            + counts["all_three"],
            total_questions,
        ),
        "101": make_label(counts["chess_mac"] - counts["all_three"], total_questions),
        "011": make_label(counts["din_mac"] - counts["all_three"], total_questions),
        "111": make_label(counts["all_three"], total_questions),
    }

    # Update the Venn diagram labels
    for region_id, label in venn_labels.items():
        if venn.get_label_by_id(region_id):
            venn.get_label_by_id(region_id).set_text(label)
            venn.get_label_by_id(region_id).set_fontsize(12)

    # Increase the font size of system labels (set labels)
    for label in venn.set_labels:
        if label:
            label.set_fontsize(16)
            label.set_fontweight("bold")

    # Customize colors and appearance
    if venn.get_patch_by_id("100"):
        venn.get_patch_by_id("100").set_color("#ff9999")
        venn.get_patch_by_id("100").set_alpha(0.7)
        venn.get_patch_by_id("100").set_edgecolor("red")
        venn.get_patch_by_id("100").set_linewidth(2)
        venn.get_patch_by_id("100").set_linestyle("-")  # Solid line for chess
    if venn.get_patch_by_id("010"):
        venn.get_patch_by_id("010").set_color("#66b3ff")
        venn.get_patch_by_id("010").set_alpha(0.7)
        venn.get_patch_by_id("010").set_edgecolor("blue")
        venn.get_patch_by_id("010").set_linewidth(2)
        venn.get_patch_by_id("010").set_linestyle("--")  # Dashed line for din-sql
    if venn.get_patch_by_id("001"):
        venn.get_patch_by_id("001").set_color("#99ff99")
        venn.get_patch_by_id("001").set_alpha(0.7)
        venn.get_patch_by_id("001").set_edgecolor("green")
        venn.get_patch_by_id("001").set_linewidth(2)
        venn.get_patch_by_id("001").set_linestyle(":")  # Dotted line for mac-sql

    # Apply line styles to intersection patches as well
    intersection_patches = ["110", "101", "011"]
    line_styles = ["-", "--", ":"]  # Different styles for intersections
    colors = ["purple", "orange", "brown"]

    for i, patch_id in enumerate(intersection_patches):
        if venn.get_patch_by_id(patch_id):
            venn.get_patch_by_id(patch_id).set_edgecolor(colors[i])
            venn.get_patch_by_id(patch_id).set_linewidth(2)
            venn.get_patch_by_id(patch_id).set_linestyle(line_styles[i])

    # Style the three-way intersection separately
    if venn.get_patch_by_id("111"):
        venn.get_patch_by_id("111").set_edgecolor("black")
        venn.get_patch_by_id("111").set_linewidth(3)
        venn.get_patch_by_id("111").set_linestyle("-")

    # Set title and labels
    if correct:
        # title = "Success Cases Distribution Over Systems (EX=1)"
        answer_type = "Correct"
    else:
        # title = "Failure Cases Distribution Over Systems (EX=0)"
        answer_type = "Failed"

    # plt.title(title, fontsize=14, fontweight="bold")

    # Save the plot in both PNG and PDF formats
    filename_base = f"{answer_type.lower()}_answers_venn"
    plt.savefig(
        os.path.join(output_dir, f"{filename_base}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, f"{filename_base}.pdf"), bbox_inches="tight")
    plt.show()


def run_analytics(csv_path: str, output_dir: str = "plots") -> None:
    """
    Main function to run the complete analytics pipeline.

    Args:
        csv_path: Path to the systems_data_with_metrics.csv file
        output_dir: Directory to save plots
    """
    print("Loading and preparing data...")
    df_filtered, df_pivot = load_and_prepare_data(csv_path)

    print("Calculating statistics...")
    stats, total_questions = calculate_system_stats(df_pivot)
    correct_intersections, failed_intersections = calculate_intersections(df_pivot)

    print("\nIntersection Analysis Results:")
    print("=" * 85)
    print_intersection_table(
        stats, correct_intersections, failed_intersections, total_questions
    )

    print("\nGenerating Venn diagrams...")
    create_venn_diagram(df_pivot, correct=True, output_dir=output_dir)
    create_venn_diagram(df_pivot, correct=False, output_dir=output_dir)

    print(f"\nAnalysis complete! Plots saved to '{output_dir}' directory.")


if __name__ == "__main__":
    # Example usage
    ROOT = "data/metrics/experiments/system_level_comparison"
    csv_path = ROOT + "/systems_data_with_metrics-with_penalty.csv"
    output_dir = ROOT + "/plots"

    run_analytics(csv_path, output_dir)

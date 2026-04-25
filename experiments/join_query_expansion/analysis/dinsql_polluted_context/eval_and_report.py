import pandas as pd
import os
from tqdm import tqdm
import signal
from contextlib import contextmanager

from src.metrics.evaluation import Evaluation, EvaluationTechnique
from src.core.database.database_handler import DBMS
from src.core.model_manager import OpenAIModel


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to set a timeout for code execution."""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def generate_report(df: pd.DataFrame, report_path: str):
    """
    Generate comprehensive report analyzing EX transitions from ex_expanded to ex_original.

    Args:
        df: DataFrame with ex_original, ex_expanded, and delta_ex columns
        report_path: Path to save the report text file
    """
    # Filter out rows with missing values
    valid_df = df[df["ex_original"].notna() & df["ex_expanded"].notna()].copy()
    total_valid = len(valid_df)

    # Build report content
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("=" * 80)
    report_lines.append("                         EXECUTION ACCURACY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("=" * 80)

    if total_valid == 0:
        report_lines.append("\nNo valid data points to analyze.")
        report_content = "\n".join(report_lines)
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"\nReport saved to: {report_path}")
        return

    # Count transitions from ex_expanded -> ex_original
    transition_1_to_0 = len(
        valid_df[(valid_df["ex_expanded"] == 1) & (valid_df["ex_original"] == 0)]
    )
    transition_0_to_1 = len(
        valid_df[(valid_df["ex_expanded"] == 0) & (valid_df["ex_original"] == 1)]
    )
    transition_0_to_0 = len(
        valid_df[(valid_df["ex_expanded"] == 0) & (valid_df["ex_original"] == 0)]
    )
    transition_1_to_1 = len(
        valid_df[(valid_df["ex_expanded"] == 1) & (valid_df["ex_original"] == 1)]
    )

    report_lines.append("\n📊 Dataset Overview:")
    report_lines.append(f"   Total rows in dataset: {len(df)}")
    report_lines.append(
        f"   Valid data points (both EX metrics computed): {total_valid}"
    )
    report_lines.append(f"   Missing data points: {len(df) - total_valid}")

    report_lines.append("\n📈 Transition Analysis (ex_expanded → ex_original):")
    report_lines.append("   ┌─────────────────────────────────────────────┐")
    report_lines.append("   │  Transition  │  Count  │  Percentage       │")
    report_lines.append("   ├─────────────────────────────────────────────┤")
    report_lines.append(
        f"   │  1 → 0       │  {transition_1_to_0:4d}   │  {(transition_1_to_0/total_valid)*100:5.2f}%         │"
    )
    report_lines.append(
        f"   │  0 → 1       │  {transition_0_to_1:4d}   │  {(transition_0_to_1/total_valid)*100:5.2f}%         │"
    )
    report_lines.append(
        f"   │  0 → 0       │  {transition_0_to_0:4d}   │  {(transition_0_to_0/total_valid)*100:5.2f}%         │"
    )
    report_lines.append(
        f"   │  1 → 1       │  {transition_1_to_1:4d}   │  {(transition_1_to_1/total_valid)*100:5.2f}%         │"
    )
    report_lines.append("   └─────────────────────────────────────────────┘")

    report_lines.append("\n📉 Performance Change Analysis:")
    improved = transition_0_to_1
    degraded = transition_1_to_0
    unchanged = transition_0_to_0 + transition_1_to_1

    report_lines.append(
        f"   ✅ Improved (0 → 1):    {improved:4d}  ({(improved/total_valid)*100:5.2f}%)"
    )
    report_lines.append(
        f"   ❌ Degraded (1 → 0):    {degraded:4d}  ({(degraded/total_valid)*100:5.2f}%)"
    )
    report_lines.append(
        f"   ➖ Unchanged:           {unchanged:4d}  ({(unchanged/total_valid)*100:5.2f}%)"
    )

    report_lines.append("\n📊 Summary Statistics:")
    report_lines.append("   EX Expanded (final_query_expanded vs gold_query):")
    report_lines.append(f"      Mean:  {valid_df['ex_expanded'].mean():.4f}")
    report_lines.append(f"      Sum:   {int(valid_df['ex_expanded'].sum())}")
    report_lines.append(
        f"      Rate:  {(valid_df['ex_expanded'].sum()/total_valid)*100:.2f}%"
    )

    report_lines.append("\n   EX Original (final_query_original vs org_SQL):")
    report_lines.append(f"      Mean:  {valid_df['ex_original'].mean():.4f}")
    report_lines.append(f"      Sum:   {int(valid_df['ex_original'].sum())}")
    report_lines.append(
        f"      Rate:  {(valid_df['ex_original'].sum()/total_valid)*100:.2f}%"
    )

    report_lines.append("\n   Delta EX (ex_original - ex_expanded):")
    report_lines.append(f"      Mean:  {valid_df['delta_ex'].mean():+.4f}")
    report_lines.append(f"      Std:   {valid_df['delta_ex'].std():.4f}")
    report_lines.append(f"      Min:   {valid_df['delta_ex'].min():+.0f}")
    report_lines.append(f"      Max:   {valid_df['delta_ex'].max():+.0f}")

    # Net change
    net_change = int(valid_df["ex_original"].sum() - valid_df["ex_expanded"].sum())
    net_change_pct = (net_change / total_valid) * 100

    report_lines.append("\n🔄 Net Change:")
    report_lines.append(
        f"   Total change in correct predictions: {net_change:+d} ({net_change_pct:+.2f}%)"
    )

    if net_change > 0:
        report_lines.append(
            "   📈 Overall IMPROVEMENT from expanded to original context"
        )
    elif net_change < 0:
        report_lines.append(
            "   📉 Overall DEGRADATION from expanded to original context"
        )
    else:
        report_lines.append("   ➖ No net change (improvements = degradations)")

    report_lines.append("\n" + "=" * 80 + "\n")

    # Write report to file
    report_content = "\n".join(report_lines)
    with open(report_path, "w") as f:
        f.write(report_content)

    # Also print to console
    print(report_content)
    print(f"\nReport saved to: {report_path}")


def generate_grouped_reports(df: pd.DataFrame, report_path: str):
    """
    Generate reports grouped by n_join column.

    Args:
        df: DataFrame with ex_original, ex_expanded, delta_ex, and n_join columns
        report_path: Base path to save the grouped report text files
    """
    # Check if n_join column exists
    if "n_join" not in df.columns:
        print(
            "\nWarning: 'n_join' column not found in dataframe. Skipping grouped reports."
        )
        return

    # Get unique n_join values
    n_join_values = sorted(df["n_join"].dropna().unique())
    print(f"\nFound n_join groups: {n_join_values}")

    # Build overall grouped report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("=" * 80)
    report_lines.append("               EXECUTION ACCURACY REPORT - GROUPED BY N_JOIN")
    report_lines.append("=" * 80)
    report_lines.append("=" * 80)

    for n_join_val in n_join_values:
        group_df = df[df["n_join"] == n_join_val]
        valid_df = group_df[
            group_df["ex_original"].notna() & group_df["ex_expanded"].notna()
        ].copy()
        total_valid = len(valid_df)

        report_lines.append(f"\n\n{'='*80}")
        report_lines.append(f"GROUP: n_join = {n_join_val}")
        report_lines.append(f"{'='*80}")

        if total_valid == 0:
            report_lines.append("\nNo valid data points to analyze for this group.")
            continue

        # Count transitions
        transition_1_to_0 = len(
            valid_df[(valid_df["ex_expanded"] == 1) & (valid_df["ex_original"] == 0)]
        )
        transition_0_to_1 = len(
            valid_df[(valid_df["ex_expanded"] == 0) & (valid_df["ex_original"] == 1)]
        )
        transition_0_to_0 = len(
            valid_df[(valid_df["ex_expanded"] == 0) & (valid_df["ex_original"] == 0)]
        )
        transition_1_to_1 = len(
            valid_df[(valid_df["ex_expanded"] == 1) & (valid_df["ex_original"] == 1)]
        )

        report_lines.append("\n📊 Dataset Overview:")
        report_lines.append(f"   Total rows in group: {len(group_df)}")
        report_lines.append(
            f"   Valid data points (both EX metrics computed): {total_valid}"
        )
        report_lines.append(f"   Missing data points: {len(group_df) - total_valid}")

        report_lines.append("\n📈 Transition Analysis (ex_expanded → ex_original):")
        report_lines.append("   ┌─────────────────────────────────────────────┐")
        report_lines.append("   │  Transition  │  Count  │  Percentage        │")
        report_lines.append("   ├─────────────────────────────────────────────┤")
        report_lines.append(
            f"   │  1 → 0       │  {transition_1_to_0:4d}      │  {(transition_1_to_0/total_valid)*100:5.2f}%         │"
        )
        report_lines.append(
            f"   │  0 → 1       │  {transition_0_to_1:4d}      │  {(transition_0_to_1/total_valid)*100:5.2f}%         │"
        )
        report_lines.append(
            f"   │  0 → 0       │  {transition_0_to_0:4d}      │  {(transition_0_to_0/total_valid)*100:5.2f}%         │"
        )
        report_lines.append(
            f"   │  1 → 1       │  {transition_1_to_1:4d}      │  {(transition_1_to_1/total_valid)*100:5.2f}%         │"
        )
        report_lines.append("   └─────────────────────────────────────────────┘")

        report_lines.append("\n📉 Performance Change Analysis:")
        improved = transition_0_to_1
        degraded = transition_1_to_0
        unchanged = transition_0_to_0 + transition_1_to_1

        report_lines.append(
            f"   ✅ Improved (0 → 1):    {improved:4d}  ({(improved/total_valid)*100:5.2f}%)"
        )
        report_lines.append(
            f"   ❌ Degraded (1 → 0):    {degraded:4d}  ({(degraded/total_valid)*100:5.2f}%)"
        )
        report_lines.append(
            f"   ➖ Unchanged:           {unchanged:4d}  ({(unchanged/total_valid)*100:5.2f}%)"
        )

        report_lines.append("\n📊 Summary Statistics:")
        report_lines.append("   EX Expanded (final_query_expanded vs gold_query):")
        report_lines.append(f"      Mean:  {valid_df['ex_expanded'].mean():.4f}")
        report_lines.append(f"      Sum:   {int(valid_df['ex_expanded'].sum())}")
        report_lines.append(
            f"      Rate:  {(valid_df['ex_expanded'].sum()/total_valid)*100:.2f}%"
        )

        report_lines.append("\n   EX Original (final_query_original vs org_SQL):")
        report_lines.append(f"      Mean:  {valid_df['ex_original'].mean():.4f}")
        report_lines.append(f"      Sum:   {int(valid_df['ex_original'].sum())}")
        report_lines.append(
            f"      Rate:  {(valid_df['ex_original'].sum()/total_valid)*100:.2f}%"
        )

        report_lines.append("\n   Delta EX (ex_original - ex_expanded):")
        report_lines.append(f"      Mean:  {valid_df['delta_ex'].mean():+.4f}")
        report_lines.append(f"      Std:   {valid_df['delta_ex'].std():.4f}")
        report_lines.append(f"      Min:   {valid_df['delta_ex'].min():+.0f}")
        report_lines.append(f"      Max:   {valid_df['delta_ex'].max():+.0f}")

        # Net change
        net_change = int(valid_df["ex_original"].sum() - valid_df["ex_expanded"].sum())
        net_change_pct = (net_change / total_valid) * 100

        report_lines.append("\n🔄 Net Change:")
        report_lines.append(
            f"   Total change in correct predictions: {net_change:+d} ({net_change_pct:+.2f}%)"
        )

        if net_change > 0:
            report_lines.append(
                "   📈 Overall IMPROVEMENT from expanded to original context"
            )
        elif net_change < 0:
            report_lines.append(
                "   📉 Overall DEGRADATION from expanded to original context"
            )
        else:
            report_lines.append("   ➖ No net change (improvements = degradations)")

    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("=" * 80)
    report_lines.append("\n")

    # Write grouped report to file
    report_content = "\n".join(report_lines)
    grouped_report_path = report_path.replace(".txt", "_grouped_by_n_join.txt")
    with open(grouped_report_path, "w") as f:
        f.write(report_content)

    print(f"\nGrouped report saved to: {grouped_report_path}")
    print(report_content)


def compute_ex_metrics(
    input_csv_path: str,
    output_csv_path: str,
    db_base_path: str,
    report_path: str,
    timeout_seconds: int = 45,
):
    """
    Compute Execution Accuracy (EX) metrics for each row in the CSV.

    Args:
        input_csv_path: Path to input logs.csv
        output_csv_path: Path to output CSV with EX metrics
        db_base_path: Base path to database files
        report_path: Path to save the report text file
        timeout_seconds: Timeout in seconds for each EX computation (default: 45)
    """
    # Read the input CSV
    print(f"Reading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} rows")

    # Initialize columns for EX metrics
    df["ex_original"] = None  # EX for final_query_original vs org_SQL
    df["ex_expanded"] = None  # EX for final_query_expanded vs gold_query

    # Configure evaluation with Execution Accuracy
    config = {
        "evaluation_technique": EvaluationTechnique.EXECUTION_ACCURACY,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": None,  # Will be set per row
        },
        "penalize_extra_pred_cols": True,
        "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        "logs_dir_path": "data/evaluation_outputs/",
        "enable_log": False,
    }

    evaluator = Evaluation(config)

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing EX metrics"):
        db_id = row["db_id"]

        # Set database path for this row
        db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
        config["db_params"]["db_path"] = db_path

        # Check if database exists
        if not os.path.exists(db_path):
            print(f"Warning: Database not found for row {idx}: {db_path}")
            df.at[idx, "ex_original"] = None
            df.at[idx, "ex_expanded"] = None
            continue

        # Compute EX for original: final_query_original vs org_SQL
        try:
            final_query_original = row["final_query_original"]
            org_sql = row["org_SQL"]

            if pd.notna(final_query_original) and pd.notna(org_sql):
                try:
                    with time_limit(timeout_seconds):
                        result = evaluator.run_evaluation(
                            predicted_sql=str(final_query_original),
                            ground_truth_sql=str(org_sql),
                            log=False,
                        )
                        if result and "metrics" in result:
                            df.at[idx, "ex_original"] = result["metrics"].get(
                                "EX", None
                            )
                        else:
                            df.at[idx, "ex_original"] = 0
                except TimeoutException:
                    print(
                        f"Timeout computing ex_original for row {idx} (exceeded {timeout_seconds}s)"
                    )
                    df.at[idx, "ex_original"] = 0
            else:
                df.at[idx, "ex_original"] = None
        except Exception as e:
            print(f"Error computing ex_original for row {idx}: {e}")
            df.at[idx, "ex_original"] = 0

        # Compute EX for expanded: final_query_modified vs gold_query
        try:
            final_query_modified = row["final_query_modified"]
            gold_query = row["gold_query"]

            if pd.notna(final_query_modified) and pd.notna(gold_query):
                try:
                    with time_limit(timeout_seconds):
                        result = evaluator.run_evaluation(
                            predicted_sql=str(final_query_modified),
                            ground_truth_sql=str(gold_query),
                            log=False,
                        )
                        if result and "metrics" in result:
                            df.at[idx, "ex_expanded"] = result["metrics"].get(
                                "EX", None
                            )
                        else:
                            df.at[idx, "ex_expanded"] = 0
                except TimeoutException:
                    print(
                        f"Timeout computing ex_expanded for row {idx} (exceeded {timeout_seconds}s)"
                    )
                    df.at[idx, "ex_expanded"] = 0
            else:
                df.at[idx, "ex_expanded"] = None
        except Exception as e:
            print(f"Error computing ex_expanded for row {idx}: {e}")
            df.at[idx, "ex_expanded"] = 0

    # Calculate delta EX (from ex_expanded to ex_original)
    df["delta_ex"] = df["ex_original"] - df["ex_expanded"]

    # Save results
    print(f"\nSaving results to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    print("Done!")

    # Generate comprehensive report
    generate_report(df, report_path)

    # Generate grouped reports by n_join
    generate_grouped_reports(df, report_path)


if __name__ == "__main__":
    # Set paths
    input_csv = (
        "data/augmentation/jqe/dinsql_polluted_context/output_408_queries/logs.csv"
    )
    output_csv = "data/augmentation/jqe/dinsql_polluted_context/output_408_queries/logs_with_ex.csv"
    report_txt = (
        "data/augmentation/jqe/dinsql_polluted_context/output_408_queries/ex_report.txt"
    )
    db_base_path = "data/benchmarks/Bird/dev_databases"

    # Timeout for each EX computation (in seconds)
    timeout = 45

    # Compute metrics
    compute_ex_metrics(input_csv, output_csv, db_base_path, report_txt, timeout)

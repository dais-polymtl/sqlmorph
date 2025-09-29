import pandas as pd
import json
import numpy as np


def parse_metric_dict(metric_str):
    """Parse the metric string to extract values"""
    if pd.isna(metric_str) or metric_str == "" or metric_str == "None":
        return None
    try:
        # Handle the case where the string might have extra quotes or formatting
        if isinstance(metric_str, str):
            metric_dict = json.loads(metric_str)
            return metric_dict
        return metric_str
    except (json.JSONDecodeError, TypeError):
        return None


def compute_average_metrics(
    system_data, system_name, techniques, is_common_failure=False
):
    """Compute average metrics for a given system data subset"""
    prefix = "common_failure_" if is_common_failure else ""
    result_row = {"system": f"{prefix}{system_name}"}

    # 1. Compute EXECUTION_ACCURACY average
    exec_accuracy_values = []
    for _, row in system_data.iterrows():
        exec_metric = parse_metric_dict(row["EXECUTION_ACCURACY"])
        if exec_metric and "EX" in exec_metric:
            exec_accuracy_values.append(exec_metric["EX"])

    if exec_accuracy_values:
        ex_avg = round(sum(exec_accuracy_values) / len(exec_accuracy_values), 4)
    else:
        ex_avg = 0.0

    result_row["EXECUTION_ACCURACY"] = json.dumps({"EX": ex_avg})

    # 2. Compute aggregated metrics for the specified techniques
    for technique in techniques:
        # Initialize accumulators
        ex_values = []
        exp_values = []
        exr_values = []
        f1_values = []
        latency_values = []

        for _, row in system_data.iterrows():
            metric = parse_metric_dict(row[technique])
            if metric:
                if "EX" in metric:
                    ex_values.append(metric["EX"])
                if "EXP" in metric:
                    exp_values.append(metric["EXP"])
                if "EXR" in metric:
                    exr_values.append(metric["EXR"])
                if "F1" in metric:
                    f1_values.append(metric["F1"])
                if "latency" in metric:
                    latency_values.append(metric["latency"])

        # Calculate averages with 4 decimal points
        agg_metrics = {}
        if ex_values:
            agg_metrics["EX"] = round(sum(ex_values) / len(ex_values), 4)
        if exp_values:
            agg_metrics["EXP"] = round(sum(exp_values) / len(exp_values), 4)
        if exr_values:
            agg_metrics["EXR"] = round(sum(exr_values) / len(exr_values), 4)
        if f1_values:
            agg_metrics["F1"] = round(sum(f1_values) / len(f1_values), 4)
        if latency_values:
            agg_metrics["latency_avg"] = round(
                sum(latency_values) / len(latency_values), 4
            )
            agg_metrics["latency_std"] = (
                round(np.std(latency_values, ddof=1), 4)
                if len(latency_values) > 1
                else 0.0
            )

        result_row[technique] = json.dumps(agg_metrics)

    return result_row


def identify_common_failures(df, systems):
    """Identify question_ids where all systems have EXECUTION_ACCURACY_EX = 0"""
    question_ids = df["question_id"].unique()
    common_failure_question_ids = []

    for question_id in question_ids:
        all_failed = True

        for system in systems:
            system_question_data = df[
                (df["system"] == system) & (df["question_id"] == question_id)
            ]

            # Skip if we don't have data for this system and question
            if len(system_question_data) == 0:
                all_failed = False
                break

            # Check if this system failed on this question
            row = system_question_data.iloc[0]
            exec_metric = parse_metric_dict(row["EXECUTION_ACCURACY"])
            if not exec_metric or "EX" not in exec_metric or exec_metric["EX"] != 0:
                all_failed = False
                break

        if all_failed:
            common_failure_question_ids.append(question_id)

    return common_failure_question_ids


def compute_system_metrics(input_path, output_path, techniques):
    # Read the CSV file
    df = pd.read_csv(input_path)

    print(f"Loaded CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique systems: {df['system'].unique()}")

    results = []
    systems = df["system"].unique()

    # STEP 1: Compute overall metrics for each system
    for system in systems:
        system_data = df[df["system"] == system]
        result_row = compute_average_metrics(system_data, system, techniques)
        results.append(result_row)
        print(f"Processed overall metrics for system: {system}")

    # STEP 2: Compute individual system failure metrics (where EX=0 for each system)
    for system in systems:
        system_data = df[df["system"] == system]

        # Find question_ids where this system has EX=0
        failure_question_ids = []
        for _, row in system_data.iterrows():
            exec_metric = parse_metric_dict(row["EXECUTION_ACCURACY"])
            if exec_metric and "EX" in exec_metric and exec_metric["EX"] == 0:
                failure_question_ids.append(row["question_id"])

        if failure_question_ids:
            system_failure_data = df[
                (df["system"] == system)
                & (df["question_id"].isin(failure_question_ids))
            ]

            failure_result_row = compute_average_metrics(
                system_failure_data,
                f"{system}_individual_failures",
                techniques,
                is_common_failure=False,
            )
            results.append(failure_result_row)
            print(
                f"Processed individual failure metrics for system: {system} ({len(failure_question_ids)} failures)"
            )

    # STEP 3: Identify common failure question_ids
    common_failure_question_ids = identify_common_failures(df, systems)
    print(f"\nFound {len(common_failure_question_ids)} common failure question IDs")

    # STEP 4: Compute metrics for common failure cases
    if common_failure_question_ids:
        for system in systems:
            system_failure_data = df[
                (df["system"] == system)
                & (df["question_id"].isin(common_failure_question_ids))
            ]

            if len(system_failure_data) > 0:
                failure_result_row = compute_average_metrics(
                    system_failure_data, system, techniques, is_common_failure=True
                )
                results.append(failure_result_row)
                print(f"Processed common failure metrics for system: {system}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print("\nSample of results:")
    print(results_df)

    return results_df


if __name__ == "__main__":
    # CONFIGURABLE PARAMETERS
    INPUT_PATH = "data/metrics/experiments/system_level_comparison/systems_data_with_metrics-with_penalty.csv"
    OUTPUT_PATH = (
        "data/metrics/experiments/system_level_comparison/system_metrics_avg_report.csv"
    )

    # Techniques to analyze
    techniques = [
        "EXACT_COLUMN_AND_EXACT_CELL",
        "EXACT_COLUMN_AND_PARTIAL_CELL",
        "SEMANTIC_COLUMN_AND_EXACT_CELL",
        "SEMANTIC_COLUMN_AND_PARTIAL_CELL",
        "NO_COLUMN_AND_PARTIAL_CELL",
        # "UNIFIED_COLUMN_AND_SEMANTIC_ROW",
    ]

    compute_system_metrics(INPUT_PATH, OUTPUT_PATH, techniques)

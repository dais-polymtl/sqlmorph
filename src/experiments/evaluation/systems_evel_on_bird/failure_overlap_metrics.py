import pandas as pd
import json


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


def compute_system_metrics(output_path):
    # Read the CSV file
    csv_path = "/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage/data/evaluation/experiments/systems_evel_on_bird/systems_data_with_metrics.csv"
    df = pd.read_csv(csv_path)

    print(f"Loaded CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique systems: {df['system'].unique()}")

    results = []

    # Group by system
    for system in df["system"].unique():
        system_data = df[df["system"] == system]
        result_row = {"system": system}

        # 1. Compute EXECUTION_ACCURACY average EX
        exec_accuracy_values = []
        for _, row in system_data.iterrows():
            exec_metric = parse_metric_dict(row["EXECUTION_ACCURACY"])
            if exec_metric and "EX" in exec_metric:
                exec_accuracy_values.append(exec_metric["EX"])

        if exec_accuracy_values:
            result_row["EXECUTION_ACCURACY_EX_avg"] = round(
                (sum(exec_accuracy_values) / len(exec_accuracy_values)) * 100, 2
            )
        else:
            result_row["EXECUTION_ACCURACY_EX_avg"] = 0.0

        # 2. Compute metrics for the three techniques
        techniques = [
            "EXACT_COLUMN_AND_EXACT_CELL",
            "SEMANTIC_COLUMN_AND_EXACT_CELL",
            "UNIFIED_COLUMN_AND_SEMANTIC_ROW",
        ]

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

            # Calculate averages and multiply by 100 with 2 decimal places
            result_row[f"{technique}_EX_avg"] = (
                round((sum(ex_values) / len(ex_values)) * 100, 2) if ex_values else 0.0
            )
            result_row[f"{technique}_EXP_avg"] = (
                round((sum(exp_values) / len(exp_values)) * 100, 2)
                if exp_values
                else 0.0
            )
            result_row[f"{technique}_EXR_avg"] = (
                round((sum(exr_values) / len(exr_values)) * 100, 2)
                if exr_values
                else 0.0
            )
            result_row[f"{technique}_F1_avg"] = (
                round((sum(f1_values) / len(f1_values)) * 100, 2) if f1_values else 0.0
            )
            result_row[f"{technique}_latency_avg"] = (
                round((sum(latency_values) / len(latency_values)) * 100, 2)
                if latency_values
                else 0.0
            )

        results.append(result_row)
        print(f"Processed system: {system}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print("\nSample of results:")
    print(results_df.head())

    return results_df


if __name__ == "__main__":
    OUTPUT_PATH = "/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage/data/evaluation/experiments/systems_evel_on_bird/system_metrics_report.csv"
    compute_system_metrics(OUTPUT_PATH)

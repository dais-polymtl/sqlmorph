import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os


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
    exec_latency_values = []
    for _, row in system_data.iterrows():
        exec_metric = parse_metric_dict(row["EXECUTION_ACCURACY"])
        if exec_metric and "EX" in exec_metric:
            exec_accuracy_values.append(exec_metric["EX"])
            if "latency" in exec_metric:
                exec_latency_values.append(exec_metric["latency"])

    if exec_accuracy_values:
        ex_avg = round((sum(exec_accuracy_values) / len(exec_accuracy_values)) * 100, 1)
    else:
        ex_avg = 0.0

    exec_result = {"EX": ex_avg, "delta": round(100 - ex_avg, 1)}
    if exec_latency_values:
        exec_result["latency_avg"] = round(
            sum(exec_latency_values) / len(exec_latency_values), 4
        )
        exec_result["latency_std"] = (
            round(np.std(exec_latency_values, ddof=1), 4)
            if len(exec_latency_values) > 1
            else 0.0
        )

    result_row["EXECUTION_ACCURACY"] = json.dumps(exec_result)

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

        # Calculate averages with percentages (1 decimal point) and delta metrics
        agg_metrics = {}
        if ex_values:
            ex_avg = round((sum(ex_values) / len(ex_values)) * 100, 1)
            agg_metrics["EX"] = ex_avg
            agg_metrics["EX_delta"] = round(100 - ex_avg, 1)
        if exp_values:
            exp_avg = round((sum(exp_values) / len(exp_values)) * 100, 1)
            agg_metrics["EXP"] = exp_avg
            agg_metrics["EXP_delta"] = round(100 - exp_avg, 1)
        if exr_values:
            exr_avg = round((sum(exr_values) / len(exr_values)) * 100, 1)
            agg_metrics["EXR"] = exr_avg
            agg_metrics["EXR_delta"] = round(100 - exr_avg, 1)
        if f1_values:
            f1_avg = round((sum(f1_values) / len(f1_values)) * 100, 1)
            agg_metrics["F1"] = f1_avg
            agg_metrics["F1_delta"] = round(100 - f1_avg, 1)
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


def extract_metrics_for_plotting(df, techniques):
    """Extract EXP, EXR, F1 metrics from each technique for plotting"""
    plot_data = []

    for _, row in df.iterrows():
        system = row["system"]

        for technique in techniques:
            metric_dict = parse_metric_dict(row[technique])
            if metric_dict:
                for metric in ["EXP", "EXR", "F1"]:
                    if metric in metric_dict:
                        plot_data.append(
                            {
                                "system": system,
                                "technique": technique,
                                "metric": metric,
                                "value": float(metric_dict[metric]),
                            }
                        )

    return pd.DataFrame(plot_data)


def create_box_plots(
    input_path,
    plots_dir,
    techniques,
    plot_tag,
    common_failures_only=False,
    common_failure_ids=None,
):
    """Create box plots for EXP, EXR, and F1 metrics across systems and techniques

    Args:
        input_path: Path to input CSV file
        plots_dir: Directory to save the plots
        techniques: List of techniques to analyze
        plot_tag: Tag to include in the filename (e.g., "with_penalty")
        common_failures_only: If True, plot only common failure cases
        common_failure_ids: List of question IDs that are common failures (optional, will be computed if not provided)
    """
    plot_type = "common_failures" if common_failures_only else "all_data"
    print(f"Creating {plot_type} box plots from {input_path}")

    # Exclude "EXECUTION_ACCURACY" from techniques for plotting
    plot_techniques = [t for t in techniques if t != "EXECUTION_ACCURACY"]

    # Read the data
    df = pd.read_csv(input_path)

    # Compute common failure IDs if not provided and needed
    if common_failures_only and common_failure_ids is None:
        systems = df["system"].unique()
        common_failure_ids = identify_common_failures(df, systems)
        print(f"Computed {len(common_failure_ids)} common failure question IDs")

    # Filter for common failures if requested
    if common_failures_only and common_failure_ids:
        df = df[df["question_id"].isin(common_failure_ids)]
        if len(df) == 0:
            print("No common failure data to plot")
            return

    # Extract metrics for plotting
    plot_df = extract_metrics_for_plotting(df, plot_techniques)

    if len(plot_df) == 0:
        print("No data available for plotting")
        return

    # Get unique systems
    systems = sorted(plot_df["system"].unique())

    # Create subplot grid with three rows (one per system)
    fig, axes = plt.subplots(
        nrows=len(systems),
        ncols=len(plot_techniques),
        figsize=(len(plot_techniques) * 3.5, len(systems) * 3),
        squeeze=False,
    )

    # Define colors for metrics
    colors = {"EXP": "lightblue", "EXR": "lightgreen", "F1": "salmon"}

    for i, system in enumerate(systems):
        for j, technique in enumerate(plot_techniques):
            ax = axes[i, j]

            # Filter data for this system and technique
            system_data = plot_df[
                (plot_df["system"] == system) & (plot_df["technique"] == technique)
            ]

            # Create empty lists for each metric
            exp_values = system_data[system_data["metric"] == "EXP"]["value"].tolist()
            exr_values = system_data[system_data["metric"] == "EXR"]["value"].tolist()
            f1_values = system_data[system_data["metric"] == "F1"]["value"].tolist()

            # Create box plot
            box_data = [exp_values, exr_values, f1_values]
            box = ax.boxplot(
                box_data,
                patch_artist=True,
                tick_labels=["EXP", "EXR", "F1"],
                medianprops=dict(color="red", linewidth=3.5),
            )

            # Color the boxes
            for patch, color in zip(box["boxes"], colors.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Set titles and labels
            if i == 0:  # Only set title for top row
                technique_name = technique.replace("_", " ").replace("AND", "&")
                ax.set_title(technique_name, fontsize=9)

            if j == 0:  # Only set y-label for leftmost column
                ax.set_ylabel(f"{system}", fontsize=10, fontweight="bold")

            # Set y-axis limits
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # # Add overall title based on the type of plot
    # title = "Metrics for Common Failure Cases" if common_failures_only else "Metrics for All Data"
    # fig.suptitle(title, fontsize=14, y=0.98)

    # Adjust layout
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the suptitle

    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Use shorter filename with plot_tag instead of the full input filename
    file_prefix = "common_failure_" if common_failures_only else "all_data_"
    output_path = os.path.join(
        plots_dir, f"{file_prefix}metrics_{plot_tag}_box_plot.pdf"
    )

    # Save plot
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Box plots saved to: {output_path}")
    plt.close()


def create_individual_failure_box_plots(input_path, plots_dir, techniques, plot_tag):
    """Create box plots for each system's individual failure cases in a single PDF"""
    print(f"Creating individual failure box plots from {input_path}")

    # Exclude "EXECUTION_ACCURACY" from techniques for plotting
    plot_techniques = [t for t in techniques if t != "EXECUTION_ACCURACY"]

    # Read the data
    df = pd.read_csv(input_path)

    # Get unique systems (excluding any aggregate rows)
    all_systems = df["system"].unique()
    base_systems = [
        s
        for s in all_systems
        if not s.startswith("common_failure_") and "_individual_failures" not in s
    ]

    # Collect failure data for all systems
    systems_failure_data = []

    for system in base_systems:
        # Find question_ids where this system has EX=0
        system_data = df[df["system"] == system]
        failure_question_ids = []

        for _, row in system_data.iterrows():
            exec_metric = parse_metric_dict(row["EXECUTION_ACCURACY"])
            if exec_metric and "EX" in exec_metric and exec_metric["EX"] == 0:
                failure_question_ids.append(row["question_id"])

        if failure_question_ids:
            # Filter data for this system's failures
            failure_data = df[
                (df["system"] == system)
                & (df["question_id"].isin(failure_question_ids))
            ]

            # Extract metrics for plotting
            plot_df = extract_metrics_for_plotting(failure_data, plot_techniques)

            if len(plot_df) > 0:
                systems_failure_data.append(
                    {
                        "system": system,
                        "failure_count": len(failure_question_ids),
                        "plot_df": plot_df,
                    }
                )
                print(f"Found {len(failure_question_ids)} failure cases for {system}")

    if not systems_failure_data:
        print("No individual failure data to plot")
        return

    # Create subplot grid with rows for each system
    fig, axes = plt.subplots(
        nrows=len(systems_failure_data),
        ncols=len(plot_techniques),
        figsize=(len(plot_techniques) * 3.5, len(systems_failure_data) * 3),
        squeeze=False,
    )

    # Define colors for metrics
    colors = {"EXP": "lightblue", "EXR": "lightgreen", "F1": "salmon"}

    for i, system_data in enumerate(systems_failure_data):
        system = system_data["system"]
        # failure_count = system_data["failure_count"]
        plot_df = system_data["plot_df"]

        for j, technique in enumerate(plot_techniques):
            ax = axes[i, j]

            # Filter data for this technique
            technique_data = plot_df[plot_df["technique"] == technique]

            # Create empty lists for each metric
            exp_values = technique_data[technique_data["metric"] == "EXP"][
                "value"
            ].tolist()
            exr_values = technique_data[technique_data["metric"] == "EXR"][
                "value"
            ].tolist()
            f1_values = technique_data[technique_data["metric"] == "F1"][
                "value"
            ].tolist()

            # Create box plot
            box_data = [exp_values, exr_values, f1_values]
            box = ax.boxplot(
                box_data,
                patch_artist=True,
                tick_labels=["EXP", "EXR", "F1"],
                medianprops=dict(color="red", linewidth=3.5),
            )

            # Color the boxes
            for patch, color in zip(box["boxes"], colors.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Set titles and labels
            if i == 0:  # Only set title for top row
                technique_name = technique.replace("_", " ").replace("AND", "&")
                ax.set_title(technique_name, fontsize=9)

            if j == 0:  # Only set y-label for leftmost column
                ax.set_ylabel(f"{system}", fontsize=10, fontweight="bold")

            # Set y-axis limits
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Save plot with single filename
    output_path = os.path.join(
        plots_dir, f"individual_failure_metrics_{plot_tag}_box_plot.pdf"
    )

    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Individual failure box plots saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # CONFIGURABLE PARAMETERS
    TAG = "ex_new_nl_original_sql_results"
    ROOT = "data/augmentation/decrease_naturalness/experiments_dev_all/"
    INPUT_PATH = f"{ROOT}/{TAG}.csv"
    OUTPUT_PATH = f"{ROOT}/system_metrics_avg_report-{TAG}.csv"

    PLOTS_DIR = f"{ROOT}/plots-{TAG}"
    # INPUT_PATH = f"data/augmentation/decrease_naturalness/experiments/{TAG}.csv"
    # OUTPUT_PATH = (
    #     f"data/augmentation/decrease_naturalness/experiments/system_metrics_avg_report-{TAG}.csv"
    # )
    #
    # PLOTS_DIR = f"data/augmentation/decrease_naturalness/experiments/plots-{TAG}"

    # Techniques to analyze
    techniques = [
        "EXECUTION_ACCURACY",
        # "EXACT_COLUMN_AND_EXACT_CELL",
        # "EXACT_COLUMN_AND_PARTIAL_CELL",
        # "SEMANTIC_COLUMN_AND_EXACT_CELL",
        # "SEMANTIC_COLUMN_AND_PARTIAL_CELL",
        # "NO_COLUMN_AND_PARTIAL_CELL",
        # "UNIFIED_COLUMN_AND_SEMANTIC_ROW",
    ]

    # Compute system metrics
    results_df = compute_system_metrics(INPUT_PATH, OUTPUT_PATH, techniques)

    # Create and save box plots for all data
    create_box_plots(INPUT_PATH, PLOTS_DIR, techniques, TAG, common_failures_only=False)

    # Create and save box plots for common failure cases only
    create_box_plots(INPUT_PATH, PLOTS_DIR, techniques, TAG, common_failures_only=True)

    # Create and save box plots for individual system failures
    create_individual_failure_box_plots(INPUT_PATH, PLOTS_DIR, techniques, TAG)

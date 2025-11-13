import pandas as pd
import json
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
import time

from src.core.database.database_handler import DBMS
from src.core.model_manager import OpenAIModel
from src.metrics import Evaluation, EvaluationTechnique


def calculate_metrics(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    technique: EvaluationTechnique,
    log_dir: str,
    embedding_model,
    penalize_extra_pred_cols: bool,
) -> dict:
    config = {
        "evaluation_technique": technique,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(db_path),
        },
        "penalize_extra_pred_cols": penalize_extra_pred_cols,
        "embedding_model": embedding_model,
        "logs_dir_path": log_dir,
    }
    evaluator = Evaluation(config)
    res = evaluator.run_evaluation(
        predicted_sql=pred_sql,
        ground_truth_sql=gold_sql,
        log=True,
    )

    # Combine metrics and latency into a single dictionary
    result = res["metrics"].copy()
    result["latency"] = res["latency"]
    return result


def evaluate_single_row(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
    technique: EvaluationTechnique,
    log_dir: str,
    embedding_model,
    penalize_extra_pred_cols: bool,
    timeout_seconds: int = 120,
) -> str:
    """Evaluate a single row with func_timeout for reliable timeout handling"""

    try:
        result = func_timeout(
            timeout_seconds,
            calculate_metrics,
            args=(
                predicted_sql,
                gold_sql,
                db_path,
                technique,
                log_dir,
                embedding_model,
                penalize_extra_pred_cols,
            ),
        )
        return result
    except FunctionTimedOut:
        print(f"Timeout occurred for db: {db_path} with technique: {technique.name}")
        return "timeout"
    except Exception as e:
        print(f"Error for db {db_path} with technique {technique.name}: {str(e)}")
        return "error"


def evaluate_with_techniques(
    csv_file_path: str,
    databases_dir: str,
    log_dir: str,
    techniques: list,
    embedding_model,
    penalize_extra_pred_cols: bool,
    output_csv_path: str = None,
    timeout_seconds: int = 60,
    sampling_ratio=None,
    random_seed=None,
) -> pd.DataFrame:
    if output_csv_path is None:
        output_csv_path = csv_file_path.replace(".csv", "_with_metrics.csv")

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Validate required columns
    required_columns = [
        "system",
        "db_name",
        "question_id",
        "question",
        "gold_sql",
        "predicted_sql",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Apply sampling if specified
    if sampling_ratio is not None:
        df = sample_dataframe(df, sampling_ratio, random_seed)
        print(f"Sampled to {len(df['question_id'].unique())} unique questions")

    # Check if output file already exists and load already evaluated rows
    evaluated_df = pd.DataFrame()
    if pd.io.common.file_exists(output_csv_path):
        evaluated_df = pd.read_csv(output_csv_path)
        print(f"Found existing output file with {len(evaluated_df)} rows")

        # Create unique identifier for comparison
        df["unique_id"] = (
            df["question_id"].astype(str)
            + "_"
            + df["system"]
            + "_"
            + df["gold_sql"]
            + "_"
            + df["predicted_sql"]
        )
        evaluated_df["unique_id"] = (
            evaluated_df["question_id"].astype(str)
            + "_"
            + evaluated_df["system"]
            + "_"
            + evaluated_df["gold_sql"]
            + "_"
            + evaluated_df["predicted_sql"]
        )

        # Filter out already evaluated rows
        already_evaluated_ids = set(evaluated_df["unique_id"])
        df_to_evaluate = df[~df["unique_id"].isin(already_evaluated_ids)].copy()

        print(f"Skipping {len(df) - len(df_to_evaluate)} already evaluated rows")
        print(f"Will evaluate {len(df_to_evaluate)} remaining rows")

        # Drop the unique_id column as it's only used for comparison
        df = df.drop("unique_id", axis=1)
        df_to_evaluate = df_to_evaluate.drop("unique_id", axis=1)
        evaluated_df = evaluated_df.drop("unique_id", axis=1)
    else:
        df_to_evaluate = df.copy()
        print(
            f"No existing output file found. Will evaluate all {len(df_to_evaluate)} rows"
        )

    # Initialize columns for each technique in the dataframe to evaluate
    for technique in techniques:
        technique_name = technique.name
        df_to_evaluate[technique_name] = None

    # Evaluate each row
    total_rows = len(df_to_evaluate)
    for idx, row in df_to_evaluate.iterrows():
        print(
            f"Evaluating row {idx + 1}/{total_rows} - DB: {row['db_name']}, System: {row['system']}, Question ID: {row['question_id']}"
        )

        db_path = f"{databases_dir}/{row['db_name']}/{row['db_name']}.sqlite"
        print(db_path)

        # Evaluate each technique independently
        # row_has_any_results = False

        for technique in techniques:
            technique_name = technique.name
            print(f"  Evaluating with {technique_name}...")

            start_time = time.time()

            result = evaluate_single_row(
                predicted_sql=row["predicted_sql"],
                gold_sql=row["gold_sql"],
                db_path=db_path,
                technique=technique,
                log_dir=log_dir,
                embedding_model=embedding_model,
                penalize_extra_pred_cols=penalize_extra_pred_cols,
                timeout_seconds=timeout_seconds,
            )

            elapsed_time = time.time() - start_time
            print(f"    {technique_name} completed in {elapsed_time:.2f}s")

            # Store result for this technique
            if result == "timeout":
                df_to_evaluate.at[idx, technique_name] = "timeout"
                print(f"    {technique_name}: TIMEOUT")
            elif result == "error":
                df_to_evaluate.at[idx, technique_name] = "error"
                print(f"    {technique_name}: ERROR")
            else:
                # Successful evaluation with metrics
                df_to_evaluate.at[idx, technique_name] = json.dumps(result)
                main_metric = list(result.values())[0] if result else None
                print(f"    {technique_name} Result: {main_metric}")
                # row_has_any_results = True

        # Save row regardless of success/failure status
        # Create a single row dataframe for this evaluated row
        evaluated_row = df_to_evaluate.loc[[idx]]

        # Append to existing results
        if len(evaluated_df) > 0:
            evaluated_df = pd.concat([evaluated_df, evaluated_row], ignore_index=True)
        else:
            evaluated_df = evaluated_row.copy()

        # Save progress after each row
        evaluated_df.to_csv(output_csv_path, index=False)
        print(f"Row {idx + 1} evaluated and saved")

    print(f"Evaluation complete. Results saved to {output_csv_path}")

    # Print summary statistics for each technique
    print("\nSummary:")
    total_evaluated = len(evaluated_df)
    print(f"Total rows evaluated: {total_evaluated}")

    for technique in techniques:
        technique_name = technique.name
        if len(evaluated_df) > 0:
            success_count = 0
            timeout_count = 0
            error_count = 0

            for value in evaluated_df[technique_name]:
                if pd.isna(value):
                    continue
                elif value == "timeout":
                    timeout_count += 1
                elif value == "error":
                    error_count += 1
                else:
                    success_count += 1

            success_percentage = (
                success_count / total_evaluated * 100 if total_evaluated > 0 else 0
            )
            timeout_percentage = (
                timeout_count / total_evaluated * 100 if total_evaluated > 0 else 0
            )
            error_percentage = (
                error_count / total_evaluated * 100 if total_evaluated > 0 else 0
            )

            print(
                f"{technique_name}: {success_count} successful ({success_percentage:.2f}%), "
                f"{timeout_count} timeouts ({timeout_percentage:.2f}%), "
                f"{error_count} errors ({error_percentage:.2f}%)"
            )

    return evaluated_df


def sample_dataframe(
    df: pd.DataFrame, sampling_ratio, random_seed=None
) -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)

    unique_question_ids = df["question_id"].unique()

    if isinstance(sampling_ratio, int):
        if sampling_ratio >= len(unique_question_ids):
            return df
        sampled_question_ids = np.random.choice(
            unique_question_ids, size=sampling_ratio, replace=False
        )
    elif isinstance(sampling_ratio, float) and 0 < sampling_ratio <= 1:
        sample_size = int(len(unique_question_ids) * sampling_ratio)
        if sample_size == 0:
            sample_size = 1
        sampled_question_ids = np.random.choice(
            unique_question_ids, size=sample_size, replace=False
        )
    else:
        raise ValueError(
            "sampling_ratio must be a positive integer or a float between 0 and 1"
        )

    # Filter to keep all systems for the selected question IDs
    sampled_df = df[df["question_id"].isin(sampled_question_ids)]

    print(f"Selected {len(sampled_question_ids)} question IDs")
    print(f"This results in {len(sampled_df)} total rows across all systems")

    return sampled_df


if __name__ == "__main__":
    # CONFIGURABLE PARAMETERS
    experiment_name = "systems_data_with_metrics-without_penalty"

    ROOT = "data/metrics/experiments/system_level_comparison"
    input_csv_path = ROOT + "/systems_data.csv"
    output_csv_path = ROOT + f"/{experiment_name}.csv"
    log_dir = ROOT + f"/logs/{experiment_name}/"

    databases_dir = "data/benchmarks/Bird/dev_databases"

    techniques = [
        EvaluationTechnique.EXECUTION_ACCURACY,
        EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.EXACT_COLUMN_AND_PARTIAL_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_PARTIAL_CELL,
        EvaluationTechnique.NO_COLUMN_AND_PARTIAL_CELL,
    ]

    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL

    timeout_seconds = 120  # Timeout for each single evaluation
    sampling_ratio = 1.0  # if int: pick exactly that many random question IDs, if float (0-1): pick that percentage of unique question IDs
    random_seed = 42  # Random seed for reproducible sampling
    penalize_extra_pred_cols = False  # Whether to penalize extra columns in predictions

    result_df = evaluate_with_techniques(
        csv_file_path=input_csv_path,
        databases_dir=databases_dir,
        log_dir=log_dir,
        techniques=techniques,
        embedding_model=embedding_model,
        penalize_extra_pred_cols=penalize_extra_pred_cols,
        output_csv_path=output_csv_path,
        timeout_seconds=timeout_seconds,
        sampling_ratio=sampling_ratio,
        random_seed=random_seed,
    )

    print("Done!")

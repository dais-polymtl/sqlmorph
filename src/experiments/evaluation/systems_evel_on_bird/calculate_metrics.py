import pandas as pd
import signal
import json
from contextlib import contextmanager
import numpy as np

from src.core.database.database_handler import DBMS
from src.core.model_manager import OpenAIModel
from src.evaluation import Evaluation, EvaluationTechnique


def calculate_metrics(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    technique: EvaluationTechnique,
    log_dir: str,
    embedding_model,
) -> dict:
    config = {
        "evaluation_technique": technique,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(db_path),
        },
        "embedding_model": embedding_model,
        "logs_dir_path": log_dir,
    }
    evaluator = Evaluation(config)
    res = evaluator.run_evaluation(
        predicted_sql=pred_sql,
        ground_truth_sql=gold_sql,
        log=True,
    )
    return res["metrics"]


class TimeoutException(Exception):
    pass


@contextmanager
def timeout_handler(seconds):
    def timeout_signal_handler(signum, frame):
        raise TimeoutException("Evaluation timed out")

    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def evaluate_single_row(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
    technique: EvaluationTechnique,
    log_dir: str,
    embedding_model,
    timeout_seconds: int = 60,
) -> dict:
    try:
        with timeout_handler(timeout_seconds):
            return calculate_metrics(
                predicted_sql, gold_sql, db_path, technique, log_dir, embedding_model
            )
    except TimeoutException:
        print(f"Timeout occurred for db: {db_path}")
        return None
    except Exception as e:
        print(f"Error evaluating db {db_path}: {str(e)}")
        return None


def evaluate_with_techniques(
    csv_file_path: str,
    databases_dir: str,
    log_dir: str,
    techniques: list,
    embedding_model,
    output_csv_path: str = None,
    timeout_seconds: int = 60,
    sampling_ratio=None,
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

    # Initialize columns for each technique
    for technique in techniques:
        technique_name = technique.name
        df[technique_name] = None

    # Apply sampling if specified
    if sampling_ratio is not None:
        df = sample_dataframe(df, sampling_ratio)
        print(f"Sampled to {len(df['question_id'].unique())} unique questions")

    # Evaluate each row
    total_rows = len(df)
    for idx, row in df.iterrows():
        print(
            f"Evaluating row {idx + 1}/{total_rows} - DB: {row['db_name']}, System: {row['system']}, Question ID: {row['question_id']}"
        )

        db_path = f"{databases_dir}/{row['db_name']}/{row['db_name']}.sqlite"
        print(db_path)

        for technique in techniques:
            technique_name = technique.name
            print(f"  Evaluating with {technique_name}...")

            metrics = evaluate_single_row(
                predicted_sql=row["predicted_sql"],
                gold_sql=row["gold_sql"],
                db_path=db_path,
                technique=technique,
                log_dir=log_dir,
                embedding_model=embedding_model,
                timeout_seconds=timeout_seconds,
            )

            # Store metrics as JSON string for CSV compatibility
            if metrics is not None:
                df.at[idx, technique_name] = json.dumps(metrics)
            else:
                df.at[idx, technique_name] = None

            # Log the result
            if metrics:
                main_metric = list(metrics.values())[0] if metrics else None
                print(f"    {technique_name} Result: {main_metric}")
            else:
                print(f"    {technique_name}: FAILED")

        # Save progress after each row
        df.to_csv(output_csv_path, index=False)
        print(f"Progress saved at row {idx + 1}")

    print(f"Evaluation complete. Results saved to {output_csv_path}")

    # Print summary statistics for each technique
    print("\nSummary:")
    print(f"Total rows: {total_rows}")

    for technique in techniques:
        technique_name = technique.name
        success_count = len(df[df[technique_name].notna()])
        success_percentage = success_count / total_rows * 100
        print(
            f"{technique_name}: {success_count} successful evaluations ({success_percentage:.2f}%)"
        )

    return df


def sample_dataframe(df: pd.DataFrame, sampling_ratio) -> pd.DataFrame:
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
    experiment_id = "ex1"

    ROOT = "data/evaluation/experiments/systems_evel_on_bird"
    input_csv_path = ROOT + "/systems_data.csv"
    output_csv_path = ROOT + f"/systems_data_with_metrics_{experiment_id}.csv"
    log_dir = ROOT + f"/logs/{experiment_id}"

    databases_dir = "data/benchmarks/Bird/dev_databases"

    techniques = [
        EvaluationTechnique.EXECUTION_ACCURACY,
        EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW,
    ]

    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL

    timeout_seconds = 60  # Timeout for each single evaluation
    sampling_ratio = 100  # if int: pick exactly that many random question IDs, if float (0-1): pick that percentage of unique question IDs

    result_df = evaluate_with_techniques(
        csv_file_path=input_csv_path,
        databases_dir=databases_dir,
        log_dir=log_dir,
        techniques=techniques,
        embedding_model=embedding_model,
        output_csv_path=output_csv_path,
        timeout_seconds=timeout_seconds,
        sampling_ratio=sampling_ratio,
    )

    print("Done!")

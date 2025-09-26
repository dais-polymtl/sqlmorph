import pandas as pd
import json
from func_timeout import func_timeout, FunctionTimedOut
import time
import os

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
    penalize_extra_columns: bool,
) -> dict:
    """Calculate evaluation metrics for a predicted SQL against gold SQL"""
    config = {
        "evaluation_technique": technique,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(db_path),
        },
        "penalize_extra_columns": penalize_extra_columns,
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


def evaluate_single_mutant(
    mutant_sql: str,
    gold_sql: str,
    db_path: str,
    technique: EvaluationTechnique,
    log_dir: str,
    penalize_extra_columns: bool,
    embedding_model,
    timeout_seconds: int = 120,
) -> str:
    """Evaluate a single mutant with func_timeout for reliable timeout handling"""
    try:
        result = func_timeout(
            timeout_seconds,
            calculate_metrics,
            args=(
                mutant_sql,
                gold_sql,
                db_path,
                technique,
                log_dir,
                embedding_model,
                penalize_extra_columns,
            ),
        )
        return result
    except FunctionTimedOut:
        print(f"Timeout occurred for technique: {technique.name}")
        return "timeout"
    except Exception as e:
        print(f"Error with technique {technique.name}: {str(e)}")
        return "error"


def evaluate_mutants(
    mutants_json_path: str,
    databases_dir: str,
    log_dir: str,
    techniques: list,
    penalize_extra_columns: bool,
    embedding_model,
    output_csv_path: str = None,
    timeout_seconds: int = 120,
) -> pd.DataFrame:
    """Evaluate all mutant queries using different evaluation techniques"""

    # Load mutants JSON
    with open(mutants_json_path, "r") as f:
        mutants_data = json.load(f)

    db_id = mutants_data["db_id"]
    gold_query = mutants_data["gold_query"]
    mutant_groups = mutants_data["mutant_queries"]

    # Set output path if not provided
    if output_csv_path is None:
        output_csv_path = mutants_json_path.replace(".json", "_evaluation_results.csv")

    # Create database path
    db_path = f"{databases_dir}/{db_id}/{db_id}.sqlite"
    print(f"Database path: {db_path}")

    # Create results list
    results = []

    # Process each mutant group
    for group_name, mutant_queries in mutant_groups.items():
        print(f"\nEvaluating group: {group_name}")
        print(f"Number of mutants in group: {len(mutant_queries)}")

        for mutant_idx, mutant_query in enumerate(mutant_queries):
            print(f"  Evaluating mutant {mutant_idx + 1}/{len(mutant_queries)}")

            # Create row for this mutant
            row_data = {
                "db_id": db_id,
                "group_name": group_name,
                "mutant_index": mutant_idx + 1,
                "gold_query": gold_query,
                "mutant_query": mutant_query,
            }

            # Evaluate with each technique
            for technique in techniques:
                technique_name = technique.name
                print(f"    Evaluating with {technique_name}...")

                start_time = time.time()

                result = evaluate_single_mutant(
                    mutant_sql=mutant_query,
                    gold_sql=gold_query,
                    db_path=db_path,
                    technique=technique,
                    log_dir=log_dir,
                    embedding_model=embedding_model,
                    penalize_extra_columns=penalize_extra_columns,
                    timeout_seconds=timeout_seconds,
                )

                elapsed_time = time.time() - start_time
                print(f"      {technique_name} completed in {elapsed_time:.2f}s")

                # Store result for this technique
                if result == "timeout":
                    row_data[technique_name] = "timeout"
                    print(f"      {technique_name}: TIMEOUT")
                elif result == "error":
                    row_data[technique_name] = "error"
                    print(f"      {technique_name}: ERROR")
                else:
                    # Successful evaluation with metrics
                    row_data[technique_name] = json.dumps(result)
                    main_metric = list(result.values())[0] if result else None
                    print(f"      {technique_name} Result: {main_metric}")

            results.append(row_data)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to: {output_csv_path}")

    return results_df


if __name__ == "__main__":
    # CONFIGURABLE PARAMETERS
    DATA = "data/metrics/experiments/table_shape_sensitivity"
    mutants_json_path = (
        "src/experiments/metrics/table_shape_sensitivity/mutants_v1.json"
    )

    experiment_name = "mutants_v1_evaluation_results_all_with_no_penalize-2025-09-26"
    output_csv_path = f"{DATA}/{experiment_name}.csv"
    log_dir = f"{DATA}/logs/{experiment_name}/"

    databases_dir = "data/benchmarks/Bird/dev_databases"

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    techniques = [
        EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.EXACT_COLUMN_AND_PARTIAL_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_PARTIAL_CELL,
        EvaluationTechnique.FREE_COLUMN_AND_PARTIAL_CELL,
        # EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW
    ]

    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL
    timeout_seconds = 120  # Timeout for each single evaluation
    penalize_extra_columns = False

    print("Starting mutant evaluation...")
    print(f"Mutants file: {mutants_json_path}")
    print(f"Output file: {output_csv_path}")
    print(f"Techniques: {[t.name for t in techniques]}")

    # Run evaluation
    result_df = evaluate_mutants(
        mutants_json_path=mutants_json_path,
        databases_dir=databases_dir,
        log_dir=log_dir,
        techniques=techniques,
        penalize_extra_columns=penalize_extra_columns,
        embedding_model=embedding_model,
        output_csv_path=output_csv_path,
        timeout_seconds=timeout_seconds,
    )

    print("\nEvaluation complete!")

import pandas as pd
import json
from pathlib import Path
from src.core.logger.logger import Logger

logger = Logger(__name__)


def evaluate_schema_linking_single(pred_schema, gold_schema):
    """
    Evaluate schema linking performance for a single query with case-insensitive matching.

    Args:
        pred_schema (dict): predicted schema dict for single query
        gold_schema (dict): golden/true schema dict for single query

    Returns:
        dict: Dictionary with evaluation metrics including:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN) - also known as Schema Linking Recall (SLR)
            - f1: harmonic mean of precision and recall
            - fpr: False Positive Rate - FP / (TP + FP)
            - slr: Schema Linking Recall (perfect retrieval) - 1 if FN == 0, else 0
            - tp, fp, fn: true positives, false positives, false negatives
    """
    # Flatten to table_column pairs with case-insensitive matching
    gold_table_cols = []
    for table in gold_schema.keys():
        cols = gold_schema[table]
        for col in cols:
            gold_table_cols.append(table.lower() + "_" + col.lower())

    pred_table_cols = []
    for table in pred_schema.keys():
        cols = pred_schema[table]
        for col in cols:
            pred_table_cols.append(table.lower() + "_" + col.lower())

    tp = len(set(gold_table_cols) & set(pred_table_cols))
    fp = len(set(pred_table_cols) - set(gold_table_cols))
    fn = len(set(gold_table_cols) - set(pred_table_cols))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # False Positive Rate: proportion of irrelevant columns retrieved over total retrieved
    fpr = fp / (tp + fp) if (tp + fp) else 0

    # Schema Linking Recall (perfect retrieval): 1 if all required columns retrieved, else 0
    slr = 1 if fn == 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "slr": slr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def generate_report(df: pd.DataFrame, output_csv_path: Path):
    """
    Generate a summary report CSV with aggregated metrics for each NL/Schema combination.

    Args:
        df: DataFrame with evaluation results
        output_csv_path: Path to save the report CSV
    """
    logger.log("info", "GENERATING_SUMMARY_REPORT")

    # Define combinations and methods
    combinations = ["OO", "OL", "LO", "LL"]
    methods = ["Full-Schema", "TCSL", "SCSL"]

    # Initialize report data
    report_data = []

    # Store OO metrics for delta calculation
    oo_metrics = {}

    for combo in combinations:
        row_data = {"NL/Schema": combo}

        for method in methods:
            col_name = f"{combo}_{method}"
            result_col_name = f"{col_name}_results"

            # Parse all results for this combination and method
            results_list = []
            for idx, row in df.iterrows():
                try:
                    result = json.loads(row[result_col_name])
                    if "error" not in result:
                        results_list.append(result)
                except Exception:
                    pass

            if results_list:
                # Calculate aggregated metrics
                avg_precision = sum(r["precision"] for r in results_list) / len(
                    results_list
                )
                avg_recall = sum(r["recall"] for r in results_list) / len(results_list)
                avg_f1 = sum(r["f1"] for r in results_list) / len(results_list)
                avg_fpr = sum(r["fpr"] for r in results_list) / len(results_list)
                avg_slr = sum(r["slr"] for r in results_list) / len(results_list)
                total_tp = sum(r["tp"] for r in results_list)
                total_fp = sum(r["fp"] for r in results_list)
                total_fn = sum(r["fn"] for r in results_list)

                # Store OO metrics for delta calculation
                if combo == "OO":
                    oo_metrics[method] = {"fpr": avg_fpr * 100, "slr": avg_slr * 100}

                # Calculate delta metrics relative to OO
                if combo == "OO":
                    delta_fpr = 0.0
                    delta_slr = 0.0
                else:
                    delta_fpr = round((avg_fpr * 100) - oo_metrics[method]["fpr"], 1)
                    delta_slr = round((avg_slr * 100) - oo_metrics[method]["slr"], 1)

                metrics_dict = {
                    "precision": round(avg_precision * 100, 1),
                    "recall": round(avg_recall * 100, 1),
                    "f1": round(avg_f1 * 100, 1),
                    "fpr": round(avg_fpr * 100, 1),
                    "slr": round(avg_slr * 100, 1),
                    "delta_fpr": delta_fpr,
                    "delta_slr": delta_slr,
                    "total_tp": total_tp,
                    "total_fp": total_fp,
                    "total_fn": total_fn,
                    "valid_queries": len(results_list),
                }

                row_data[method] = json.dumps(metrics_dict)

                logger.log(
                    "info",
                    "COMPUTED_METRICS",
                    {
                        "combination": f"{combo}_{method}",
                        "precision": f"{avg_precision:.4f}",
                        "recall": f"{avg_recall:.4f}",
                        "f1": f"{avg_f1:.4f}",
                        "fpr": f"{avg_fpr:.4f}",
                        "slr": f"{avg_slr:.4f}",
                        "delta_fpr": f"{delta_fpr:.1f}",
                        "delta_slr": f"{delta_slr:.1f}",
                    },
                )
            else:
                row_data[method] = json.dumps({"error": "No valid results"})
                logger.log(
                    "warning", "NO_VALID_RESULTS", {"combination": f"{combo}_{method}"}
                )

        report_data.append(row_data)

    # Create report DataFrame
    report_df = pd.DataFrame(report_data)

    # Save to CSV
    report_df.to_csv(output_csv_path, index=False)
    logger.log("info", "REPORT_SAVED", {"path": str(output_csv_path)})

    # Print summary table
    print("\n=== Summary Report ===")
    print("\n" + report_df.to_string(index=False))


def evaluate_linking(
    input_csv_path: Path, output_csv_path: Path, report_csv_path: Path
):
    # Read input CSV
    df = pd.read_csv(input_csv_path)
    logger.log(
        "info", "LOADED_INPUT_CSV", {"rows": len(df), "path": str(input_csv_path)}
    )

    # Define schema linking columns to evaluate
    schema_linking_columns = [
        "OO_Full-Schema",
        "OO_TCSL",
        "OO_SCSL",
        "OL_Full-Schema",
        "OL_TCSL",
        "OL_SCSL",
        "LO_Full-Schema",
        "LO_TCSL",
        "LO_SCSL",
        "LL_Full-Schema",
        "LL_TCSL",
        "LL_SCSL",
    ]

    # Initialize result columns
    result_columns = [col + "_results" for col in schema_linking_columns]
    for col in result_columns:
        df[col] = None

    # Process each row
    for idx, row in df.iterrows():
        logger.log(
            "info",
            "EVALUATING_ROW",
            {"row": idx + 1, "total": len(df), "question_id": row["question_id"]},
        )

        try:
            # Determine which gold schema to use based on second letter
            # If second letter is 'O', use original_gold_schema
            # If second letter is 'L', use new_gold_schema

            for col in schema_linking_columns:
                # Extract second letter from column name (e.g., 'OO_Full-Schema' -> 'O')
                second_letter = col[1]

                # Select appropriate gold schema
                if second_letter == "O":
                    gold_schema_str = row["original_gold_schema"]
                else:  # second_letter == 'L'
                    gold_schema_str = row["new_gold_schema"]

                # Parse gold schema
                if gold_schema_str == "N/A" or pd.isna(gold_schema_str):
                    logger.log(
                        "warning", "NO_GOLD_SCHEMA", {"row": idx + 1, "column": col}
                    )
                    df.at[idx, col + "_results"] = json.dumps(
                        {"error": "No gold schema available"}
                    )
                    continue

                gold_schema = json.loads(gold_schema_str)

                # Parse predicted schema
                pred_schema_str = row[col]
                if (
                    pred_schema_str == "ERROR"
                    or pd.isna(pred_schema_str)
                    or pred_schema_str == ""
                ):
                    logger.log(
                        "warning", "PREDICTION_ERROR", {"row": idx + 1, "column": col}
                    )
                    df.at[idx, col + "_results"] = json.dumps(
                        {"error": "Prediction error"}
                    )
                    continue

                pred_schema = json.loads(pred_schema_str)

                # Evaluate
                results = evaluate_schema_linking_single(pred_schema, gold_schema)
                df.at[idx, col + "_results"] = json.dumps(results)

                logger.log(
                    "debug",
                    "EVALUATED_COLUMN",
                    {
                        "column": col,
                        "precision": f"{results['precision']:.3f}",
                        "recall": f"{results['recall']:.3f}",
                        "f1": f"{results['f1']:.3f}",
                        "fpr": f"{results['fpr']:.3f}",
                        "slr": results["slr"],
                    },
                )

            logger.log("info", "ROW_EVALUATED_SUCCESSFULLY", {"row": idx + 1})

        except Exception as e:
            logger.log(
                "error",
                "EVALUATION_ERROR",
                {"row": idx + 1, "question_id": row["question_id"], "error": str(e)},
            )
            # Fill with error marker for all result columns if not already set
            for col in schema_linking_columns:
                result_col = col + "_results"
                if pd.isna(df.at[idx, result_col]) or df.at[idx, result_col] == "":
                    df.at[idx, result_col] = json.dumps({"error": str(e)})

        # Write output after each row to preserve progress
        df.to_csv(output_csv_path, index=False)
        logger.log("debug", "SAVED_PROGRESS", {"path": str(output_csv_path)})

    logger.log(
        "info",
        "EVALUATION_COMPLETED",
        {"total_rows": len(df), "output_path": str(output_csv_path)},
    )

    # Print summary statistics
    logger.log("info", "GENERATING_SUMMARY_STATISTICS")
    for col in schema_linking_columns:
        result_col = col + "_results"

        # Parse all results
        results_list = []
        for idx, row in df.iterrows():
            try:
                result = json.loads(row[result_col])
                if "error" not in result:
                    results_list.append(result)
            except Exception:
                pass

        if results_list:
            avg_precision = sum(r["precision"] for r in results_list) / len(
                results_list
            )
            avg_recall = sum(r["recall"] for r in results_list) / len(results_list)
            avg_f1 = sum(r["f1"] for r in results_list) / len(results_list)
            avg_fpr = sum(r["fpr"] for r in results_list) / len(results_list)
            avg_slr = sum(r["slr"] for r in results_list) / len(results_list)
            total_tp = sum(r["tp"] for r in results_list)
            total_fp = sum(r["fp"] for r in results_list)
            total_fn = sum(r["fn"] for r in results_list)

            logger.log(
                "info",
                "COLUMN_STATISTICS",
                {
                    "column": col,
                    "avg_precision": f"{avg_precision:.4f}",
                    "avg_recall": f"{avg_recall:.4f}",
                    "avg_f1": f"{avg_f1:.4f}",
                    "avg_fpr": f"{avg_fpr:.4f}",
                    "avg_slr": f"{avg_slr:.4f}",
                    "total_tp": total_tp,
                    "total_fp": total_fp,
                    "total_fn": total_fn,
                    "valid_evaluations": f"{len(results_list)}/{len(df)}",
                },
            )

    # Generate summary report
    generate_report(df, report_csv_path)


if __name__ == "__main__":
    # Define file paths
    input_csv_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking/new_sql_nl_schema_queries_with_schema_linking.csv"
    )
    output_csv_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking/new_sql_nl_schema_queries_with_schema_linking_evaluated.csv"
    )
    report_csv_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking/report.csv"
    )

    evaluate_linking(input_csv_path, output_csv_path, report_csv_path)

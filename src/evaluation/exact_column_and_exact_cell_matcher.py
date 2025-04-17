import json
import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from src.core.database.database_handler import DatabaseHandler, DBMS
from src.core.logger import Logger

logger = Logger(__name__)


def execute_query(context):
    """Execute both SQL queries and store results in context."""
    db_handler = DatabaseHandler(
        dbms=context["db_params"]["dbms"], connection_params=context["db_params"]
    )

    try:
        # Execute the SQL queries
        gt_columns, gt_rows = db_handler.run_query(context["ground_truth_sql"])
        pred_columns, pred_rows = db_handler.run_query(context["predicted_sql"])
        context.update(
            {
                "gt_columns": gt_columns,
                "gt_rows": gt_rows,
                "pred_columns": pred_columns,
                "pred_rows": pred_rows,
            }
        )

        context["ground_truth_cells"] = len(gt_rows) * len(gt_columns)
        context["predicted_cells"] = len(pred_rows) * len(pred_columns)

    except Exception as e:
        logger.log("error", "QUERY_EXECUTION_FAILED", {"error": str(e)})
        context["has_error"] = True
        context["metrics"] = {
            "EX": 0,
            "EXP": 0.0,
            "EXR": 0.0,
            "F1": 0.0,
        }
        context["ground_truth_cells"] = 0
        context["predicted_cells"] = 0

    context["matched_cells"] = 0


def match_columns(context):
    """Find column intersections and prepare indices for row matching."""
    if context["has_error"]:
        return

    gt_columns = context["gt_columns"]
    pred_columns = context["pred_columns"]

    # Find intersection of columns
    common_cols = set(gt_columns) & set(pred_columns)

    # Check if there are no common columns
    if len(common_cols) == 0:
        context["has_error"] = True
        context["metrics"] = {
            "EX": 0,
            "EXP": 0.0,
            "EXR": 0.0,
            "F1": 0.0,
        }
        context["matched_cells"] = 0
        context["ground_truth_cells"] = len(context["gt_rows"]) * len(gt_columns)
        context["predicted_cells"] = len(context["pred_rows"]) * len(pred_columns)
        return

    # Create index mappings for common columns
    gt_col_to_idx = {col: idx for idx, col in enumerate(gt_columns)}
    pred_col_to_idx = {col: idx for idx, col in enumerate(pred_columns)}

    gt_common_indices = [
        gt_col_to_idx[col] for col in common_cols if col in gt_col_to_idx
    ]
    pred_common_indices = [
        pred_col_to_idx[col] for col in common_cols if col in pred_col_to_idx
    ]

    # Calculate binary execution accuracy
    # TODO: ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0
    ex = (
        1
        if (
            set(gt_columns) == set(pred_columns)
            and set(context["gt_rows"]) == set(context["pred_rows"])
        )
        else 0
    )

    context.update(
        {
            "common_cols": common_cols,
            "gt_common_indices": gt_common_indices,
            "pred_common_indices": pred_common_indices,
            "ex": ex,
        }
    )


def match_rows(context):
    """Project rows to common columns and compute match statistics."""
    if context["has_error"]:
        return

    gt_rows = context["gt_rows"]
    pred_rows = context["pred_rows"]
    gt_common_indices = context["gt_common_indices"]
    pred_common_indices = context["pred_common_indices"]
    common_cols = context["common_cols"]
    gt_columns = context["gt_columns"]

    # Project rows to only include common columns
    gt_projected_rows = [
        tuple(row[idx] for idx in gt_common_indices) for row in gt_rows
    ]
    pred_projected_rows = [
        tuple(row[idx] for idx in pred_common_indices) for row in pred_rows
    ]

    # Calculate total cells and rows
    g_rows = len(gt_rows)
    p_rows = len(pred_rows)
    g_cells = g_rows * len(gt_columns)
    p_cells = p_rows * len(common_cols)

    # Count frequencies of projected rows
    gt_counter = Counter(gt_projected_rows)
    pred_counter = Counter(pred_projected_rows)

    # Calculate matched cells
    matched_rows = 0
    for row_tuple in set(gt_counter) & set(pred_counter):
        matched_rows += min(gt_counter[row_tuple], pred_counter[row_tuple])

    matched_cells = matched_rows * len(common_cols)

    context.update(
        {
            "gt_projected_rows": gt_projected_rows,
            "pred_projected_rows": pred_projected_rows,
            "ground_truth_cells": g_cells,
            "predicted_cells": p_cells,
            "matched_rows": matched_rows,
            "matched_cells": matched_cells,
        }
    )


def assign_metrics(context):
    """Calculate evaluation metrics based on match statistics."""
    if context["has_error"]:
        # Add time taken separately from metrics
        context["time_taken"] = time.time() - context["start_time"]
        return

    # Extract values from context
    matched_cells = context["matched_cells"]
    g_cells = context["ground_truth_cells"]
    p_cells = context["predicted_cells"]
    ex = context["ex"]

    # Calculate time taken
    context["time_taken"] = time.time() - context["start_time"]

    # Handle empty result cases
    if g_cells == 0 and p_cells == 0:
        context["metrics"] = {"EX": 1, "EXP": 1.0, "EXR": 1.0, "F1": 1.0}
    elif g_cells == 0:
        context["metrics"] = {"EX": 0, "EXP": 0.0, "EXR": 1.0, "F1": 0.0}
    elif p_cells == 0:
        context["metrics"] = {"EX": 0, "EXP": 1.0, "EXR": 0.0, "F1": 0.0}
    else:
        # Calculate metrics for normal case
        exp = matched_cells / p_cells  # Execution Precision
        exr = matched_cells / g_cells  # Execution Recall
        f1 = 2 * exp * exr / (exp + exr) if (exp + exr) > 0 else 0.0  # F1 Score

        context["metrics"] = {"EX": ex, "EXP": exp, "EXR": exr, "F1": f1}

    # Log results
    logger.log(
        "info",
        "EVALUATION_COMPLETE",
        {
            "EX": context["metrics"]["EX"],
            "EXP": context["metrics"]["EXP"],
            "EXR": context["metrics"]["EXR"],
            "F1": context["metrics"]["F1"],
            "MATCHED_CELLS": matched_cells,
            "GROUND_TRUTH_CELLS": g_cells,
            "PREDICTED_CELLS": p_cells,
        },
    )


def dump_logs(context, log_files_dir=None):
    """
    Report evaluation results and optionally save to file.
    """
    metrics = context.get("metrics", {})
    print("=================== Evaluation Results ===================")
    print(f"EX (Binary Execution Accuracy): {metrics.get('EX', 0)}")
    print(f"EXP (Execution Precision): {metrics.get('EXP', 0.0):.4f}")
    print(f"EXR (Execution Recall): {metrics.get('EXR', 0.0):.4f}")
    print(f"F1 Score: {metrics.get('F1', 0.0):.4f}")
    print(f"Time taken: {context.get('time_taken', 0.0):.2f} seconds")
    print("==========================================================")

    def json_serializer(obj):
        """Convert non-serializable objects to strings or other JSON-serializable types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)  # For any other type, just convert to string

    if log_files_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_files_dir, exist_ok=True)
        log_filename = os.path.join(
            log_files_dir, f"evaluation_results_{timestamp}.json"
        )

        with open(log_filename, "w") as f:
            json.dump(context, f, indent=2, default=json_serializer)

        logger.log("info", "EVALUATION_LOGS_SAVED", {"log_file": log_filename})


def run_evaluation_pipeline(
    db_params, predicted_sql, ground_truth_sql, log_files_dir=None
):
    """Orchestrate the complete SQL evaluation pipeline."""
    context = {
        "db_params": db_params,
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "start_time": time.time(),
        "has_error": False,
    }

    execute_query(context)
    match_columns(context)
    match_rows(context)
    assign_metrics(context)

    dump_logs(context, log_files_dir=log_files_dir)

    return context


if __name__ == "__main__":
    predicted_sql = """
    SELECT T3.Phone
    FROM satscores T1 
    JOIN schools T3 ON T1.cds = T3.CDSCode 
    WHERE T1.NumTstTakr IS NOT NULL AND T1.NumGE1500 IS NOT NULL 
    ORDER BY (T1.NumGE1500 * 1.0 / T1.NumTstTakr) DESC 
    LIMIT 10;
    """
    ground_truth_sql = """
    SELECT T1.Phone
    FROM schools AS T1 
    INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds 
    ORDER BY CAST(T2.NumGE1500 AS REAL) / T2.NumTstTakr DESC 
    LIMIT 10;
    """

    db_params = {
        "dbms": DBMS.SQLITE,
        "db_path": "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite",
    }
    log_file_dir = ".data/evaluation_metrics_logs/_1_row_semantic_matcher/"

    run_evaluation_pipeline(
        db_params=db_params,
        predicted_sql=predicted_sql,
        ground_truth_sql=ground_truth_sql,
        log_files_dir=log_file_dir,
    )

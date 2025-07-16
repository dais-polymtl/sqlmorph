import time
from collections import Counter

from src.core.database.database_handler import DatabaseHandler, DBMS
from src.core.logger import Logger

logger = Logger(__name__)


def execute_query(context: dict):
    logger.log("debug", "function execute_query called")
    try:
        db_handler = DatabaseHandler(
            dbms=context["db_params"]["dbms"], connection_params=context["db_params"]
        )

        pred_cols, pred_rows = db_handler.run_query(context["predicted_sql"])
        gt_cols, gt_rows = db_handler.run_query(context["ground_truth_sql"])

        context.update(
            {
                "has_error": False,
                "pred_cols": pred_cols,
                "pred_rows": pred_rows,
                "gt_cols": gt_cols,
                "gt_rows": gt_rows,
                "ground_truth_cells": len(gt_rows) * len(gt_cols),
                "predicted_cells": len(pred_rows) * len(pred_cols),
            }
        )
        return context
    except Exception as e:
        logger.log("error", "QUERY_EXECUTION_FAILED", {"error": str(e)})
        context.update(
            {
                "has_error": True,
                "error_message": str(e),
                "matched_cells": 0,
                "ground_truth_cells": 0,
                "predicted_cells": 0,
                "metrics": {"EXP": 0.0, "EXR": 0.0, "F1": 0.0, "EX": 0},
            }
        )
        return context


def match_columns(context: dict):
    logger.log("debug", "function match_columns called")
    if context["has_error"]:
        return context

    gt_cols = context["gt_cols"]
    pred_cols = context["pred_cols"]

    # Find intersection of columns
    common_cols = set(gt_cols) & set(pred_cols)

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
        context["ground_truth_cells"] = len(context["gt_rows"]) * len(gt_cols)
        context["predicted_cells"] = len(context["pred_rows"]) * len(pred_cols)
        return context

    # Create index mappings for common columns
    gt_col_to_idx = {col: idx for idx, col in enumerate(gt_cols)}
    pred_col_to_idx = {col: idx for idx, col in enumerate(pred_cols)}

    gt_common_indices = [
        gt_col_to_idx[col] for col in common_cols if col in gt_col_to_idx
    ]
    pred_common_indices = [
        pred_col_to_idx[col] for col in common_cols if col in pred_col_to_idx
    ]

    context.update(
        {
            "common_cols": common_cols,
            "gt_common_indices": gt_common_indices,
            "pred_common_indices": pred_common_indices,
        }
    )

    return context


def match_rows(context: dict):
    logger.log("debug", "function match_rows called")
    if context["has_error"]:
        return context

    gt_rows = context["gt_rows"]
    pred_rows = context["pred_rows"]
    gt_common_indices = context["gt_common_indices"]
    pred_common_indices = context["pred_common_indices"]
    common_cols = context["common_cols"]
    gt_cols = context["gt_cols"]

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
    g_cells = g_rows * len(gt_cols)
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

    return context


def assign_metrics(context: dict):
    logger.log("debug", "function assign_metrics called")
    if context["has_error"]:
        return context

    # Extract values from context
    matched_cells = context["matched_cells"]
    g_cells = context["ground_truth_cells"]
    p_cells = context["predicted_cells"]
    ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0

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

    logger.log(
        "info",
        "EVALUATION_COMPLETE",
        {
            "EX": context["metrics"]["EX"],
            "EXP": context["metrics"]["EXP"],
            "EXR": context["metrics"]["EXR"],
            "F1": context["metrics"]["F1"],
        },
    )

    return context


def run_evaluation_pipeline(
    predicted_sql: str,
    ground_truth_sql: str,
    db_params: dict,
):
    context = {
        "db_params": db_params,
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "metrics": {},
        "has_error": False,
    }

    start_time = time.time()

    context = execute_query(context)
    context = match_columns(context)
    context = match_rows(context)
    context = assign_metrics(context)

    context["latency"] = time.time() - start_time

    return context


if __name__ == "__main__":
    # ad-hoc example to test the evaluation technique! check out evaluation_metrics.py for the main entry point

    # input
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
    LIMIT 20;
    """

    # config
    db_params = {
        "dbms": DBMS.SQLITE,
        "db_path": "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite",
    }

    context = run_evaluation_pipeline(
        predicted_sql=predicted_sql,
        ground_truth_sql=ground_truth_sql,
        db_params=db_params,
    )

    # print evaluation results
    metrics = context.get("metrics", {})
    print("=================== Evaluation Results ===================")
    print(f"EX (Binary Execution Accuracy): {metrics.get('EX', 0)}")
    print(f"EXP (Execution Precision): {metrics.get('EXP', 0.0):.4f}")
    print(f"EXR (Execution Recall): {metrics.get('EXR', 0.0):.4f}")
    print(f"F1 Score: {metrics.get('F1', 0.0):.4f}")
    print(f"Time taken: {context.get('latency', 0.0):.2f} seconds")
    print("==========================================================")

import time

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


def check_ex(context: dict):
    logger.log("debug", "function check_execution_accuracy called")
    if context["has_error"]:
        return context

    ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0

    # If EX is 1, set all metrics to 1 and skip remaining stages
    if ex == 1:
        context["metrics"] = {"EX": 1, "EXP": 1.0, "EXR": 1.0, "F1": 1.0}
        context["ex_is_one"] = True
        logger.log(
            "info",
            "EX_IS_ONE_SKIPPING_REMAINING_STAGES",
            {
                "EX": 1,
                "EXP": 1.0,
                "EXR": 1.0,
                "F1": 1.0,
            },
        )
    else:
        context["ex_is_one"] = False

    return context


def match_columns(context: dict):
    logger.log("debug", "function match_columns called")
    if context["has_error"] or context.get("ex_is_one", False):
        return context

    # No column matching needed - we keep all columns as cell tokens

    return context


def match_rows(context: dict):
    logger.log("debug", "function match_rows called")
    if context["has_error"] or context.get("ex_is_one", False):
        return context

    gt_rows = context["gt_rows"]
    pred_rows = context["pred_rows"]
    gt_cols = context["gt_cols"]
    pred_cols = context["pred_cols"]

    # Step 1: Convert rows to cell token sets (col_name=value)
    def row_to_cell_tokens(row, columns):
        """Convert a row to a set of cell tokens in the form 'col_name=value'"""
        return set(f"{col}={val}" for col, val in zip(columns, row))

    gt_cell_sets = [row_to_cell_tokens(row, gt_cols) for row in gt_rows]
    pred_cell_sets = [row_to_cell_tokens(row, pred_cols) for row in pred_rows]

    # Calculate total cells
    g_cells = sum(len(cell_set) for cell_set in gt_cell_sets)
    p_cells = sum(len(cell_set) for cell_set in pred_cell_sets)

    # Step 2: Two-phase row matching

    # Phase A: Exact row matching
    exact_matched_cells = 0
    remaining_gt_indices = list(range(len(gt_cell_sets)))
    remaining_pred_indices = list(range(len(pred_cell_sets)))

    # Find exact matches
    gt_to_remove = []
    pred_to_remove = []

    for gt_idx in remaining_gt_indices:
        for pred_idx in remaining_pred_indices:
            if gt_cell_sets[gt_idx] == pred_cell_sets[pred_idx]:
                # Exact match found
                exact_matched_cells += len(gt_cell_sets[gt_idx])
                gt_to_remove.append(gt_idx)
                pred_to_remove.append(pred_idx)
                break  # 1-to-1 matching

    # Remove exact matches from remaining indices
    for idx in sorted(gt_to_remove, reverse=True):
        remaining_gt_indices.remove(idx)
    for idx in sorted(pred_to_remove, reverse=True):
        remaining_pred_indices.remove(idx)

    logger.log(
        "debug", f"Phase A complete: {exact_matched_cells} cells from exact matches"
    )

    # Phase B: Partial row matching using Jaccard similarity
    partial_matched_cells = 0

    while remaining_gt_indices and remaining_pred_indices:
        best_similarity = 0
        best_gt_idx = -1
        best_pred_idx = -1
        best_intersection_size = 0

        # Find the pair with highest Jaccard similarity
        for gt_idx in remaining_gt_indices:
            for pred_idx in remaining_pred_indices:
                gt_set = gt_cell_sets[gt_idx]
                pred_set = pred_cell_sets[pred_idx]

                intersection = gt_set & pred_set
                union = gt_set | pred_set

                if len(union) > 0:
                    jaccard_sim = len(intersection) / len(union)

                    # Break ties deterministically using lexicographic order
                    if jaccard_sim > best_similarity or (
                        jaccard_sim == best_similarity
                        and (gt_idx, pred_idx) < (best_gt_idx, best_pred_idx)
                    ):
                        best_similarity = jaccard_sim
                        best_gt_idx = gt_idx
                        best_pred_idx = pred_idx
                        best_intersection_size = len(intersection)

        # If we found a match with some similarity, record it
        if best_similarity > 0:
            partial_matched_cells += best_intersection_size
            remaining_gt_indices.remove(best_gt_idx)
            remaining_pred_indices.remove(best_pred_idx)
        else:
            # No more matches possible
            break

    logger.log(
        "debug", f"Phase B complete: {partial_matched_cells} cells from partial matches"
    )

    # Step 3: Compute totals
    total_matched_cells = exact_matched_cells + partial_matched_cells

    context.update(
        {
            "gt_cell_sets": gt_cell_sets,
            "pred_cell_sets": pred_cell_sets,
            "ground_truth_cells": g_cells,
            "predicted_cells": p_cells,
            "matched_cells": total_matched_cells,
            "exact_matched_cells": exact_matched_cells,
            "partial_matched_cells": partial_matched_cells,
        }
    )

    return context


def assign_metrics(context: dict):
    logger.log("debug", "function assign_metrics called")
    if context["has_error"] or context.get("ex_is_one", False):
        return context

    # Extract values from context
    matched_cells = context["matched_cells"]
    g_cells = context["ground_truth_cells"]
    p_cells = context["predicted_cells"]
    ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0

    # Handle empty result cases
    if g_cells == 0 and p_cells == 0:
        context["metrics"] = {"EX": ex, "EXP": 1.0, "EXR": 1.0, "F1": 1.0}
    elif g_cells == 0:
        context["metrics"] = {"EX": ex, "EXP": 0.0, "EXR": 1.0, "F1": 0.0}
    elif p_cells == 0:
        context["metrics"] = {"EX": ex, "EXP": 1.0, "EXR": 0.0, "F1": 0.0}
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
        "ex_is_one": False,
    }

    start_time = time.time()

    context = execute_query(context)
    context = check_ex(context)
    context = match_columns(context)
    context = match_rows(context)
    context = assign_metrics(context)

    context["latency"] = time.time() - start_time

    return context


if __name__ == "__main__":
    # ad-hoc example to test the evaluation technique! check out evaluation_metrics.py for the main entry point

    # input
    predicted_sql = """
    SELECT T3.Phone, T3.City, T3.State, T3.MailStreet
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

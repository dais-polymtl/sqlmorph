import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from src.core.database import DatabaseHandler, DBMS
from src.core.logger import Logger
from src.core.model_manager import (
    ModelManager,
    ModelProvider,
    ModelType,
    OpenAIModel,
    OllamaModel,
    HuggingFaceModel,
)

logger = Logger(__name__)


def execute_query(context):
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
                "gt_df": pd.DataFrame(gt_rows, columns=gt_cols),
                "pred_df": pd.DataFrame(pred_rows, columns=pred_cols),
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


def match_columns(context):
    logger.log("debug", "function match_columns called")
    if context["has_error"] or context.get("ex_is_one", False):
        return context

    embedding_model = ModelManager.create_model(
        model_provider=ModelProvider.OPENAI,
        model_type=ModelType.EMBEDDING,
        model_name=context["embedding_model_name"],
        openai_api_key=os.getenv("OPENAI_API_KEY", None),
    )
    # Extract column data for embedding generation
    gt_col_data = {}
    pred_col_data = {}

    # Extract column data from ground truth dataframe
    for col in context["gt_df"].columns:
        data_type = str(context["gt_df"][col].dtype)

        # Get top frequent values (up to 10)
        if len(context["gt_df"]) > 0:
            value_counts = context["gt_df"][col].value_counts().head(10)
            top_values = [str(v) for v in value_counts.index]
        else:
            top_values = []

        gt_col_data[col] = {
            "name": col,
            "data_type": data_type,
            "top_values": top_values,
        }

    # Extract column data from predicted dataframe
    for col in context["pred_df"].columns:
        data_type = str(context["pred_df"][col].dtype)

        # Get top frequent values (up to 10)
        if len(context["pred_df"]) > 0:
            value_counts = context["pred_df"][col].value_counts().head(10)
            top_values = [str(v) for v in value_counts.index]
        else:
            top_values = []

        pred_col_data[col] = {
            "name": col,
            "data_type": data_type,
            "top_values": top_values,
        }

    # Generate column embeddings for ground truth
    gt_embeddings = {}
    for col, data in gt_col_data.items():
        # Combine column name and top values into a single text representation
        text = f"Column name: {data['name']}. Data type: {data['data_type']}. "

        if data["top_values"]:
            text += f"Sample values: {', '.join(data['top_values'])}"

        # Generate embedding
        embedding = embedding_model.get_embedding(text)
        gt_embeddings[col] = embedding

    # Generate column embeddings for predictions
    pred_embeddings = {}
    for col, data in pred_col_data.items():
        # Combine column name and top values into a single text representation
        text = f"Column name: {data['name']}. Data type: {data['data_type']}. "

        if data["top_values"]:
            text += f"Sample values: {', '.join(data['top_values'])}"

        # Generate embedding
        embedding = embedding_model.get_embedding(text)
        pred_embeddings[col] = embedding

    # Compute similarity matrix between ground truth and predicted columns
    pred_cols = list(pred_embeddings.keys())
    gt_cols = list(gt_embeddings.keys())

    similarity_matrix = np.zeros((len(pred_cols), len(gt_cols)))

    for i, pred_col in enumerate(pred_cols):
        pred_embedding = pred_embeddings[pred_col]

        for j, gt_col in enumerate(gt_cols):
            gt_embedding = gt_embeddings[gt_col]

            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(pred_embedding, gt_embedding)
            similarity_matrix[i, j] = similarity

    # Use the Hungarian algorithm to find optimal matching between columns
    cost_matrix = -similarity_matrix  # Convert similarities to costs
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get matched column pairs (only those with similarity > 0.7)
    matched_cols = []
    for pred_idx, gt_idx in enumerate(col_ind):
        if (
            similarity_matrix[pred_idx, gt_idx] > 0.7
        ):  # Only consider sufficiently similar columns
            matched_cols.append(
                (context["pred_cols"][pred_idx], context["gt_cols"][gt_idx])
            )

    # Sort matched columns lexicographically for deterministic behavior
    matched_cols.sort(key=lambda x: (x[0], x[1]))

    logger.log("debug", "COLUMN_MATCHING", {"MATCHED_COLUMNS": matched_cols})

    # Update context with column matching results
    context.update(
        {
            "matched_cols": matched_cols,
            "num_matched_cols": len(matched_cols),
            "similarity_matrix": similarity_matrix,
            "col_matches": col_ind,
        }
    )

    return context


def match_rows(context):
    logger.log("debug", "function match_rows called")
    if context["has_error"] or context.get("ex_is_one", False):
        return context

    gt_cols = context["gt_cols"]
    pred_cols = context["pred_cols"]
    gt_rows = context["gt_rows"]
    pred_rows = context["pred_rows"]
    matched_cols = context["matched_cols"]

    # Get indices for matched columns in their original order for deterministic behavior
    gt_matched_indices = []
    pred_matched_indices = []

    # Process matched columns (already sorted in match_columns function)
    for pred_col, gt_col in matched_cols:
        if gt_col in gt_cols and pred_col in pred_cols:
            gt_idx = gt_cols.index(gt_col)
            pred_idx = pred_cols.index(pred_col)
            gt_matched_indices.append(gt_idx)
            pred_matched_indices.append(pred_idx)

    # Project rows to only include matched columns
    gt_projected_rows = []
    for row in gt_rows:
        projected = tuple(row[idx] for idx in gt_matched_indices)
        gt_projected_rows.append(projected)

    pred_projected_rows = []
    for row in pred_rows:
        projected = tuple(row[idx] for idx in pred_matched_indices)
        pred_projected_rows.append(projected)

    # Calculate total cells counts
    g_rows = len(gt_rows)
    p_rows = len(pred_rows)
    g_cells = g_rows * len(gt_cols)
    p_cells = p_rows * len(pred_cols)
    num_matched_cols = len(matched_cols)

    # Phase 1: Exact Row Matching using efficient frequency-based approach
    gt_counter = Counter(gt_projected_rows)
    pred_counter = Counter(pred_projected_rows)

    # Find exact matches and count matched cells
    exact_matched_cells = 0
    matched_row_counts = {}

    for row in gt_counter:
        if row in pred_counter:
            # Number of matches is the minimum count between gt and pred
            matches = min(gt_counter[row], pred_counter[row])
            matched_row_counts[row] = matches
            exact_matched_cells += matches * num_matched_cols

    logger.log(
        "debug", f"Phase 1 complete: {exact_matched_cells} cells from exact matches"
    )

    # Phase 2: Partial Row Matching for remaining unmatched rows
    # Create lists of remaining unmatched rows
    remaining_gt_rows = []
    remaining_pred_rows = []

    # Add unmatched gt rows
    for row, count in gt_counter.items():
        matched_count = matched_row_counts.get(row, 0)
        remaining_count = count - matched_count
        remaining_gt_rows.extend([row] * remaining_count)

    # Add unmatched pred rows
    for row, count in pred_counter.items():
        matched_count = matched_row_counts.get(row, 0)
        remaining_count = count - matched_count
        remaining_pred_rows.extend([row] * remaining_count)

    # Perform partial matching on remaining rows using greedy strategy
    partial_matched_cells = 0.0

    while remaining_gt_rows and remaining_pred_rows:
        best_similarity = 0
        best_gt_idx = -1
        best_pred_idx = -1

        # Find the pair with highest similarity score
        for gt_idx, gt_row in enumerate(remaining_gt_rows):
            for pred_idx, pred_row in enumerate(remaining_pred_rows):
                # Calculate cell-level similarity between rows
                matching_cells = sum(
                    1
                    for gt_val, pred_val in zip(gt_row, pred_row)
                    if gt_val == pred_val
                )
                similarity = (
                    matching_cells / num_matched_cols if num_matched_cols > 0 else 0
                )  # Normalize by number of matched columns

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gt_idx = gt_idx
                    best_pred_idx = pred_idx

        # If we found a match with some similarity, record it and remove the rows
        if best_similarity > 0:
            # Partial match contributes fractional cells based on similarity
            partial_matched_cells += best_similarity * num_matched_cols
            # Remove the matched rows from consideration
            remaining_gt_rows.pop(best_gt_idx)
            remaining_pred_rows.pop(best_pred_idx)
        else:
            # No more matches possible, break out of loop
            break

    logger.log(
        "debug", f"Phase 2 complete: {partial_matched_cells} cells from partial matches"
    )

    # Total matched cells = exact matches + partial matches
    total_matched_cells = exact_matched_cells + partial_matched_cells

    # Update context with row matching results
    context.update(
        {
            "gt_projected_rows": gt_projected_rows,
            "pred_projected_rows": pred_projected_rows,
            "matched_rows": (
                exact_matched_cells // num_matched_cols if num_matched_cols > 0 else 0
            ),  # For backward compatibility
            "matched_cells": total_matched_cells,
            "exact_matched_cells": exact_matched_cells,
            "partial_matched_cells": partial_matched_cells,
            "ground_truth_cells": g_cells,
            "predicted_cells": p_cells,
        }
    )

    return context


def assign_metrics(context):
    logger.log("debug", "function assign_metrics called")
    if context["has_error"] or context.get("ex_is_one", False):
        return context

    g_cells = context["ground_truth_cells"]
    p_cells = context["predicted_cells"]
    matched_cells = context["matched_cells"]

    # Handle edge cases
    if g_cells == 0 and p_cells == 0:
        metrics = {
            "EX": 1,  # Exact match (both empty)
            "EXP": 1.0,  # Both empty, perfect precision
            "EXR": 1.0,  # Both empty, perfect recall
            "F1": 1.0,  # Perfect F1 score
        }
    elif g_cells == 0:
        metrics = {
            "EX": 0,  # Not an exact match
            "EXP": 0.0,  # Predicted something when nothing expected
            "EXR": 1.0,  # No ground truth to recall
            "F1": 0.0,  # Zero F1 score
        }
    elif p_cells == 0:
        metrics = {
            "EX": 0,  # Not an exact match
            "EXP": 1.0,  # No prediction to be wrong
            "EXR": 0.0,  # Failed to recall anything
            "F1": 0.0,  # Zero F1 score
        }
    else:
        EX = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0
        EXP = matched_cells / p_cells if p_cells > 0 else 0.0
        EXR = matched_cells / g_cells if g_cells > 0 else 0.0
        F1 = 2 * EXP * EXR / (EXP + EXR) if (EXP + EXR) > 0 else 0.0
        metrics = {"EX": EX, "EXP": EXP, "EXR": EXR, "F1": F1}

    context["metrics"] = metrics

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
    embedding_model: OpenAIModel | OllamaModel | HuggingFaceModel,
):
    """
    Orchestrate the SQL result evaluation pipeline by running the four stages in sequence.
    """
    context = {
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "db_params": db_params,
        "embedding_model_name": embedding_model,
        "metrics": {},
        "has_error": False,
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
    SELECT T3.Phone AS P, T3.City AS SHA
    FROM satscores T1 
    JOIN schools T3 ON T1.cds = T3.CDSCode 
    WHERE T1.NumTstTakr IS NOT NULL AND T1.NumGE1500 IS NOT NULL 
    ORDER BY (T1.NumGE1500 * 1.0 / T1.NumTstTakr) DESC 
    LIMIT 10;
    """
    ground_truth_sql = """
    SELECT T1.Phone, T1.City, T1.State, T1.MailStreet
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
    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL

    context = run_evaluation_pipeline(
        predicted_sql=predicted_sql,
        ground_truth_sql=ground_truth_sql,
        db_params=db_params,
        embedding_model=embedding_model,
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

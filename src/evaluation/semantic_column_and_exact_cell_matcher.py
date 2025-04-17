import os
import time
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from src.core.database import DatabaseHandler, DBMS
from src.core.logger import Logger
from src.core.model_manager import ModelManager, ModelProvider, ModelType, OpenAIModel

logger = Logger(__name__)


def execute_query(context):
    """
    Execute the SQL queries and store the results in the context.
    """

    db_handler = DatabaseHandler(
        dbms=context["db_params"]["dbms"], connection_params=context["db_params"]
    )

    # TODO: Fix raising error in the database handler
    # Execute ground truth and predicted queries
    gt_columns, gt_rows = db_handler.run_query(context["ground_truth_sql"])
    pred_columns, pred_rows = db_handler.run_query(context["predicted_sql"])

    # Initialize metrics dictionary in context if it doesn't exist
    if "metrics" not in context:
        context["metrics"] = {}

    # Check for query execution failure
    if not gt_columns or not pred_columns:
        logger.log(
            "error",
            "QUERY_EXECUTION_FAILED",
            {
                "GROUND_TRUTH_COLUMNS": len(gt_columns),
                "PREDICTED_COLUMNS": len(pred_columns),
            },
        )

        # Store metrics in the metrics dictionary
        context["metrics"].update({"EX": 0, "EXP": 0.0, "EXR": 0.0, "F1": 0.0})

        context.update(
            {
                "execution_failed": True,
                "matched_cells": 0,
                "ground_truth_cells": (
                    len(gt_rows) * len(gt_columns) if gt_columns else 0
                ),
                "predicted_cells": (
                    len(pred_rows) * len(pred_columns) if pred_columns else 0
                ),
            }
        )
        return context

    # Binary execution accuracy - only true if rows match exactly
    ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0

    # Store metrics in the metrics dictionary
    context["metrics"]["EX"] = ex

    # Update context with query results
    context.update(
        {
            "execution_failed": False,
            "gt_columns": gt_columns,
            "gt_rows": gt_rows,
            "pred_columns": pred_columns,
            "pred_rows": pred_rows,
            "gt_df": pd.DataFrame(gt_rows, columns=gt_columns),
            "pred_df": pd.DataFrame(pred_rows, columns=pred_columns),
        }
    )

    return context


def match_columns(context):
    """
    Perform semantic column matching to find corresponding columns.

    Args:
        context (dict): Pipeline context with query results

    Returns:
        dict: Updated context with column matching results
    """
    if context.get("execution_failed", False):
        return context

    # Initialize embedding model if not already in context
    if "embedding_model" not in context:
        context["embedding_model"] = ModelManager.create_model(
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
        embedding = context["embedding_model"].get_embedding(text)
        gt_embeddings[col] = embedding

    # Generate column embeddings for predictions
    pred_embeddings = {}
    for col, data in pred_col_data.items():
        # Combine column name and top values into a single text representation
        text = f"Column name: {data['name']}. Data type: {data['data_type']}. "

        if data["top_values"]:
            text += f"Sample values: {', '.join(data['top_values'])}"

        # Generate embedding
        embedding = context["embedding_model"].get_embedding(text)
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
    matched_columns = []
    for pred_idx, gt_idx in enumerate(col_ind):
        if (
            similarity_matrix[pred_idx, gt_idx] > 0.7
        ):  # Only consider sufficiently similar columns
            matched_columns.append(
                (context["pred_columns"][pred_idx], context["gt_columns"][gt_idx])
            )

    logger.log("debug", "COLUMN_MATCHING", {"MATCHED_COLUMNS": matched_columns})

    # Update context with column matching results
    context.update(
        {
            "matched_columns": matched_columns,
            "num_matched_columns": len(matched_columns),
            "similarity_matrix": similarity_matrix,
            "col_matches": col_ind,
        }
    )

    return context


def match_rows(context):
    """
    Match rows across corresponding columns and calculate cell-level matches.

    Args:
        context (dict): Pipeline context with matched columns

    Returns:
        dict: Updated context with row matching results
    """
    if context.get("execution_failed", False):
        return context

    # Extract data needed for row matching
    gt_columns = context["gt_columns"]
    pred_columns = context["pred_columns"]
    gt_rows = context["gt_rows"]
    pred_rows = context["pred_rows"]
    matched_columns = context["matched_columns"]

    # Get indices for matched columns
    gt_matched_indices = [
        idx
        for idx, col in enumerate(gt_columns)
        if col in [gt_col for _, gt_col in matched_columns]
    ]
    pred_matched_indices = [
        idx
        for idx, col in enumerate(pred_columns)
        if col in [pred_col for pred_col, _ in matched_columns]
    ]

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
    g_cells = g_rows * len(gt_columns)
    p_cells = p_rows * len(pred_columns)

    # Count frequencies of projected rows in both datasets
    gt_counter = Counter(gt_projected_rows)
    pred_counter = Counter(pred_projected_rows)

    # Calculate matched rows (intersection of row patterns)
    matched_rows = 0
    for row_tuple in set(gt_counter) & set(pred_counter):
        min_count = min(gt_counter[row_tuple], pred_counter[row_tuple])
        matched_rows += min_count

    # Calculate matched cells: |MatchedRows| × |MatchedColumns|
    matched_cells = matched_rows * context["num_matched_columns"]

    # Update context with row matching results
    context.update(
        {
            "gt_projected_rows": gt_projected_rows,
            "pred_projected_rows": pred_projected_rows,
            "matched_rows": matched_rows,
            "matched_cells": matched_cells,
            "ground_truth_cells": g_cells,
            "predicted_cells": p_cells,
        }
    )

    return context


def assign_metrics(context):
    """
    Calculate final evaluation metrics based on matched cells.

    Args:
        context (dict): Pipeline context with matched cells data

    Returns:
        dict: Updated context with final evaluation metrics
    """
    if context.get("execution_failed", False):
        return context

    g_cells = context["ground_truth_cells"]
    p_cells = context["predicted_cells"]
    matched_cells = context["matched_cells"]

    # Handle special cases for empty results
    if g_cells == 0 and p_cells == 0:
        context["metrics"].update(
            {
                "EX": 1,  # Exact match (both empty)
                "EXP": 1.0,  # Both empty, perfect precision
                "EXR": 1.0,  # Both empty, perfect recall
                "F1": 1.0,  # Perfect F1 score
            }
        )
        return context
    elif g_cells == 0:
        context["metrics"].update(
            {
                "EX": 0,  # Not an exact match
                "EXP": 0.0,  # Predicted something when nothing expected
                "EXR": 1.0,  # No ground truth to recall
                "F1": 0.0,  # Zero F1 score
            }
        )
        return context
    elif p_cells == 0:
        context["metrics"].update(
            {
                "EX": 0,  # Not an exact match
                "EXP": 1.0,  # No prediction to be wrong
                "EXR": 0.0,  # Failed to recall anything
                "F1": 0.0,  # Zero F1 score
            }
        )
        return context

    # Calculate metrics
    exp = matched_cells / p_cells if p_cells > 0 else 0.0  # Execution Precision
    exr = matched_cells / g_cells if g_cells > 0 else 0.0  # Execution Recall
    f1 = 2 * exp * exr / (exp + exr) if (exp + exr) > 0 else 0.0  # F1 Score

    # Update metrics in the metrics dictionary
    context["metrics"].update({"EXP": exp, "EXR": exr, "F1": f1})

    # Log evaluation results
    logger.log(
        "info",
        "EVALUATION_COMPLETE",
        {
            "EX": context["metrics"]["EX"],
            "EXP": exp,
            "EXR": exr,
            "F1": f1,
            "MATCHED_CELLS": matched_cells,
            "GROUND_TRUTH_CELLS": g_cells,
            "PREDICTED_CELLS": p_cells,
        },
    )

    return context


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
    predicted_sql, ground_truth_sql, db_params, embedding_model, log_file_dir=None
):
    """
    Orchestrate the SQL result evaluation pipeline by running the four stages in sequence.
    """
    context = {
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "db_params": db_params,
        "embedding_model_name": embedding_model,
        "start_time": time.time(),
        "metrics": {},
    }

    context = execute_query(context)
    context = match_columns(context)
    context = match_rows(context)
    context = assign_metrics(context)

    context["time_taken"] = time.time() - context["start_time"]

    dump_logs(context, log_file_dir)


if __name__ == "__main__":
    # Example usage
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

    db_params = {
        "dbms": DBMS.SQLITE,
        "db_path": "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite",
    }
    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL
    log_file_dir = ".data/evaluation_metrics_logs/_1_row_semantic_matcher/"

    run_evaluation_pipeline(
        predicted_sql=predicted_sql,
        ground_truth_sql=ground_truth_sql,
        db_params=db_params,
        embedding_model=embedding_model,
        log_file_dir=log_file_dir,
    )

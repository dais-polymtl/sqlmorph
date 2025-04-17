import json
import os
import time
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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
    """
    Execute SQL queries and handle errors. Runs both predicted and ground truth queries,
    calculates binary execution accuracy, and updates the context with the results.
    """
    db_handler = DatabaseHandler(
        dbms=context["db_params"]["dbms"], connection_params=context["db_params"]
    )

    # TODO: Fix raising error in the database handler
    try:
        # Execute both SQL queries
        pred_cols, pred_rows = db_handler.run_query(context["predicted_sql"])
        gt_cols, gt_rows = db_handler.run_query(context["ground_truth_sql"])

        # Compute binary execution accuracy (exact match of results)
        # TODO: ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0
        ex = (
            1
            if (
                sorted(pred_cols) == sorted(gt_cols)
                and sorted(pred_rows) == sorted(gt_rows)
            )
            else 0
        )

        context.update(
            {
                "pred_cols": pred_cols,
                "pred_rows": pred_rows,
                "gt_cols": gt_cols,
                "gt_rows": gt_rows,
                "ex": ex,
                "has_error": False,
            }
        )

    except Exception as e:
        error_message = f"SQL execution failed: {str(e)}"
        logger.log("error", "QUERY_EXECUTION_FAILED", {"error": str(e)})

        context.update(
            {
                "has_error": True,
                "error_message": error_message,
                "metrics": {"EXP": 0.0, "EXR": 0.0, "F1": 0.0, "EX": 0},
                "time_taken": float(time.time() - context["start_time"]),
                "error": error_message,
            }
        )


def match_columns(context):
    """
    Match columns and build row representations. Creates unified column sets,
    builds string representations for rows, and calculates column coverage penalty.
    """
    if context.get("has_error", False):
        return

    pred_cols = context["pred_cols"]
    gt_cols = context["gt_cols"]
    pred_rows = context["pred_rows"]
    gt_rows = context["gt_rows"]

    # Compute column union and sort
    all_cols = sorted(set(pred_cols).union(set(gt_cols)))

    # Build row representations for predicted results
    pred_row_strings = []
    pred_row_dicts = []
    for row in pred_rows:
        row_dict = dict(zip(pred_cols, row))
        final_values = []
        final_dict = {}

        for col in all_cols:
            value = row_dict.get(col, "N/A")
            final_values.append(str(value))
            final_dict[col] = value

        pred_row_strings.append(" | ".join(final_values))
        pred_row_dicts.append(final_dict)

    # Build row representations for ground truth results
    gt_row_strings = []
    gt_row_dicts = []
    for row in gt_rows:
        row_dict = dict(zip(gt_cols, row))
        final_values = []
        final_dict = {}

        for col in all_cols:
            value = row_dict.get(col, "N/A")
            final_values.append(str(value))
            final_dict[col] = value

        gt_row_strings.append(" | ".join(final_values))
        gt_row_dicts.append(final_dict)

    # Compute column coverage penalty
    gt_set = set(gt_cols)
    coverage_penalty = (
        len(set(pred_cols).intersection(gt_set)) / len(gt_set) if gt_set else 1.0
    )

    context.update(
        {
            "all_cols": all_cols,
            "pred_row_strings": pred_row_strings,
            "pred_row_dicts": pred_row_dicts,
            "gt_row_strings": gt_row_strings,
            "gt_row_dicts": gt_row_dicts,
            "coverage_penalty": coverage_penalty,
        }
    )


def match_rows(context):
    """
    Match rows using embeddings and bipartite matching. Embeds row strings,
    computes similarity matrix, and performs optimal matching using the Hungarian algorithm.
    """
    if context.get("has_error", False):
        return

    pred_row_strings = context["pred_row_strings"]
    gt_row_strings = context["gt_row_strings"]
    pred_row_dicts = context["pred_row_dicts"]
    gt_row_dicts = context["gt_row_dicts"]
    coverage_penalty = context["coverage_penalty"]
    model_name = context["embedding_model"]

    # Init embedding model and get row counts
    embedding_model = ModelManager.create_model(
        model_provider=ModelProvider.OPENAI,
        model_type=ModelType.EMBEDDING,
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY", None),
    )

    P, G = len(pred_row_strings), len(gt_row_strings)

    # Generate embeddings
    pred_embeddings = np.zeros((0, 0), dtype=np.float32)
    gt_embeddings = np.zeros((0, 0), dtype=np.float32)

    if pred_row_strings and gt_row_strings:
        # Embed predicted and ground truth rows
        pred_embeddings_list = []
        for row_string in pred_row_strings:
            pred_embeddings_list.append(embedding_model.get_embedding(row_string))
        pred_embeddings = np.array(pred_embeddings_list, dtype=np.float32)

        gt_embeddings_list = []
        for row_string in gt_row_strings:
            gt_embeddings_list.append(embedding_model.get_embedding(row_string))
        gt_embeddings = np.array(gt_embeddings_list, dtype=np.float32)

    # Calculate similarity matrix
    similarity_matrix = np.zeros((P, G), dtype=np.float32)
    if P > 0 and G > 0:
        # Compute cosine similarity
        pred_norms = np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8
        gt_norms = np.linalg.norm(gt_embeddings, axis=1, keepdims=True) + 1e-8
        similarity_matrix = (pred_embeddings @ gt_embeddings.T) / (
            pred_norms @ gt_norms.T
        )

        # Set exact matches to 1.0
        for p in range(P):
            for g in range(G):
                if pred_row_strings[p] == gt_row_strings[g]:
                    similarity_matrix[p, g] = 1.0

    # Perform bipartite matching
    matched_pairs, unmatched_pred, unmatched_gt = [], set(), set()
    if similarity_matrix.size > 0:
        cost_matrix = -similarity_matrix
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(pred_indices, gt_indices))

        # Identify unmatched rows
        all_pred = set(range(similarity_matrix.shape[0]))
        all_gt = set(range(similarity_matrix.shape[1]))
        matched_pred = set(pred_indices)
        matched_gt = set(gt_indices)

        unmatched_pred = all_pred - matched_pred
        unmatched_gt = all_gt - matched_gt

    # Create match records and calculate similarity
    matched_row_details = []
    sum_matched_sim = 0.0

    for p_idx, g_idx in matched_pairs:
        raw_sim = similarity_matrix[p_idx, g_idx]
        penalized_sim = raw_sim * coverage_penalty
        sum_matched_sim += penalized_sim

        match_detail = {
            "predicted_row_index": int(p_idx),
            "ground_truth_row_index": int(g_idx),
            "raw_similarity": float(raw_sim),
            "penalized_similarity": float(penalized_sim),
            "predicted_row": pred_row_dicts[p_idx],
            "ground_truth_row": gt_row_dicts[g_idx],
            "predicted_row_string": pred_row_strings[p_idx],
            "ground_truth_row_string": gt_row_strings[g_idx],
            "is_exact_match": pred_row_strings[p_idx] == gt_row_strings[g_idx],
        }
        matched_row_details.append(match_detail)

    # Generate details for unmatched rows
    unmatched_pred_details = [
        {
            "row_index": int(idx),
            "row": pred_row_dicts[idx],
            "row_string": pred_row_strings[idx],
        }
        for idx in unmatched_pred
    ]

    unmatched_gt_details = [
        {
            "row_index": int(idx),
            "row": gt_row_dicts[idx],
            "row_string": gt_row_strings[idx],
        }
        for idx in unmatched_gt
    ]

    # Sort matched rows by similarity
    matched_row_details.sort(key=lambda x: x["penalized_similarity"], reverse=True)

    context.update(
        {
            "P": P,
            "G": G,
            "matched_row_details": matched_row_details,
            "unmatched_pred_details": unmatched_pred_details,
            "unmatched_gt_details": unmatched_gt_details,
            "sum_matched_sim": sum_matched_sim,
        }
    )


def assign_metrics(context):
    """
    Calculate final metrics based on matching results.
    Computes precision, recall, F1 score and handles edge cases for empty result sets.
    """
    if context.get("has_error", False):
        return

    sum_matched_sim = context["sum_matched_sim"]
    P = context["P"]
    G = context["G"]
    ex = context["ex"]
    start_time = context["start_time"]

    # Calculate time taken
    context["time_taken"] = float(time.time() - start_time)

    # Handle edge cases
    if P == 0 and G == 0:
        # Both result sets empty
        metrics = {"EXP": 1.0, "EXR": 1.0, "F1": 1.0, "EX": ex}
    elif P == 0:
        # No predicted rows but ground truth has rows
        metrics = {"EXP": 0.0, "EXR": 0.0, "F1": 0.0, "EX": ex}
    elif G == 0:
        # Has predicted rows but no ground truth
        metrics = {"EXP": 0.0, "EXR": 0.0, "F1": 0.0, "EX": ex}
    else:
        # Calculate metrics
        EXP = float(sum_matched_sim / P) if P > 0 else 0.0
        EXR = float(sum_matched_sim / G) if G > 0 else 0.0
        F1 = float(2 * (EXP * EXR) / (EXP + EXR)) if (EXP + EXR) > 0 else 0.0

        metrics = {"EXP": EXP, "EXR": EXR, "F1": F1, "EX": ex}

    logger.log(
        "INFO",
        "EVALUATION_COMPLETE",
        {
            "EXP": metrics.get("EXP", 0.0),
            "EXR": metrics.get("EXR", 0.0),
            "F1": metrics.get("F1", 0.0),
            "EX": metrics.get("EX", 0),
            "time_taken": context["time_taken"],
        },
    )

    context["metrics"] = metrics


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


def run_eval_pipeline(
    predicted_sql: str,
    ground_truth_sql: str,
    db_params: Dict,
    embedding_model: OpenAIModel | OllamaModel | HuggingFaceModel,
    log_file_dir: str,
):
    """
    Run the complete SQL evaluation pipeline. Coordinates the four stages of evaluation and returns the context with all results.
    """
    context = {
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "db_params": db_params,
        "embedding_model": embedding_model,
        "start_time": time.time(),
        "has_error": False,
    }

    execute_query(context)
    match_columns(context)
    match_rows(context)
    assign_metrics(context)

    dump_logs(context, log_file_dir)


if __name__ == "__main__":
    # Example usage
    predicted_sql = """
    SELECT T3.Phone, T3.City, T3.State, T3.MailStreet
    FROM satscores T1 
    JOIN schools T3 ON T1.cds = T3.CDSCode 
    WHERE T1.NumTstTakr IS NOT NULL AND T1.NumGE1500 IS NOT NULL 
    ORDER BY (T1.NumGE1500 * 1.0 / T1.NumTstTakr) DESC 
    LIMIT 20;
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
    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL
    log_file_dir = ".data/evaluation_metrics_logs/_1_row_semantic_matcher/"

    run_eval_pipeline(
        predicted_sql=predicted_sql,
        ground_truth_sql=ground_truth_sql,
        db_params=db_params,
        embedding_model=embedding_model,
        log_file_dir=log_file_dir,
    )

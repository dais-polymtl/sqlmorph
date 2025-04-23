import os
import time
from typing import Any

import numpy as np
import scipy

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
    try:
        db_handler = DatabaseHandler(
            dbms=context["db_params"]["dbms"], connection_params=context["db_params"]
        )

        pred_cols, pred_rows = db_handler.run_query(context["predicted_sql"])
        gt_cols, gt_rows = db_handler.run_query(context["ground_truth_sql"])

        context.update(
            {
                "pred_cols": pred_cols,
                "pred_rows": pred_rows,
                "gt_cols": gt_cols,
                "gt_rows": gt_rows,
                "has_error": False,
            }
        )

    except Exception as e:
        logger.log("error", "QUERY_EXECUTION_FAILED", {"error": str(e)})
        context.update(
            {
                "has_error": True,
                "error_message": str(e),
                "metrics": {"EXP": 0.0, "EXR": 0.0, "F1": 0.0, "EX": 0},
            }
        )


def match_columns(context):
    if context["has_error"]:
        return context

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
    if context["has_error"]:
        return context

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
        pred_indices, gt_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
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
    if context["has_error"]:
        return context

    sum_matched_sim = context["sum_matched_sim"]
    P = context["P"]
    G = context["G"]

    # Handle edge cases
    if P == 0 and G == 0:
        # Both result sets empty
        metrics = {"EX": 1, "EXP": 1.0, "EXR": 1.0, "F1": 1.0}
    elif P == 0:
        # No predicted rows but ground truth has rows
        metrics = {"EX": 0, "EXP": 0.0, "EXR": 0.0, "F1": 0.0}
    elif G == 0:
        # Has predicted rows but no ground truth
        metrics = {"EX": 0, "EXP": 0.0, "EXR": 0.0, "F1": 0.0}
    else:
        EX = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0
        EXP = float(sum_matched_sim / P) if P > 0 else 0.0
        EXR = float(sum_matched_sim / G) if G > 0 else 0.0
        F1 = float(2 * (EXP * EXR) / (EXP + EXR)) if (EXP + EXR) > 0 else 0.0
        metrics = {"EX": EX, "EXP": EXP, "EXR": EXR, "F1": F1}

    context["metrics"] = metrics

    logger.log("INFO", "EVALUATION_COMPLETE", {"METRICS": metrics})


def run_eval_pipeline(
    predicted_sql: str,
    ground_truth_sql: str,
    db_params: dict[str, Any],
    embedding_model: OpenAIModel | OllamaModel | HuggingFaceModel,
):
    context = {
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "db_params": db_params,
        "embedding_model": embedding_model,
        "metrics": {},
        "has_error": False,
    }

    start_time = time.time()

    execute_query(context)
    match_columns(context)
    match_rows(context)
    assign_metrics(context)

    context["latency"] = time.time() - start_time

    return context


if __name__ == "__main__":
    # ad-hoc example to test the evaluation technique! check out evaluation_metrics.py for the main entry point

    # input data
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

    # config
    db_params = {
        "dbms": DBMS.SQLITE,
        "db_path": "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite",
    }
    embedding_model = OpenAIModel.TEXT_EMBEDDING_3_SMALL

    # run evaluation pipeline
    context = run_eval_pipeline(
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

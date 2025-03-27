import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Set, Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from utils.embedding_calculator import EmbeddingCalculator
from utils.logger import Logger, make_json_serializable
from utils.query_executor import QueryExecutor, SQLiteQueryExecutor

logger = Logger(__name__)


class BipartiteMatcher:
    """Performs bipartite matching (Hungarian algorithm) on a similarity matrix."""

    def match(self, similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
        """Find the optimal 1-to-1 matching that maximizes total similarity."""
        if similarity_matrix.size == 0:
            return [], set(), set()

        # Hungarian algorithm expects cost matrix (minimize cost)
        cost_matrix = -similarity_matrix
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(pred_indices, gt_indices))

        all_pred = set(range(similarity_matrix.shape[0]))
        all_gt = set(range(similarity_matrix.shape[1]))
        matched_pred = set(pred_indices)
        matched_gt = set(gt_indices)

        return matched_pairs, all_pred - matched_pred, all_gt - matched_gt


class SQLResultEvaluator:
    def __init__(
            self,
            query_executor: QueryExecutor,
            embedding_model_name: str = "text-embedding-ada-002"
    ):
        self.query_executor = query_executor
        self.embedding_calculator = EmbeddingCalculator(model_name=embedding_model_name)
        self.matcher = BipartiteMatcher()
        logger.log("debug", "SQL_RESULT_EVALUATOR_INITIALIZED", {"embedding_model": embedding_model_name})

    def _build_row_representations(self, cols: List[str], rows: List[Tuple], all_cols: List[str]) -> Tuple[
        List[str], List[Dict[str, Any]]]:
        """Build row string and dictionary representations based on the column union."""
        row_strings = []
        row_dicts = []

        for row in rows:
            row_dict = dict(zip(cols, row))
            final_values = []
            final_dict = {}

            for col in all_cols:
                value = row_dict.get(col, "N/A")
                final_values.append(str(value))
                final_dict[col] = value

            row_strings.append(" | ".join(final_values))
            row_dicts.append(final_dict)

        return row_strings, row_dicts

    def embed_rows(self, row_strings: List[str]) -> np.ndarray:
        """Generate embeddings for row strings."""
        if not row_strings:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings_lists = self.embedding_calculator.get_batch_embeddings(row_strings)
        return np.array(embeddings_lists, dtype=np.float32) if embeddings_lists else np.zeros((0, 0), dtype=np.float32)

    def _get_zero_result(self, start_time: float, error_message: Optional[str] = None) -> Dict[str, Any]:
        """Return zero-valued results when evaluation fails."""
        result = {
            "EXP": 0.0,
            "EXR": 0.0,
            "F1": 0.0,
            "EX": 0,
            "time_taken": float(time.time() - start_time),
            "matched_rows": [],
            "unmatched_predicted_rows": [],
            "unmatched_ground_truth_rows": []
        }
        if error_message:
            result["error"] = error_message
        return result

    def evaluate(self, predicted_sql: str, ground_truth_sql: str) -> Dict[str, Any]:
        """Evaluate predicted SQL query results against ground truth SQL query results."""
        start_time = time.time()
        logger.log("debug", "EVALUATION_STARTED", {
            "predicted_sql_length": len(predicted_sql),
            "ground_truth_sql_length": len(ground_truth_sql)
        })

        # 1. Execute both queries with combined error handling
        try:
            pred_cols, pred_rows = self.query_executor.execute_query(predicted_sql)
            gt_cols, gt_rows = self.query_executor.execute_query(ground_truth_sql)
        except Exception as e:
            error_type = "QUERY_EXECUTION_FAILED"
            error_message = f"SQL execution failed: {str(e)}"
            logger.log("error", error_type, {"error": str(e)})
            return self._get_zero_result(start_time, error_message)

        # 2. Compute column union and build row representations
        all_cols = sorted(set(pred_cols).union(set(gt_cols)))
        predicted_row_strings, predicted_row_dicts = self._build_row_representations(pred_cols, pred_rows, all_cols)
        gt_row_strings, gt_row_dicts = self._build_row_representations(gt_cols, gt_rows, all_cols)

        # 3. Compute binary execution accuracy (EX)
        ex = 1 if sorted(predicted_row_strings) == sorted(gt_row_strings) else 0

        # 4. Handle edge cases for empty result sets
        P, G = len(pred_rows), len(gt_rows)

        if P == 0 and G == 0:
            return {
                "EXP": 1.0,
                "EXR": 1.0,
                "F1": 1.0,
                "EX": ex,
                "time_taken": float(time.time() - start_time),
                "matched_rows": []
            }
        elif P == 0:
            return {
                "EXP": 0.0,
                "EXR": 0.0,
                "F1": 0.0,
                "EX": ex,
                "time_taken": float(time.time() - start_time),
                "matched_rows": [],
                "unmatched_predicted_rows": [],
                "unmatched_ground_truth_rows": [
                    {"row_index": i, "row": row, "row_string": gt_row_strings[i]}
                    for i, row in enumerate(gt_row_dicts)
                ]
            }
        elif G == 0:
            return {
                "EXP": 0.0,
                "EXR": 0.0,
                "F1": 0.0,
                "EX": ex,
                "time_taken": float(time.time() - start_time),
                "matched_rows": [],
                "unmatched_predicted_rows": [
                    {"row_index": i, "row": row, "row_string": predicted_row_strings[i]}
                    for i, row in enumerate(predicted_row_dicts)
                ],
                "unmatched_ground_truth_rows": []
            }

        # 5. Compute Column Coverage Penalty
        gt_set = set(gt_cols)
        coverage_penalty = len(set(pred_cols).intersection(gt_set)) / len(gt_set) if gt_set else 1.0

        # 6. Embed rows and compute similarity
        pred_embeddings = self.embed_rows(predicted_row_strings)
        gt_embeddings = self.embed_rows(gt_row_strings)

        # Calculate cosine similarity matrix efficiently
        similarity_matrix = np.zeros((P, G), dtype=np.float32)
        if P > 0 and G > 0:
            pred_norms = np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8
            gt_norms = np.linalg.norm(gt_embeddings, axis=1, keepdims=True) + 1e-8
            similarity_matrix = (pred_embeddings @ gt_embeddings.T) / (pred_norms @ gt_norms.T)

            # Set exact matches to 1.0 to avoid floating-point precision issues
            for p in range(P):
                for g in range(G):
                    if predicted_row_strings[p] == gt_row_strings[g]:
                        similarity_matrix[p, g] = 1.0

        # 7. Perform Bipartite Matching and calculate metrics
        matched_pairs, unmatched_pred, unmatched_gt = self.matcher.match(similarity_matrix)

        # 8. Create detailed match records and calculate final metrics
        sum_matched_sim = 0.0
        matched_row_details = []

        for p_idx, g_idx in matched_pairs:
            raw_sim = similarity_matrix[p_idx, g_idx]
            penalized_sim = raw_sim * coverage_penalty
            sum_matched_sim += penalized_sim

            match_detail = {
                "predicted_row_index": int(p_idx),
                "ground_truth_row_index": int(g_idx),
                "raw_similarity": float(raw_sim),
                "penalized_similarity": float(penalized_sim),
                "predicted_row": predicted_row_dicts[p_idx],
                "ground_truth_row": gt_row_dicts[g_idx],
                "predicted_row_string": predicted_row_strings[p_idx],
                "ground_truth_row_string": gt_row_strings[g_idx],
                "is_exact_match": predicted_row_strings[p_idx] == gt_row_strings[g_idx]
            }
            matched_row_details.append(match_detail)

        # Generate unmatched row details
        unmatched_pred_details = [
            {"row_index": int(idx), "row": predicted_row_dicts[idx], "row_string": predicted_row_strings[idx]}
            for idx in unmatched_pred
        ]

        unmatched_gt_details = [
            {"row_index": int(idx), "row": gt_row_dicts[idx], "row_string": gt_row_strings[idx]}
            for idx in unmatched_gt
        ]

        # 9. Calculate final metrics
        EXP = float(sum_matched_sim / P) if P > 0 else 0.0
        EXR = float(sum_matched_sim / G) if G > 0 else 0.0
        F1 = float(2 * (EXP * EXR) / (EXP + EXR)) if (EXP + EXR) > 0 else 0.0

        # Sort matched rows by similarity for better readability
        matched_row_details.sort(key=lambda x: x["penalized_similarity"], reverse=True)

        result = {
            "EXP": EXP,
            "EXR": EXR,
            "F1": F1,
            "EX": ex,
            "coverage_penalty": coverage_penalty,
            "time_taken": float(time.time() - start_time),
            "matched_rows": matched_row_details,
            "unmatched_predicted_rows": unmatched_pred_details,
            "unmatched_ground_truth_rows": unmatched_gt_details
        }

        logger.log("INFO", "EVALUATION_COMPLETE", {
            "EXP": EXP,
            "EXR": EXR,
            "F1": F1,
            "EX": ex,
            "coverage_penalty": coverage_penalty,
            "num_matched_rows": len(matched_pairs),
            "num_unmatched_predicted": len(unmatched_pred),
            "num_unmatched_gt": len(unmatched_gt),
            "time_taken": result["time_taken"]
        })

        return result

    def get_dataframes(self, predicted_sql: str, ground_truth_sql: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute SQL queries and return results as DataFrames."""
        try:
            gt_columns, gt_rows = self.query_executor.execute_query(ground_truth_sql)
            pred_columns, pred_rows = self.query_executor.execute_query(predicted_sql)

            gt_df = pd.DataFrame(gt_rows, columns=gt_columns)
            pred_df = pd.DataFrame(pred_rows, columns=pred_columns)

            return gt_df, pred_df
        except Exception as e:
            logger.log("error", "DATAFRAME_CREATION_FAILED", {"error": str(e)})
            return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    db_path = "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite"

    # Example SQL queries
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

    # Initialize evaluator and run evaluation
    executor = SQLiteQueryExecutor(db_path)
    evaluator = SQLResultEvaluator(executor)

    # Run evaluation
    results = evaluator.evaluate(predicted_sql, ground_truth_sql)

    # Get DataFrames for inspection
    gt_df, pred_df = evaluator.get_dataframes(predicted_sql, ground_truth_sql)

    # Display DataFrames
    print("\n================= Ground Truth DataFrame =================")
    print("Shape:", gt_df.shape)
    print(gt_df)
    print("\n================= Predicted DataFrame =================")
    print("Shape:", pred_df.shape)
    print(pred_df)

    # Print evaluation results
    print("\n================= Evaluation Results =================")
    print(f"EX (Binary Execution Accuracy): {results['EX']}")
    print(f"EXP (Execution Precision): {results['EXP']:.4f}")
    print(f"EXR (Execution Recall): {results['EXR']:.4f}")
    print(f"F1 Score: {results['F1']:.4f}")
    print(f"Time taken: {results['time_taken']:.2f} seconds")

    # Display matched rows
    num_matches = len(results['matched_rows'])
    if num_matches > 0:
        print(f"\n================= Matched Rows ({num_matches}) =================")
        for i, match in enumerate(results['matched_rows'][:5], 1):  # Show top 5 matches
            print(f"\nMatch {i} (Similarity: {match['penalized_similarity']:.4f}):")
            print(f"  Predicted ({match['predicted_row_index']}): {match['predicted_row_string']}")
            print(f"  Ground Truth ({match['ground_truth_row_index']}): {match['ground_truth_row_string']}")

        if num_matches > 5:
            print(f"\n... and {num_matches - 5} more matches")

    # Display unmatched rows
    num_unmatched_pred = len(results['unmatched_predicted_rows'])
    if num_unmatched_pred > 0:
        print(f"\n================= Unmatched Predicted Rows ({num_unmatched_pred}) =================")
        for i, row in enumerate(results['unmatched_predicted_rows'][:3], 1):
            print(f"\nUnmatched Predicted {i} (Index: {row['row_index']}):")
            print(f"  {row['row_string']}")
        if num_unmatched_pred > 3:
            print(f"\n... and {num_unmatched_pred - 3} more unmatched predicted rows")

    num_unmatched_gt = len(results['unmatched_ground_truth_rows'])
    if num_unmatched_gt > 0:
        print(f"\n================= Unmatched Ground Truth Rows ({num_unmatched_gt}) =================")
        for i, row in enumerate(results['unmatched_ground_truth_rows'][:3], 1):
            print(f"\nUnmatched Ground Truth {i} (Index: {row['row_index']}):")
            print(f"  {row['row_string']}")
        if num_unmatched_gt > 3:
            print(f"\n... and {num_unmatched_gt - 3} more unmatched ground truth rows")

    # Save detailed results to JSON file
    log_filename = logger.get_log_filename("evaluation_results")
    output_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": {
            "EX": results["EX"],
            "EXP": results["EXP"],
            "EXR": results["EXR"],
            "F1": results["F1"],
            "coverage_penalty": results.get("coverage_penalty", 1.0),
            "time_taken": results["time_taken"]
        },
        "ground_truth_sql": ground_truth_sql,
        "predicted_sql": predicted_sql,
        "ground_truth_dataframe": gt_df.to_dict(orient="records"),
        "predicted_dataframe": pred_df.to_dict(orient="records"),
        "matching_details": {
            "matched_rows": results["matched_rows"],
            "unmatched_predicted_rows": results["unmatched_predicted_rows"],
            "unmatched_ground_truth_rows": results["unmatched_ground_truth_rows"]
        }
    }
    with open(log_filename, 'w') as f:
        json.dump(make_json_serializable(output_data), f, indent=2)
    print(f"\nDetailed results saved to: {log_filename}")

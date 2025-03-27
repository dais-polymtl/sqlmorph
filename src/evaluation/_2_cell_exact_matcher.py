import time
from collections import Counter

import pandas as pd

from utils.logger import Logger
from utils.query_executor import QueryExecutor, SQLiteQueryExecutor

logger = Logger(__name__)


class SQLResultEvaluator:
    def __init__(self, query_executor: QueryExecutor):
        self.query_executor = query_executor

    def evaluate(self, predicted_sql: str, ground_truth_sql: str):
        """Evaluate predicted SQL query results against ground truth SQL query results."""
        start_time = time.time()

        # 1. Execute queries
        gt_columns, gt_rows = self.query_executor.execute_query(ground_truth_sql)
        pred_columns, pred_rows = self.query_executor.execute_query(predicted_sql)

        # Check for query execution failure - return zeros for all metrics if either query failed
        if not gt_columns or not pred_columns:
            logger.log("error", "QUERY_EXECUTION_FAILED",
                       {"GROUND_TRUTH_COLUMNS": len(gt_columns), "PREDICTED_COLUMNS": len(pred_columns)})
            return {
                "EX": 0,
                "EXP": 0.0,
                "EXR": 0.0,
                "F1": 0.0,
                "matched_cells": 0,
                "ground_truth_cells": len(gt_rows) * len(gt_columns) if gt_columns else 0,
                "predicted_cells": len(pred_rows) * len(pred_columns) if pred_columns else 0,
                "time_taken": time.time() - start_time
            }

        # Binary execution accuracy (EX)
        # Only true if columns and rows match exactly (as sets)
        ex = 1 if (set(gt_columns) == set(pred_columns) and set(gt_rows) == set(pred_rows)) else 0

        # 2. Find intersection of columns c(q^) ∩ c(q)
        common_cols = set(gt_columns) & set(pred_columns)
        logger.log("debug", "COLUMN_INTERSECTION", {"COMMON_COLUMNS": list(common_cols)})

        # 3. Project rows to only include common columns
        # Create mapping from column to index for both datasets
        gt_col_to_idx = {col: idx for idx, col in enumerate(gt_columns)}
        pred_col_to_idx = {col: idx for idx, col in enumerate(pred_columns)}

        # Pre-compute common column indices (optimization)
        gt_common_indices = [gt_col_to_idx[col] for col in common_cols if col in gt_col_to_idx]
        pred_common_indices = [pred_col_to_idx[col] for col in common_cols if col in pred_col_to_idx]

        # Project rows to only include common columns
        gt_projected_rows = []
        for row in gt_rows:
            # Project row to only include common columns
            projected = tuple(row[idx] for idx in gt_common_indices)
            gt_projected_rows.append(projected)

        pred_projected_rows = []
        for row in pred_rows:
            # Project row to only include common columns
            projected = tuple(row[idx] for idx in pred_common_indices)
            pred_projected_rows.append(projected)

        # Calculate total cells
        g_rows = len(gt_rows)
        p_rows = len(pred_rows)

        # Total cells in ground truth: |Grows| × |c(q)|
        g_cells = g_rows * len(gt_columns)

        # Total cells in predicted results (only for common columns): |Prows| × |c(q^) ∩ c(q)|
        p_cells = p_rows * len(common_cols)

        # 4. Count frequencies of projected rows in both datasets
        gt_counter = Counter(gt_projected_rows)
        pred_counter = Counter(pred_projected_rows)

        # 5. Calculate matched cells: |MatchedRows| × |c(q^) ∩ c(q)|
        # For each unique row, take minimum frequency from both sides
        matched_rows = 0
        for row_tuple in set(gt_counter) & set(pred_counter):  # Iterate over common row patterns
            # Take minimum count of this row pattern between GT and prediction
            min_count = min(gt_counter[row_tuple], pred_counter[row_tuple])
            matched_rows += min_count

        # Total matched cells is matched rows times number of common columns
        matched_cells = matched_rows * len(common_cols)

        # Handle empty result cases
        if g_cells == 0 and p_cells == 0:
            return {
                "EX": 1,  # Exact match (both empty)
                "EXP": 1.0,  # Both empty, perfect precision
                "EXR": 1.0,  # Both empty, perfect recall
                "F1": 1.0,  # Perfect F1 score
                "matched_cells": 0,
                "ground_truth_cells": 0,
                "predicted_cells": 0,
                "time_taken": time.time() - start_time
            }
        elif g_cells == 0:
            return {
                "EX": 0,  # Not an exact match
                "EXP": 0.0,  # Predicted something when nothing expected
                "EXR": 1.0,  # No ground truth to recall
                "F1": 0.0,  # Zero F1 score
                "matched_cells": 0,
                "ground_truth_cells": 0,
                "predicted_cells": p_cells,
                "time_taken": time.time() - start_time
            }
        elif p_cells == 0:
            return {
                "EX": 0,  # Not an exact match
                "EXP": 1.0,  # No prediction to be wrong
                "EXR": 0.0,  # Failed to recall anything
                "F1": 0.0,  # Zero F1 score
                "matched_cells": 0,
                "ground_truth_cells": g_cells,
                "predicted_cells": 0,
                "time_taken": time.time() - start_time
            }

        # 6. Calculate metrics
        exp = matched_cells / p_cells if p_cells > 0 else 0.0  # Execution Precision
        exr = matched_cells / g_cells if g_cells > 0 else 0.0  # Execution Recall
        f1 = 2 * exp * exr / (exp + exr) if (exp + exr) > 0 else 0.0  # F1 Score

        logger.log("info", "EVALUATION_COMPLETE", {
            "EX": ex,
            "EXP": exp,
            "EXR": exr,
            "F1": f1,
            "MATCHED_CELLS": matched_cells,
            "GROUND_TRUTH_CELLS": g_cells,
            "PREDICTED_CELLS": p_cells
        })

        return {
            "EX": ex,
            "EXP": exp,
            "EXR": exr,
            "F1": f1,
            "matched_cells": matched_cells,
            "ground_truth_cells": g_cells,
            "predicted_cells": p_cells,
            "time_taken": time.time() - start_time
        }

    def get_dataframes(self, predicted_sql: str, ground_truth_sql: str):
        """
        Execute both SQL queries and return the results as pandas DataFrames for inspection.
        """
        gt_columns, gt_rows = self.query_executor.execute_query(ground_truth_sql)
        pred_columns, pred_rows = self.query_executor.execute_query(predicted_sql)

        # Create DataFrames
        gt_df = pd.DataFrame(gt_rows, columns=gt_columns)
        pred_df = pd.DataFrame(pred_rows, columns=pred_columns)

        return gt_df, pred_df

if __name__ == "__main__":
    # Example usage
    db_path = "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite"
    executor = SQLiteQueryExecutor(db_path)
    evaluator = SQLResultEvaluator(executor)

    predicted_sql = """
    SELECT T3.Phone
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

    # Get DataFrames for inspection
    gt_df, pred_df = evaluator.get_dataframes(predicted_sql, ground_truth_sql)

    print("\n================= Ground Truth DataFrame =================")
    print("\nGround Truth DataFrame Shape:", gt_df.shape)
    print(gt_df)
    print("\n================= Predicted DataFrame =================")
    print("Predicted DataFrame Shape:", pred_df.shape)
    print(pred_df)

    # Run evaluation
    results = evaluator.evaluate(predicted_sql, ground_truth_sql)

    print("\n================= Evaluation Results =================")
    print("EX (Binary Execution Accuracy):", results["EX"])
    print("EXP (Execution Precision):", results["EXP"])
    print("EXR (Execution Recall):", results["EXR"])
    print("F1 Score:", results["F1"])
    print("Matched Cells:", results["matched_cells"])
    print("Ground Truth Cells:", results["ground_truth_cells"])
    print("Predicted Cells:", results["predicted_cells"])
    print("Time taken (seconds):", results["time_taken"])

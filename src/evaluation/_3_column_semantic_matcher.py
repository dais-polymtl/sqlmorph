import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from utils.embedding_calculator import EmbeddingCalculator
from utils.logger import Logger
from utils.query_executor import QueryExecutor, SQLiteQueryExecutor

logger = Logger(__name__)


class SQLResultEvaluator:
    def __init__(self, query_executor: QueryExecutor):
        self.query_executor = query_executor
        self.embedding_calculator = EmbeddingCalculator()

    def evaluate(self, predicted_sql: str, ground_truth_sql: str):
        """
        Evaluate predicted SQL query results against ground truth using cell-level precision
        and recall with semantic column alignment.
        """
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

        # Create dataframes for easier data processing
        gt_df = pd.DataFrame(gt_rows, columns=gt_columns)
        pred_df = pd.DataFrame(pred_rows, columns=pred_columns)

        # 2. Preprocessing: Extract and prepare column data
        gt_col_data = self._extract_column_data(gt_df)
        pred_col_data = self._extract_column_data(pred_df)

        # 3. Generate column embeddings
        gt_embeddings = self._get_column_embeddings(gt_col_data)
        pred_embeddings = self._get_column_embeddings(pred_col_data)

        # 4. Compute column similarity matrix
        similarity_matrix = self._compute_similarity_matrix(gt_embeddings, pred_embeddings)

        # 5. Perform bipartite matching to find optimal column pairs
        col_matches = self._bipartite_matcher(similarity_matrix)

        # Get the matched column pairs
        matched_columns = []
        for pred_idx, gt_idx in enumerate(col_matches):
            if gt_idx >= 0 and similarity_matrix[pred_idx, gt_idx] > 0.7:  # Only consider sufficiently similar columns
                matched_columns.append((pred_columns[pred_idx], gt_columns[gt_idx]))

        logger.log("debug", "COLUMN_MATCHING", {"MATCHED_COLUMNS": matched_columns})

        # 6. Project rows to only include matched columns
        # Create mapping for matched columns
        gt_col_to_pred_col = {gt_col: pred_col for pred_col, gt_col in matched_columns}

        # Get indices for matched columns
        gt_matched_indices = [idx for idx, col in enumerate(gt_columns) if
                              col in [gt_col for _, gt_col in matched_columns]]
        pred_matched_indices = [idx for idx, col in enumerate(pred_columns) if
                                col in [pred_col for pred_col, _ in matched_columns]]

        # Project rows to only include matched columns
        gt_projected_rows = []
        for row in gt_rows:
            # Project row to only include matched columns
            projected = tuple(row[idx] for idx in gt_matched_indices)
            gt_projected_rows.append(projected)

        pred_projected_rows = []
        for row in pred_rows:
            # Project row to only include matched columns
            projected = tuple(row[idx] for idx in pred_matched_indices)
            pred_projected_rows.append(projected)

        # Calculate total cells
        g_rows = len(gt_rows)
        p_rows = len(pred_rows)

        # Total cells in ground truth
        g_cells = g_rows * len(gt_columns)

        # Total cells in predicted results
        p_cells = p_rows * len(pred_columns)

        # Number of matched columns
        num_matched_columns = len(matched_columns)

        # 7. Count frequencies of projected rows in both datasets
        gt_counter = Counter(gt_projected_rows)
        pred_counter = Counter(pred_projected_rows)

        # 8. Calculate matched cells: |MatchedRows| × |MatchedColumns|
        # For each unique row, take minimum frequency from both sides
        matched_rows = 0
        for row_tuple in set(gt_counter) & set(pred_counter):  # Iterate over common row patterns
            # Take minimum count of this row pattern between GT and prediction
            min_count = min(gt_counter[row_tuple], pred_counter[row_tuple])
            matched_rows += min_count

        # Total matched cells is matched rows times number of matched columns
        matched_cells = matched_rows * num_matched_columns

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

        # 9. Calculate metrics
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

    def _extract_column_data(self, df):
        """Extract column names and top frequent values for embedding generation"""
        column_data = {}

        for col in df.columns:
            # Get data type for the column
            data_type = str(df[col].dtype)

            # Get top frequent values (up to 10)
            if len(df) > 0:
                value_counts = df[col].value_counts().head(10)
                top_values = [str(v) for v in value_counts.index]
            else:
                top_values = []

            column_data[col] = {
                'name': col,
                'data_type': data_type,
                'top_values': top_values
            }

        return column_data

    def _get_column_embeddings(self, column_data):
        """Generate embeddings for columns based on name and top values"""
        embeddings = {}

        for col, data in column_data.items():
            # Combine column name and top values into a single text representation
            text = f"Column name: {data['name']}. Data type: {data['data_type']}. "

            if data['top_values']:
                text += f"Sample values: {', '.join(data['top_values'])}"

            # Generate embedding
            embedding = self.embedding_calculator.get_text_embedding(text)
            embeddings[col] = embedding

        return embeddings

    def _compute_similarity_matrix(self, gt_embeddings, pred_embeddings):
        """Compute similarity matrix between ground truth and predicted columns"""
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

        return similarity_matrix

    def _bipartite_matcher(self, similarity_matrix):
        """Use the Hungarian algorithm to find optimal matching between columns"""
        # Convert similarities to costs for the linear_sum_assignment algorithm
        # (which minimizes total cost, so we negate the similarities)
        cost_matrix = -similarity_matrix

        # Get row_ind, col_ind using scipy's optimized implementation
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create a mapping from predicted column index to ground truth column index
        matches = col_ind

        return matches

    def get_dataframes(self, predicted_sql: str, ground_truth_sql: str):
        """Execute both SQL queries and return the results as pandas DataFrames for inspection."""
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

import os
import sqlite3
import time

import numpy as np
from openai import OpenAI
from scipy.optimize import linear_sum_assignment


class QueryExecutor:
    """
    Abstract base class for different SQL dialect executors.
    """
    def execute_query(self, query: str):
        """
        Execute a query and return (column_names, rows).

        column_names: list of str
        rows: list of tuples (each tuple is a row)
        """
        raise NotImplementedError("Subclasses should implement this method.")

class SQLiteQueryExecutor(QueryExecutor):
    """
    Concrete implementation for SQLite.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute_query(self, query: str):
        """
        Execute a query against the SQLite database.
        Returns (column_names, rows).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        except Exception as e:
            print(f"[ERROR] Failed to execute query:\n{query}\n{e}")
            column_names, rows = [], []
        finally:
            cursor.close()
            conn.close()
        return column_names, rows

class EmbeddingCalculator:
    """
    Uses an OpenAI client to produce embeddings for row text.
    """
    def __init__(
        self,
        model_name="text-embedding-ada-002",
        encoding_format=None,
        dimensions=None,
        user=None
    ):
        """
        :param model_name: e.g. "text-embedding-ada-002" or "text-embedding-3-small"
        :param encoding_format: optional param for OpenAI embeddings
        :param dimensions: optional param for OpenAI embeddings
        :param user: optional param for OpenAI embeddings
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.user = user

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Returns a NumPy vector of floats for the embedding.
        """
        payload = {
            "model": self.model_name,
            "input": [text],
        }
        optional_params = {
            "encoding_format": self.encoding_format,
            "dimensions": self.dimensions,
            "user": self.user,
        }
        for key, value in optional_params.items():
            if value is not None:
                payload[key] = value
        try:
            response = self.client.embeddings.create(**payload)
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to get embedding for text: '{text}'\n{e}")
            return np.zeros(1536, dtype=np.float32)

    def embed_rows(self, row_strings) -> np.ndarray:
        """
        Given a list of row strings (with placeholders if needed),
        produce embeddings for each row string. Return a 2D array [num_rows x embedding_dim].
        """
        embeddings = []
        for row_str in row_strings:
            emb = self.get_text_embedding(row_str)
            embeddings.append(emb)
        if len(embeddings) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        else:
            return np.vstack(embeddings)

class BipartiteMatcher:
    """
    Performs bipartite matching (Hungarian algorithm) on a similarity matrix.
    """
    def match(self, similarity_matrix: np.ndarray):
        """
        Given a (P x G) similarity matrix, find the optimal 1-to-1 matching
        that maximizes total similarity.

        Returns:
          matched_pairs: list of (pred_idx, gt_idx) for all pairs (no threshold filtering)
          unmatched_pred_indices: set of predicted row indices not matched
          unmatched_gt_indices: set of ground-truth row indices not matched
        """
        if similarity_matrix.size == 0:
            return [], set(), set()
        cost_matrix = -similarity_matrix
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        # Remove threshold check: include all matches
        matched_pairs = [(p, g) for p, g in zip(pred_indices, gt_indices)]
        all_pred = set(range(similarity_matrix.shape[0]))
        all_gt = set(range(similarity_matrix.shape[1]))
        matched_pred = set(p for p, _ in matched_pairs)
        matched_gt = set(g for _, g in matched_pairs)
        unmatched_pred_indices = all_pred - matched_pred
        unmatched_gt_indices = all_gt - matched_gt
        return matched_pairs, unmatched_pred_indices, unmatched_gt_indices

class SQLResultEvaluator:
    """
    Evaluates predicted vs ground-truth SQL queries at the row level,
    penalizing missing/extra columns by including them in the row representation
    with placeholders ("N/A").

    The final metrics are:
      - EXP (execution precision): sum of matched similarities / (# predicted rows)
      - EXR (execution recall): sum of matched similarities / (# ground-truth rows)
      - EX (binary execution accuracy): 1 if the predicted result set exactly equals the ground truth (ignoring order), else 0.
    """
    def __init__(
        self,
        query_executor: QueryExecutor,
        embedding_calculator: EmbeddingCalculator,
        matcher: BipartiteMatcher
    ):
        self.query_executor = query_executor
        self.embedding_calculator = embedding_calculator
        self.matcher = matcher

    def _build_row_strings(self, cols, rows, all_cols):
        """Helper method to build a list of row strings based on the union of columns."""
        row_strings = []
        for row in rows:
            row_dict = {c: v for c, v in zip(cols, row)}
            final_values = []
            for col in all_cols:
                if col in row_dict:
                    final_values.append(str(row_dict[col]))
                else:
                    final_values.append("N/A")
            row_strings.append(" | ".join(final_values))
        return row_strings

    def evaluate(self, predicted_sql: str, ground_truth_sql: str):
        """
        Execute both queries, compute row-level similarity via embeddings and bipartite matching,
        and return the following metrics in a single result:
          - EXP: execution precision = (sum of matched similarities) / (# predicted rows)
          - EXR: execution recall = (sum of matched similarities) / (# ground-truth rows)
          - EX: binary execution accuracy: 1 if the predicted and ground-truth result sets are exactly equal (ignoring order), otherwise 0.
        Also returns additional matching details.

        Additionally, we explicitly penalize for missing ground-truth columns by computing a column coverage penalty.
        """
        start_time = time.time()

        # 1. Execute queries
        pred_cols, pred_rows = self.query_executor.execute_query(predicted_sql)
        gt_cols, gt_rows = self.query_executor.execute_query(ground_truth_sql)
        all_cols = sorted(set(pred_cols).union(set(gt_cols)))

        # 2. Build row strings for both predicted and ground truth
        predicted_row_strings = self._build_row_strings(pred_cols, pred_rows, all_cols)
        gt_row_strings = self._build_row_strings(gt_cols, gt_rows, all_cols)

        # 3. Compute binary execution accuracy (EX)
        ex = 1 if sorted(predicted_row_strings) == sorted(gt_row_strings) else 0

        # 4. Handle edge cases for graded metrics
        P = len(pred_rows)
        G = len(gt_rows)
        if P == 0 and G == 0:
            return {
                "EXP": 1.0,
                "EXR": 1.0,
                "EX": ex,
                "sum_matched_sim": 0.0,
                "matched_pairs": [],
                "unmatched_pred": [],
                "unmatched_gt": []
            }
        elif P == 0:
            return {
                "EXP": 0.0,
                "EXR": 0.0,
                "EX": ex,
                "sum_matched_sim": 0.0,
                "matched_pairs": [],
                "unmatched_pred": [],
                "unmatched_gt": list(range(G))
            }
        elif G == 0:
            return {
                "EXP": 0.0,
                "EXR": 0.0,
                "EX": ex,
                "sum_matched_sim": 0.0,
                "matched_pairs": [],
                "unmatched_pred": list(range(P)),
                "unmatched_gt": []
            }

        # 5. Compute explicit column coverage penalty
        #    Penalty = (# common columns between predicted and ground truth) / (# ground-truth columns)
        gt_set = set(gt_cols)
        common_cols = set(pred_cols).intersection(gt_set)
        coverage_penalty = len(common_cols) / len(gt_set) if len(gt_set) > 0 else 1.0

        # 6. Embed row strings (without applying penalty yet)
        pred_embeddings = self.embedding_calculator.embed_rows(predicted_row_strings)
        gt_embeddings = self.embedding_calculator.embed_rows(gt_row_strings)

        # 7. Compute similarity matrix (P x G) using cosine similarity
        pred_norms = np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8
        gt_norms = np.linalg.norm(gt_embeddings, axis=1, keepdims=True) + 1e-8
        similarity_matrix = (pred_embeddings @ gt_embeddings.T) / (pred_norms * gt_norms.T)

        # 8. Override similarity to 1.0 if the row strings are exactly identical
        for p in range(P):
            for g in range(G):
                if predicted_row_strings[p] == gt_row_strings[g]:
                    similarity_matrix[p, g] = 1.0

        # 9. Perform bipartite matching without threshold filtering (i.e. threshold=0)
        matched_pairs, unmatched_pred, unmatched_gt = self.matcher.match(similarity_matrix)

        # 10. Apply column coverage penalty to the similarity of matched pairs
        sum_matched_sim = 0.0
        for p_idx, g_idx in matched_pairs:
            sum_matched_sim += similarity_matrix[p_idx, g_idx] * coverage_penalty

        # 11. Calculate graded metrics: EXP and EXR
        EXP = sum_matched_sim / P
        EXR = sum_matched_sim / G

        return {
            "EXP": EXP,
            "EXR": EXR,
            "EX": ex,
            "sum_matched_sim": sum_matched_sim,
            "matched_pairs": matched_pairs,
            "unmatched_pred": list(unmatched_pred),
            "unmatched_gt": list(unmatched_gt),
            "time_taken": time.time() - start_time
        }

if __name__ == "__main__":
    # Example usage

    db_path = "data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite"
    executor = SQLiteQueryExecutor(db_path)
    embed_calc = EmbeddingCalculator(model_name="text-embedding-ada-002")
    matcher = BipartiteMatcher()
    evaluator = SQLResultEvaluator(executor, embed_calc, matcher)

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

    results = evaluator.evaluate(predicted_sql, ground_truth_sql)

    print("EXP (Execution Precision):", results["EXP"])
    print("EXR (Execution Recall):", results["EXR"])
    print("EX (Binary Execution Accuracy):", results["EX"])
    print("Sum of Matched Similarities:", results["sum_matched_sim"])
    print("Matched Pairs:", results["matched_pairs"])
    print("Unmatched Predicted:", results["unmatched_pred"])
    print("Unmatched GroundTruth:", results["unmatched_gt"])
    print("Time taken (seconds):", results["time_taken"])

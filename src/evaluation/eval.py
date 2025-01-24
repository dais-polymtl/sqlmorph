import os
import sqlite3
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
    This version uses `response.data[0].embedding` rather than dictionary indexing.
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
        # Read API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # Create the OpenAI client with the given API key
        self.client = OpenAI(api_key=api_key)

        # Store embedding params
        self.model_name = model_name
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.user = user

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Call OpenAI Embeddings API in 'new style', building a payload with optional params.
        Returns a NumPy vector of floats for the embedding, using response.data[0].embedding.
        """
        payload = {
            "model": self.model_name,
            "input": [text],
        }
        # Add optional params if they are not None
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
            # Use object notation:
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to get embedding for text: '{text}'\n{e}")
            # Fallback: return a zero vector if there's an error
            return np.zeros(1536, dtype=np.float32)  # typical dimension for e.g. text-embedding-ada-002

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
    def match(self, similarity_matrix: np.ndarray, threshold=0.0):
        """
        Given a (P x G) similarity matrix, find the optimal 1-to-1 matching
        that maximizes total similarity.

        Returns:
          matched_pairs: list of (pred_idx, gt_idx) for pairs above 'threshold'
          unmatched_pred_indices: set of predicted row indices not matched
          unmatched_gt_indices: set of ground-truth row indices not matched
        """

        if similarity_matrix.size == 0:
            # Edge case: empty => no matches
            return [], set(), set()

        # Convert similarity to cost: cost = -similarity
        cost_matrix = -similarity_matrix

        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        for p, g in zip(pred_indices, gt_indices):
            sim = similarity_matrix[p, g]
            if sim >= threshold:
                matched_pairs.append((p, g))

        # Identify unmatched
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

    def evaluate(
        self,
        predicted_sql: str,
        ground_truth_sql: str,
        threshold: float = 0.8
    ):
        """
        Execute both queries, compute row-level similarity + bipartite matching,
        then compute precision/recall using the specified threshold.

        We penalize extra/missing columns by building row strings that include
        placeholders for columns not present in that query's result.
        """

        # 1. Execute predicted and ground-truth queries
        pred_cols, pred_rows = self.query_executor.execute_query(predicted_sql)
        gt_cols, gt_rows = self.query_executor.execute_query(ground_truth_sql)

        # 2. Collect ALL columns to handle missing/extra
        all_cols = sorted(set(pred_cols).union(set(gt_cols)))

        # 3. Build row strings for predicted with placeholders
        pred_col_to_idx = {col: i for i, col in enumerate(pred_cols)}
        predicted_row_strings = []
        for row in pred_rows:
            row_dict = {}
            for c, val in zip(pred_cols, row):
                row_dict[c] = val

            final_values = []
            for col in all_cols:
                if col in row_dict:
                    final_values.append(str(row_dict[col]))
                else:
                    final_values.append("N/A")

            predicted_row_strings.append(" | ".join(final_values))

        # 4. Build row strings for ground truth with placeholders
        gt_col_to_idx = {col: i for i, col in enumerate(gt_cols)}
        gt_row_strings = []
        for row in gt_rows:
            row_dict = {}
            for c, val in zip(gt_cols, row):
                row_dict[c] = val

            final_values = []
            for col in all_cols:
                if col in row_dict:
                    final_values.append(str(row_dict[col]))
                else:
                    final_values.append("N/A")

            gt_row_strings.append(" | ".join(final_values))

        # 5. Handle edge cases
        P = len(pred_rows)
        G = len(gt_rows)
        if P == 0 and G == 0:
            # both empty => perfect match
            return {
                "precision": 1.0,
                "recall": 1.0,
                "matched_pairs": [],
                "unmatched_pred": [],
                "unmatched_gt": []
            }
        elif P == 0:
            # predicted empty but ground-truth not => 0
            return {
                "precision": 0.0,
                "recall": 0.0,
                "matched_pairs": [],
                "unmatched_pred": [],
                "unmatched_gt": list(range(G))
            }
        elif G == 0:
            # ground-truth empty but predicted not => 0
            return {
                "precision": 0.0,
                "recall": 0.0,
                "matched_pairs": [],
                "unmatched_pred": list(range(P)),
                "unmatched_gt": []
            }

        # 6. Embed row strings
        pred_embeddings = self.embedding_calculator.embed_rows(predicted_row_strings)
        gt_embeddings = self.embedding_calculator.embed_rows(gt_row_strings)

        # 7. Compute similarity matrix (P x G) with cosine similarity
        pred_norms = np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8
        gt_norms = np.linalg.norm(gt_embeddings, axis=1, keepdims=True) + 1e-8
        similarity_matrix = (pred_embeddings @ gt_embeddings.T) / (pred_norms * gt_norms.T)

        # 8. Perform bipartite matching
        matched_pairs, unmatched_pred, unmatched_gt = self.matcher.match(
            similarity_matrix, threshold=threshold
        )

        # 9. Precision & Recall
        precision = len(matched_pairs) / P
        recall = len(matched_pairs) / G

        return {
            "precision": precision,
            "recall": recall,
            "matched_pairs": matched_pairs,
            "unmatched_pred": list(unmatched_pred),
            "unmatched_gt": list(unmatched_gt),
        }


if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment before running

    db_path = "california_schools.sqlite"

    executor = SQLiteQueryExecutor(db_path)
    embed_calc = EmbeddingCalculator(model_name="text-embedding-ada-002")
    matcher = BipartiteMatcher()
    evaluator = SQLResultEvaluator(executor, embed_calc, matcher)

    # Example queries
    predicted_sql = "SELECT T3.Phone FROM satscores T1 JOIN schools T3 ON T1.cds = T3.CDSCode WHERE T1.NumTstTakr IS NOT NULL AND T1.NumGE1500 IS NOT NULL ORDER BY (T1.NumGE1500 * 1.0 / T1.NumTstTakr) DESC LIMIT 3;"
    ground_truth_sql = "SELECT T1.Phone FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds ORDER BY CAST(T2.NumGE1500 AS REAL) / T2.NumTstTakr DESC LIMIT 3;"

    # Evaluate
    results = evaluator.evaluate(predicted_sql, ground_truth_sql, threshold=0.5)

    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("Matched Pairs:", results["matched_pairs"])
    print("Unmatched Predicted:", results["unmatched_pred"])
    print("Unmatched GroundTruth:", results["unmatched_gt"])

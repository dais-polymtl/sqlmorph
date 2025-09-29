import argparse
import enum
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.metrics import (
    execution_accuracy,
    exact_column_exact_cell,
    exact_column_partial_cell,
    semantic_column_exact_cell,
    semantic_column_partial_cell,
    no_column_partial_cell,
    unified_column_semantic_row,
)
from src.core.database.database_handler import DBMS
from src.core.logger import Logger
from src.core.model_manager import OpenAIModel

logger = Logger(__name__)


class EvaluationTechnique(enum.Enum):
    EXECUTION_ACCURACY = "execution_accuracy"

    EXACT_COLUMN_AND_EXACT_CELL = "exact_column_and_exact_cell"
    EXACT_COLUMN_AND_PARTIAL_CELL = "exact_column_and_partial_cell"
    SEMANTIC_COLUMN_AND_EXACT_CELL = "semantic_column_and_exact_cell"
    SEMANTIC_COLUMN_AND_PARTIAL_CELL = "semantic_column_and_partial_cell"
    NO_COLUMN_AND_PARTIAL_CELL = "no_column_and_partial_cell"
    UNIFIED_COLUMN_AND_SEMANTIC_ROW = "unified_column_and_semantic_row"


class Evaluation:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run_evaluation(
        self,
        predicted_sql: str,
        ground_truth_sql: str,
        log: bool = True,
    ):
        context = None
        if (
            self.config["evaluation_technique"]
            == EvaluationTechnique.EXECUTION_ACCURACY
        ):
            context = execution_accuracy.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL
        ):
            context = exact_column_exact_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                penalize_extra_pred_cols=self.config["penalize_extra_pred_cols"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL
        ):
            context = semantic_column_exact_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                embedding_model=self.config["embedding_model"],
                penalize_extra_pred_cols=self.config["penalize_extra_pred_cols"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.EXACT_COLUMN_AND_PARTIAL_CELL
        ):
            context = exact_column_partial_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                penalize_extra_pred_cols=self.config["penalize_extra_pred_cols"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.SEMANTIC_COLUMN_AND_PARTIAL_CELL
        ):
            context = semantic_column_partial_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                embedding_model=self.config["embedding_model"],
                penalize_extra_pred_cols=self.config["penalize_extra_pred_cols"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.NO_COLUMN_AND_PARTIAL_CELL
        ):
            context = no_column_partial_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW
        ):
            context = unified_column_semantic_row.run_eval_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                embedding_model=self.config["embedding_model"],
            )

        if log:
            self._dump_logs(context)
        return context

    def _dump_logs(self, context):
        def json_serializer(obj):
            # Convert non-serializable objects to strings or other JSON-serializable types
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
            elif isinstance(obj, enum.Enum):
                return obj.value
            return str(obj)  # For any other type, convert to string

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        context = {} if context is None else context
        context.update({"evaluation_timestamp": timestamp})

        technique_dir = os.path.join(
            self.config["logs_dir_path"], self.config["evaluation_technique"].value
        )
        log_filename = os.path.join(technique_dir, f"evaluation-{timestamp}.json")
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        with open(log_filename, "w") as f:
            json.dump(context, f, indent=2, default=json_serializer)

        logger.log(
            "info",
            "EVALUATION_LOGS_SAVED",
            {
                "log_file": log_filename,
                "approach": str(self.config["evaluation_technique"]),
            },
        )


def load_config_from_env():
    """Load configuration from environment variables set by metrics_config.sh"""

    # Get evaluation technique
    eval_technique_str = os.environ.get("EVAL_TECHNIQUE")
    if not eval_technique_str:
        raise ValueError(
            "EVAL_TECHNIQUE environment variable is required. Please source metrics_config.sh"
        )

    try:
        eval_technique = EvaluationTechnique(eval_technique_str)
    except ValueError:
        logger.log("error", "INVALID_EVAL_TECHNIQUE", {"technique": eval_technique_str})
        raise ValueError(f"Invalid evaluation technique: {eval_technique_str}")

    # Get DBMS type
    dbms_str = os.environ.get("DBMS")
    if not dbms_str:
        raise ValueError(
            "DBMS environment variable is required. Please source metrics_config.sh"
        )

    try:
        dbms = getattr(DBMS, dbms_str.upper())
    except AttributeError:
        logger.log("error", "INVALID_DBMS", {"dbms": dbms_str})
        raise ValueError(f"Invalid DBMS: {dbms_str}")

    # Get database path
    db_path = os.environ.get("DB_PATH")
    if not db_path:
        raise ValueError(
            "DB_PATH environment variable is required. Please source metrics_config.sh"
        )

    # Get embedding model
    embedding_model_str = os.environ.get("EMBEDDING_MODEL")
    if not embedding_model_str:
        raise ValueError(
            "EMBEDDING_MODEL environment variable is required. Please source metrics_config.sh"
        )

    try:
        embedding_model = getattr(OpenAIModel, embedding_model_str)
    except AttributeError:
        logger.log("error", "INVALID_EMBEDDING_MODEL", {"model": embedding_model_str})
        raise ValueError(f"Invalid embedding model: {embedding_model_str}")

    # Get logs directory
    logs_dir_path = os.environ.get("LOGS_DIR_PATH")
    if not logs_dir_path:
        raise ValueError(
            "LOGS_DIR_PATH environment variable is required. Please source metrics_config.sh"
        )

    # Get penalize extra columns setting
    penalize_extra_pred_cols_str = os.environ.get("PENALIZE_EXTRA_PRED_COLS", "true")
    penalize_extra_pred_cols = penalize_extra_pred_cols_str.lower() == "true"

    # Get enable log setting
    enable_log_str = os.environ.get("ENABLE_LOG", "false")
    enable_log = enable_log_str.lower() == "true"

    # Build configuration
    config = {
        "evaluation_technique": eval_technique,
        "db_params": {
            "dbms": dbms,
            "db_path": db_path,
        },
        "penalize_extra_pred_cols": penalize_extra_pred_cols,
        "embedding_model": embedding_model,
        "logs_dir_path": logs_dir_path,
        "enable_log": enable_log,
    }

    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SQL Query Evaluation Tool")
    parser.add_argument("--predicted-sql", required=True, help="Predicted SQL query")
    parser.add_argument(
        "--ground-truth-sql", required=True, help="Ground truth SQL query"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Check if CLI arguments are provided
    if len(sys.argv) > 1:
        # CLI mode
        args = parse_args()

        # Load configuration from environment variables (set by metrics_config.sh)
        config = load_config_from_env()

        # Create evaluator
        evaluator = Evaluation(config)

        # Run evaluation
        res = evaluator.run_evaluation(
            predicted_sql=args.predicted_sql,
            ground_truth_sql=args.ground_truth_sql,
            log=config["enable_log"],
        )

        print("Evaluation Results:")
        if res and "metrics" in res:
            print(f"Metrics: {res['metrics']}")
        if res and "latency" in res:
            print(f"Latency: {res['latency']}")

    else:
        # Example usage
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

        db_name = "california_schools"

        config = {
            "evaluation_technique": EvaluationTechnique.SEMANTIC_COLUMN_AND_PARTIAL_CELL,
            "db_params": {
                "dbms": DBMS.SQLITE,
                "db_path": f"data/benchmarks/Bird/dev_databases/{db_name}/{db_name}.sqlite",
            },
            "penalize_extra_pred_cols": True,
            "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
            "logs_dir_path": "data/evaluation_outputs/",
        }

        evaluator = Evaluation(config)
        res = evaluator.run_evaluation(predicted_sql, ground_truth_sql, log=True)

        print("Semantic Evaluation Results:")
        print(f"Metrics: {res['metrics']}, Latency: {res['latency']}")

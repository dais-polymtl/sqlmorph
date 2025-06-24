import enum
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    unified_column_and_semantic_row_matcher,
    exact_column_and_exact_cell_matcher,
    semantic_column_and_exact_cell_matcher,
    execution_accuracy,
)
from src.core.database.database_handler import DBMS
from src.core.logger import Logger
from src.core.model_manager import OpenAIModel

logger = Logger(__name__)


class EvaluationTechnique(enum.Enum):
    EXECUTION_ACCURACY = "execution_accuracy"
    EXACT_COLUMN_AND_EXACT_CELL = "exact_column_and_exact_cell"
    SEMANTIC_COLUMN_AND_EXACT_CELL = "semantic_column_and_exact_cell"
    UNIFIED_COLUMN_AND_SEMANTIC_ROW = (
        "unified_column_and_semantic_row"  # Added new technique
    )


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
            == EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL
        ):
            context = exact_column_and_exact_cell_matcher.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL
        ):
            context = semantic_column_and_exact_cell_matcher.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                embedding_model=self.config["embedding_model"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW
        ):
            context = unified_column_and_semantic_row_matcher.run_eval_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                embedding_model=self.config["embedding_model"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.EXECUTION_ACCURACY
        ):
            context = execution_accuracy.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
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

        for technique in EvaluationTechnique:
            technique_dir = os.path.join(self.config["logs_dir_path"], technique.value)
            os.makedirs(technique_dir, exist_ok=True)
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


if __name__ == "__main__":
    # Example usage
    predicted_sql = """
    SELECT DISTINCT T1.bond_type 
    FROM bond AS T1 
    INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id INNER JOIN atom AS T3 
    WHERE T3.element <> 'cl'
    """

    ground_truth_sql = """
    SELECT DISTINCT T1.bond_type 
    FROM bond AS T1 
    INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id INNER JOIN atom AS T3 ON T2.atom_id = T3.atom_id 
    WHERE T3.element = 'cl'
    """

    # Single config dictionary that works for all techniques
    config = {
        "evaluation_technique": EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": "data/benchmarks/Bird/dev_databases/toxicology/toxicology.sqlite",
        },
        "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        "logs_dir_path": "data/evaluation_outputs/",
    }

    exact_evaluator = Evaluation(config)
    res = exact_evaluator.run_evaluation(
        predicted_sql=predicted_sql, ground_truth_sql=ground_truth_sql, log=False
    )
    print("Semantic Evaluation Results:")
    print(f"Metrics: {res['metrics']}, Latency: {res['latency']}")

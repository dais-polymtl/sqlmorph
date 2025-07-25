import enum
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    unified_column_and_semantic_row,
    exact_column_and_exact_cell,
    semantic_column_and_exact_cell,
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
            == EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL
        ):
            context = exact_column_and_exact_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL
        ):
            context = semantic_column_and_exact_cell.run_evaluation_pipeline(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_params=self.config["db_params"],
                embedding_model=self.config["embedding_model"],
            )
        elif (
            self.config["evaluation_technique"]
            == EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW
        ):
            context = unified_column_and_semantic_row.run_eval_pipeline(
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
    predicted_sql = "SELECT sub.MailStreet, sub.School, sub.MailCity, sub.MailState, sub.FRPM FROM (SELECT T2.MailStreet AS MailStreet, T2.School AS School, T2.MailCity AS MailCity, T2.MailState AS MailState, T1.`FRPM Count (K-12)` AS FRPM, T2.County AS County, T2.District AS District, T2.Zip AS Zip, T2.Phone AS Phone FROM frpm AS T1 JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY FRPM DESC LIMIT 9) AS sub ORDER BY sub.FRPM DESC LIMIT 5"
    ground_truth_sql = "SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1"

    db_name = "california_schools"

    # Single config dictionary that works for all techniques
    config = {
        "evaluation_technique": EvaluationTechnique.EXECUTION_ACCURACY,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": f"data/benchmarks/Bird/dev_databases/{db_name}/{db_name}.sqlite",
        },
        "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        "logs_dir_path": "data/evaluation_outputs/",
    }

    exact_evaluator = Evaluation(config)
    res = exact_evaluator.run_evaluation(
        predicted_sql=predicted_sql, ground_truth_sql=ground_truth_sql, log=True
    )
    print("Semantic Evaluation Results:")
    print(f"Metrics: {res['metrics']}, Latency: {res['latency']}")

from src.core.logger.logger import Logger
from src.core.database.database_handler import DBMS
from src.metrics import Evaluation, EvaluationTechnique
from src.core.model_manager import OpenAIModel
import re

logger = Logger(name=__name__)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().lower().rstrip(";"))


def compute_exec_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> float:
    """
    Executes and compares predicted and gold SQL queries to compute execution accuracy.

    Args:
        pred_sql: The predicted SQL query.
        gold_sql: The reference SQL query.
        db_path: Path to the SQLite database.

    Returns:
        Execution accuracy score (0 or 1).
    """
    pred_sql = pred_sql.replace("STR_POSITION", "INSTR")
    gold_sql = gold_sql.replace("STR_POSITION", "INSTR")
    normalized_pred_sql = normalize_sql(pred_sql)
    normalized_gold_sql = normalize_sql(gold_sql)
    target_sql = normalize_sql(
        """
        SELECT DISTINCT t.account_id FROM trans t
        JOIN loan l ON t.account_id = l.account_id
        WHERE t.date = (
            SELECT MIN(date) FROM trans
            WHERE strftime('%Y', date) = '1995'
            AND account_id = t.account_id
        )
    """
    )

    if normalized_pred_sql == target_sql or normalized_gold_sql == target_sql:
        return 0.0

    config = {
        "evaluation_technique": EvaluationTechnique.EXECUTION_ACCURACY,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(db_path),
        },
        "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        "logs_dir_path": "data/evaluation_outputs/",
    }
    evaluator = Evaluation(config)
    res = evaluator.run_evaluation(
        predicted_sql=pred_sql, ground_truth_sql=gold_sql, log=False
    )
    return res["metrics"]["EX"]


# def main():
#     """
#     Main function to demonstrate the execution accuracy computation.
#     This is a placeholder for actual usage.
#     """
#     # Example usage
#     pred_sql = """
#     SELECT DISTINCT t.account_id
#     FROM trans t
#     JOIN loan l ON t.account_id = l.account_id
#     WHERE t.date = (
#         SELECT MIN(date)
#         FROM trans
#         WHERE strftime('%Y', date) = '1995'
#         AND account_id = t.account_id
#         )
#     """
#     gold_sql = """
#     SELECT trans.account_id FROM trans INNER JOIN loan AS et ON et.account_id = trans.account_id WHERE STRFTIME('%Y', trans.date) = '1995' ORDER BY trans.date ASC LIMIT 1

#     """
#     db_path = "data/benchmarks/Bird/bird_databases/financial/financial.sqlite"

#     accuracy = compute_exec_accuracy(pred_sql, gold_sql, db_path)
#     logger.log("info", f"Execution accuracy: {accuracy}")

# if __name__ == "__main__":
#     main()

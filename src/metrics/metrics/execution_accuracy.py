import time
from src.core.database.database_handler import DatabaseHandler
from src.core.logger import Logger

logger = Logger(__name__)


def execute_query(context: dict):
    logger.log("debug", "function execute_query called")
    try:
        db_handler = DatabaseHandler(
            dbms=context["db_params"]["dbms"], connection_params=context["db_params"]
        )

        pred_cols, pred_rows = db_handler.run_query(context["predicted_sql"])
        gt_cols, gt_rows = db_handler.run_query(context["ground_truth_sql"])

        context.update(
            {
                "has_error": False,
                "pred_rows": pred_rows,
                "gt_rows": gt_rows,
            }
        )
        return context
    except Exception as e:
        logger.log("error", "QUERY_EXECUTION_FAILED", {"error": str(e)})
        context.update(
            {
                "has_error": True,
                "error_message": str(e),
                "metrics": {"EX": 0},
            }
        )
        return context


def match_columns(context: dict):
    logger.log("debug", "function match_columns called")
    if context["has_error"]:
        return context

    # Calculate binary execution accuracy
    ex = 1 if set(context["gt_rows"]) == set(context["pred_rows"]) else 0

    context.update({"EX": ex})
    return context


def match_rows(context: dict):
    logger.log("debug", "function match_rows called")
    pass
    return context


def assign_metrics(context: dict):
    logger.log("debug", "function assign_metrics called")
    if context["has_error"]:
        return context

    ex = context.get("EX", 0)
    context["metrics"] = {"EX": ex}

    logger.log("info", "EVALUATION_COMPLETE", {"EX": context["metrics"]["EX"]})
    return context


def run_evaluation_pipeline(
    predicted_sql: str,
    ground_truth_sql: str,
    db_params: dict,
):
    context = {
        "db_params": db_params,
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "metrics": {},
        "has_error": False,
    }

    start_time = time.time()

    context = execute_query(context)
    context = match_columns(context)
    context = match_rows(context)
    context = assign_metrics(context)

    context["latency"] = time.time() - start_time

    return context

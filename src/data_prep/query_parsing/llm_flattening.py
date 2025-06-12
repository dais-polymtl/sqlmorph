from __future__ import annotations
from typing import List, Dict, Tuple

from src.core.model_manager.utils import compose_chat_messages
from src.core.model_manager.model_manager import ModelManager, ModelProvider, ModelType
from src.core.prompt_renderer.prompt_renderer import PromptRenderer
from src.core.model_manager.openai_model import OpenAIModel
from src.core.logger.logger import Logger
from src.core.database.database_handler import DatabaseHandler, DBMS

from sqlglot import parse_one
from sqlglot.expressions import Subquery
from pathlib import Path
import os

logger = Logger(__name__)


def flatten_queries(sql_query: str) -> str:
    """
    Transform a nested SQL query into a single flat SQL query with only one SELECT statement.

    Args:
        sql_query (str): The SQL query to process.

    Returns:
        str: The flattened SQL query as a continuous string without newlines.
    """
    try:
        prompt_template_path = os.path.join(
            os.path.dirname(__file__), "prompt_templates"
        )
        prompt = [
            PromptRenderer(prompt_template_path).render(
                template_name="flatten_query", context={"sql_query": sql_query}
            )
        ]
        messages = compose_chat_messages(user_messages=prompt)

        model = ModelManager.create_model(
            model_provider=ModelProvider.OPENAI,
            model_type=ModelType.COMPLETION,
            model_name=OpenAIModel.GPT_4O,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

        response = model.get_chat_completion(
            messages=messages,
            max_tokens=3000,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return response["completion_content"][0].strip().replace("\n", " ")

    except Exception as e:
        logger.log(
            level="error",
            action="Failed to flatten SQL query.",
            details={"sql_query": sql_query, "error": str(e)},
        )
        return sql_query  # Fallback to the original if flattening fails


def compare_queries(
    nested_query: str, flattened_query: str, db_path: str | Path
) -> bool:
    """
    Compare two SQL queries by executing them and checking if they return the same rows (order matters).

    Args:
        nested_query (str): Original nested query.
        flattened_query (str): Flattened version to compare.
        db_path (str | Path): Path to the database file.

    Returns:
        bool: True if queries return the same result, False otherwise.
    """
    db_handler = DatabaseHandler(DBMS.SQLITE, {"db_path": str(db_path)})
    db_handler.connect_to_database()

    try:
        _, nested_rows = db_handler.run_query(nested_query, return_cursor=False)
        _, flattened_rows = db_handler.run_query(flattened_query, return_cursor=False)

        return nested_rows == flattened_rows

    except Exception as e:
        logger.log(
            level="error",
            action="Failed to compare nested and flattened SQL queries.",
            details={
                "nested_query": nested_query,
                "flattened_query": flattened_query,
                "error": str(e),
            },
        )
        return False


def flatten_queries_automatically(
    queries: List[Dict[str, str]], dataset: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Attempt to flatten nested SQL queries automatically and validate them.

    Args:
        queries: List of query dictionaries (assumed to be nested).
        dataset: Either 'Bird' or 'Beaver'.

    Returns:
        A tuple:
            - Flattened queries that are correct
            - Queries still nested or incorrect after flattening
    """
    still_nested = []
    successfully_flattened = []

    db_root = (
        Path(os.getenv("DATA_FOLDER"))
        / "benchmarks"
        / dataset
        / f"{dataset.lower()}_databases"
    )

    for i, query in enumerate(queries):
        sql = query.get("SQL", "") or query.get("sql", "")
        db_id = query.get("db_id", "")
        if not sql or not db_id:
            continue

        db_path = (
            db_root
            / db_id
            / (f"{db_id}.sqlite" if dataset == "Bird" else f"{db_id}.sql")
        )

        # Flatten using LLM
        flattened = flatten_queries(sql)

        try:
            is_flat = parse_one(flattened, dialect="mysql").find(Subquery) is None
        except Exception as e:
            logger.log("warning", f"Failed to parse flattened query at index {i}: {e}")
            still_nested.append(query)
            continue

        is_equivalent = compare_queries(sql, flattened, db_path)

        logger.log(
            "info",
            f"[{i + 1}/{len(queries)}] Flattened: {'Yes' if is_flat else 'No'} | Same Result: {'Yes' if is_equivalent else 'No'}",
        )

        if is_flat and is_equivalent:
            query["flattened_query"] = flattened
            successfully_flattened.append(query)
        else:
            still_nested.append(query)

    return successfully_flattened, still_nested

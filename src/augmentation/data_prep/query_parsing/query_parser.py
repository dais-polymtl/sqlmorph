import re
from typing import Dict, Tuple, List
import sqlparse
from sqlglot import parse_one
import sqlglot.expressions as exp
from sqlparse.sql import Comparison
from src.core.logger.logger import Logger

logger = Logger(__name__)


def extract_tables(sql: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Extract tables and their aliases from the SQL query.

    Args:
        sql (str): The SQL query to extract tables from.

    Returns:
        Tuple: A tuple containing a list of table information and a dictionary of aliases to table names.
    """
    tables_info = []
    aliases_to_table = {}
    try:
        parsed_query = parse_one(sql, dialect="mysql")  # Parse the query
        if parsed_query:  # Check if parsing was successful
            for table in parsed_query.find_all(exp.Table):
                table_name = table.name
                table_alias = table.alias
                tables_info.append(
                    {"table_name": table_name, "table_alias": table_alias}
                )
                if table_alias:
                    aliases_to_table[table_alias] = table_name
    except Exception as e:
        logger.log(
            level="error",
            action="Failed to parse SQL query for table extraction.",
            details={"sql": sql, "error": str(e)},
        )
    return tables_info, aliases_to_table


def replace_aliases_in_join_conditions(
    join_conditions: List[str], aliases_to_table: Dict[str, str]
) -> List[str]:
    """
    Replace aliases in join conditions with actual table names.

    Args:
        join_conditions (List[str]): The list of join conditions.
        aliases_to_table (Dict[str, str]): A dictionary of aliases to table names.

    Returns:
        List[str]: The updated join conditions with aliases replaced by table names.
    """
    updated_conditions = []

    for condition in join_conditions:
        left, right = condition.split("=")
        left, right = left.strip(), right.strip()

        table_or_alias1, col1 = left.split(".")
        table_or_alias2, col2 = right.split(".")

        table_1 = aliases_to_table.get(table_or_alias1, table_or_alias1)
        table_2 = aliases_to_table.get(table_or_alias2, table_or_alias2)

        updated_condition = f"{table_1}.{col1} = {table_2}.{col2}"
        updated_conditions.append(updated_condition)

    return updated_conditions


# Function to extract join conditions using sqlparse
def extract_joins(token_list) -> List[Comparison]:
    """
    Extract join conditions from the SQL query tokens.

    Args:
        token_list: The list of SQL query tokens.

    Returns:
        List[Comparison]: A list of join conditions.
    """
    join_pattern = re.compile(
        r"""^\s*         # optional leading spaces
        (\w+\.`[^`]+`|\w+\.\w+)    # table.column (supports backticks)
        \s*=\s*      # equals with optional spaces
        (\w+\.`[^`]+`|\w+\.\w+)    # table.column (supports backticks)
        \s*$         # optional trailing spaces
    """,
        re.IGNORECASE | re.VERBOSE,
    )
    join_conditions = []
    for token in token_list:
        if isinstance(token, Comparison):
            parent_token = token.parent
            if any(str(t).upper() == "ON" for t in parent_token.tokens):
                if join_pattern.match(str(token)):
                    join_conditions.append(token)
        elif token.is_group:
            join_conditions.extend(extract_joins(token.tokens))
    return join_conditions


def get_joins_w_aliases(sql: str) -> List[str]:
    """
    Extract join conditions from the SQL query.

    Args:
        sql (str): The SQL query to extract join conditions from.

    Returns:
        List[str]: A list of join conditions extracted from the SQL query.
    """
    parsed = sqlparse.parse(sql)
    join_conditions = []
    for statement in parsed:
        join_conditions.extend([str(t) for t in extract_joins(statement.tokens)])
    return join_conditions


def parse_queries(
    queries: List[Dict],
) -> None:

    queries_per_db = {}
    for i, query in enumerate(queries):
        sql_query = (
            query.get("SQL", "") or query.get("sql", "")
            if query.get("flattened_query") is None
            else query.get("flattened_query", "")
        )
        tables, aliases_to_table_map = extract_tables(sql_query)
        join_conditions_with_aliases = get_joins_w_aliases(sql_query)
        join_conditions_without_aliases = replace_aliases_in_join_conditions(
            join_conditions_with_aliases, aliases_to_table_map
        )

        parsed_data = {
            "question_id": query.get("question_id", "") or i,
            "db_id": query.get("db_id", ""),
            "SQL": query.get("SQL", "") or query.get("sql", ""),
            "question": query.get("question", ""),
            "evidence": query.get("evidence", ""),
            "difficulty": query.get("difficulty", ""),
            "tables": tables,
            "jr_w_aliases": join_conditions_with_aliases,
            "jr_wo_aliases": join_conditions_without_aliases,
            "flattened_query": query.get("flattened_query", ""),
        }

        queries_per_db.setdefault(query.get("db_id", ""), []).append(parsed_data)

    return queries_per_db

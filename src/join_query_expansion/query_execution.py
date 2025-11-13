from src.core.database.database_handler import DatabaseHandler, DBMS
from src.core.logger.logger import Logger

logger = Logger(__name__)


def execute_query(query, db_path):
    duckdb_handler = DatabaseHandler(DBMS.DUCKDB, {"db_path": db_path})
    duckdb_handler.connect_to_database()

    try:
        query_results = duckdb_handler.run_query(query, return_cursor=False)

    except Exception:
        return None
    finally:
        duckdb_handler.close_connection()

    return query_results


def execute_test_query_and_replace_placeholders(
    db_path, test_query, main_query, filtering_columns
):
    """
    Execute a test query and replace placeholders in the main query with actual filtering values.
    Guarantees that no `None` values are used to replace placeholders.
    """
    try:
        # Execute the test query

        result = execute_query(test_query, db_path)
        if result is None:
            return main_query

        _, rows = result
        if not rows:
            return main_query

        rows.sort(key=lambda row: row[-1], reverse=True)

        for row in rows:
            main_query
            num_values = len(filtering_columns)
            potential_values = (
                row[-(num_values + 1) : -1]
                if num_values > 0
                else row[: -(num_values + 1)]
            )
            for value in potential_values:
                if value is None:
                    main_query = main_query.replace("= ?", "IS NULL", 1)
                else:
                    escaped_value = str(value).replace("'", "''")
                    main_query = main_query.replace("= ?", f"= '{escaped_value}'", 1)
            break

    except Exception as e:
        logger.log(
            "error",
            "Test query execution failed, main query will be returned unchanged.",
            {
                "query": test_query,
                "error": str(e),
            },
        )

    return main_query


def add_values_to_translated_queries(pattern, db_path):
    test_query = pattern.get("test_query")
    main_query = pattern.get("main_query")

    filtering_columns = pattern.get("filtering_columns")
    final_query = execute_test_query_and_replace_placeholders(
        db_path, test_query, main_query, filtering_columns
    )

    pattern["main_query"] = final_query
    return pattern


def execute_new_queries(query, db_path):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - rule_data: List of dictionaries containing query information.
    - db_file: Path to the SQLite database file.

    Returns:
    - List of dictionaries with valid queries.
    """
    try:
        new_query = query.copy()
        main_query = new_query.get("main_query", "")
        query_results = execute_query(main_query, db_path)
        if len(query_results[1]) == 0:
            return ""

    except Exception:
        return ""
    new_query["main_query"] = main_query
    return new_query


def is_result_meaningful(results, query):
    if not results or not isinstance(results, list):
        return False

    # Normalize the query for keyword matching
    query_lower = query.lower()
    is_aggregate_query = any(
        agg in query_lower for agg in ["count(", "avg(", "sum(", "max(", "min("]
    )

    # If it returns exactly one row and one column
    if len(results) == 1 and len(results[0]) == 1:
        value = results[0][0]
        if value in (0, None) and is_aggregate_query:
            return False

    return True


def execute_extended_queries(ext_jqg, db_path):
    """
    Executes and validates an extended query, restoring formatting after evaluation.

    Parameters:
    - ext_jqg: Dict containing query metadata (must include 'new_query').
    - db_path: Path to the database.

    Returns:
    - Tuple: (Modified query dict, Bool indicating if query is valid and meaningful)
    """
    ext_jqg_copy = ext_jqg.copy()
    new_query = ext_jqg.get("new_query", "")

    # Temporarily replace known placeholders for compatibility
    replacements = {
        "STR_POSITION": "STRPOS",
        "teamInfo.team_long_name": "ANY_VALUE(teamInfo.team_long_name)",
    }
    for original, replacement in replacements.items():
        new_query = new_query.replace(original, replacement)

    try:
        query_results = execute_query(new_query, db_path)
        results = (
            query_results[1]
            if isinstance(query_results, (list, tuple)) and len(query_results) > 1
            else query_results
        )
    except Exception:
        return ext_jqg_copy, False

    if is_result_meaningful(results, new_query):
        # Restore original placeholders
        for original, replacement in replacements.items():
            new_query = new_query.replace(replacement, original)

        ext_jqg_copy["new_query"] = new_query
        return ext_jqg_copy, True

    return ext_jqg_copy, False

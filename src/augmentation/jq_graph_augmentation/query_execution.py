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
        if not result:
            return main_query

        rows.sort(key=lambda row: row[-1], reverse=True)

        for row in rows:
            num_values = len(filtering_columns)
            potential_values = (
                row[-(num_values + 1) : -1]
                if num_values > 0
                else row[: -(num_values + 1)]
            )
            if all(value is not None for value in potential_values):
                for value in potential_values:
                    main_query = main_query.replace("?", f"'{value}'", 1)
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
    main_query = (
        main_query.replace("order", '"order"') if "order" in main_query else main_query
    )
    test_query = (
        test_query.replace("order", '"order"') if "order" in test_query else test_query
    )

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
        query_results = execute_query(query["main_query"], db_path)
        if len(query_results[1]) == 0:
            return ""
    except Exception:
        return ""

    return query


def execute_extended_queries(query_list, db_path):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - query_list: List of dictionaries containing query information.
    - adapter: An instance of the database adapter.

    Returns:
    - Tuple: (List of dictionaries with valid queries, boolean indicating if at least one was valid)
    """
    valid_query_list = []

    for query in query_list:
        new_query = query.get("new_query", "")

        # Replace placeholders for compatibility
        new_query = new_query.replace("STR_POSITION", "STRPOS")
        new_query = new_query.replace(
            "teamInfo.team_long_name", "ANY_VALUE(teamInfo.team_long_name)"
        )

        try:
            query_results = execute_query(new_query, db_path)
            results = (
                query_results[1]
                if isinstance(query_results, (list, tuple)) and len(query_results) > 1
                else query_results
            )
        except Exception:
            continue  # Skip to next query

        if results:
            # Restore placeholders back to original
            new_query = new_query.replace("STRPOS", "STR_POSITION")
            new_query = new_query.replace(
                "ANY_VALUE(teamInfo.team_long_name)", "teamInfo.team_long_name"
            )

            query["new_query"] = new_query
            valid_query_list.append(query)
            break  # You break after the first valid one, correct?

    valid_query = len(valid_query_list) > 0
    return valid_query_list, valid_query

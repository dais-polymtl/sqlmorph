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
    is_aggregate_query = any(agg in query_lower for agg in ["count(", "avg(", "sum("])

    # If it returns exactly one row and one column
    if len(results) == 1 and len(results[0]) == 1:
        value = results[0][0]
        if value in (0, None) and is_aggregate_query:
            return False

    return True


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
    skip_query_1 = "SELECT T3.Title FROM postLinks AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN posts AS T3 ON T1.RelatedPostId = T3.Id INNER JOIN badges AS extra_table ON extra_table.UserId = T2.OwnerUserId WHERE T2.Title = 'How to tell if something happened in a data set which monitors a value over time'"
    skip_query_2 = "SELECT T3.Title, T2.LinkTypeId FROM posts AS T1 INNER JOIN postLinks AS T2 ON T1.Id = T2.PostId INNER JOIN posts AS T3 ON T2.RelatedPostId = T3.Id INNER JOIN comments AS extra_table ON extra_table.PostId = T2.RelatedPostId WHERE T1.Title = 'What are principal component scores?'"
    skip_query_3 = "SELECT T3.Title, T2.LinkTypeId FROM posts AS T1 INNER JOIN postLinks AS T2 ON T1.Id = T2.PostId INNER JOIN posts AS T3 ON T2.RelatedPostId = T3.Id INNER JOIN badges AS extra_table ON extra_table.UserId = T1.OwnerUserId WHERE T1.Title = 'What are principal component scores?'"
    skip_query_4 = "SELECT T3.Title FROM postLinks AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN posts AS T3 ON T1.RelatedPostId = T3.Id INNER JOIN comments AS extra_table ON extra_table.PostId = T2.Id WHERE T2.Title = 'How to tell if something happened in a data set which monitors a value over time'"
    skip_query_5 = "SELECT T3.Title, T2.LinkTypeId FROM posts AS T1 INNER JOIN postLinks AS T2 ON T1.Id = T2.PostId INNER JOIN posts AS T3 ON T2.RelatedPostId = T3.Id INNER JOIN comments AS extra_table ON extra_table.PostId = T1.Id WHERE T1.Title = 'What are principal component scores?'"
    skip_query_6 = "SELECT T3.Title FROM postLinks AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN posts AS T3 ON T1.RelatedPostId = T3.Id INNER JOIN postHistory AS extra_table ON extra_table.PostId = T1.RelatedPostId WHERE T2.Title = 'How to tell if something happened in a data set which monitors a value over time'"
    for query in query_list:
        if (
            query.get("new_query", "") == skip_query_1
            or query.get("new_query", "") == skip_query_2
            or query.get("new_query", "") == skip_query_3
            or query.get("new_query", "") == skip_query_4
            or query.get("new_query", "") == skip_query_5
            or query.get("new_query", "") == skip_query_6
        ):
            continue

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

        if is_result_meaningful(results, new_query):
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

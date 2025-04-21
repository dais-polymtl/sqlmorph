import pandas as pd
import duckdb
import re


def execute_test_query_and_replace_placeholders(
    adapter, test_query, main_query, filtering_columns
):
    """
    Execute a test query and replace placeholders in the main query with actual filtering values.
    Guarantees that no `None` values are used to replace placeholders.
    """
    try:
        # Execute the test query

        query_result = adapter.run_query(test_query)

        _, results = query_result

        results.sort(key=lambda row: row[-1], reverse=True)

        # Now find the first row with valid (non-None) filtering values
        max_values = None
        for row in results:
            potential_values = (
                row[-(len(filtering_columns) + 1) : -1]
                if filtering_columns
                else row[: -(len(filtering_columns) + 1)]
            )
            if all(value is not None for value in potential_values):
                max_values = potential_values
                break

        if max_values is None:
            return main_query

        for value in max_values:
            main_query = main_query.replace("?", f"'{value}'", 1)

    except Exception as e:
        print(f"Error executing test query: {e}")

    return main_query


# def execute_query(db_file, query):
#     """
#     Executes a query on a SQLite file using DuckDB.

#     Parameters:
#         db_file: Path to the SQLite database file.
#         query: SQL query to execute.

#     Returns:
#         DataFrame with results, or empty DataFrame on error.
#     """
#     try:
#         conn = duckdb.connect()
#         conn.execute("INSTALL sqlite_scanner;")
#         conn.execute("LOAD sqlite_scanner;")
#         # conn.execute(f"ATTACH DATABASE '{db_file}' (TYPE sqlite);")
#         conn.execute(f"CALL sqlite_attach('{db_file}');")  # attaches as 'main' implicitly


#         df = conn.execute(query).fetchdf()
#         return df, "", ""
#     except Exception as e:
#         return pd.DataFrame(), query, str(e)
#     finally:
#         conn.close()


def add_values_to_translated_queries(pattern, db_id, adapter):
    adapter.connect()
    test_query = pattern.get("test_query")
    main_query = pattern.get("main_query")
    filtering_columns = pattern.get("filtering_columns")

    final_query = execute_test_query_and_replace_placeholders(
        adapter, test_query, main_query, filtering_columns
    )

    pattern["main_query"] = final_query

    adapter.close_connection()
    return pattern


def execute_new_queries(query, adapter):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - rule_data: List of dictionaries containing query information.
    - db_file: Path to the SQLite database file.

    Returns:
    - List of dictionaries with valid queries.
    """
    adapter.connect()

    # Execute the query and check if it returns results
    query_result = adapter.run_query(query["main_query"], return_cursor=False)

    if query_result is None:
        valid_query = False

    else:
        valid_query = True

    adapter.close_connection()
    return query, valid_query


def execute_extended_queries(query_list, adapter):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - rule_data: List of dictionaries containing query information.
    - db_file: Path to the SQLite database file.

    Returns:
    - List of dictionaries with valid queries.
    """

    adapter.connect()

    valid_query_list = []

    for query in query_list:
        new_query = query.get("new_query", "")
        new_query = (
            new_query.replace("STR_POSITION", "STRPOS")
            if "STR_POSITION" in new_query
            else new_query
        )
        new_query = (
            new_query.replace(
                "teamInfo.team_long_name", "ANY_VALUE(teamInfo.team_long_name)"
            )
            if "teamInfo.team_long_name" in new_query
            else new_query
        )
        query_results = adapter.run_query(new_query, return_cursor=False)
        results = query_results[1] if query_results is not None else []

        if query_results is not None and results != []:
            new_query = (
                new_query.replace("STRPOS", "STR_POSITION")
                if "STRPOS" in new_query
                else new_query
            )
            new_query = (
                new_query.replace(
                    "ANY_VALUE(teamInfo.team_long_name)", "teamInfo.team_long_name"
                )
                if "ANY_VALUE(teamInfo.team_long_name)" in new_query
                else new_query
            )

            query["new_query"] = new_query

            valid_query_list.append(query)
            break

    if len(valid_query_list) == 0:
        valid_query = False
    elif len(valid_query_list) > 0:
        valid_query = True

    adapter.close_connection()
    return valid_query_list, valid_query

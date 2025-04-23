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


def add_values_to_translated_queries(pattern, adapter):
    adapter.connect_to_database()
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
    adapter.connect_to_database()
    try:
        query_result = adapter.run_query(query["main_query"], return_cursor=False)
        valid_query = query_result is not None
    except Exception:
        valid_query = False

    adapter.close_connection()
    return query, valid_query


def execute_extended_queries(query_list, adapter):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - query_list: List of dictionaries containing query information.
    - adapter: An instance of the database adapter.

    Returns:
    - Tuple: (List of dictionaries with valid queries, boolean indicating if at least one was valid)
    """
    adapter.connect_to_database()
    valid_query_list = []

    for query in query_list:
        new_query = query.get("new_query", "")

        # Replace placeholders for compatibility
        new_query = new_query.replace("STR_POSITION", "STRPOS")
        new_query = new_query.replace(
            "teamInfo.team_long_name", "ANY_VALUE(teamInfo.team_long_name)"
        )

        try:
            query_results = adapter.run_query(new_query, return_cursor=False)
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

    adapter.close_connection()

    valid_query = len(valid_query_list) > 0
    return valid_query_list, valid_query

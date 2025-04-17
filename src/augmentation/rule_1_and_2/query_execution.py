import sqlite3
import pandas as pd
import multiprocessing


def vacuum_database(conn):
    """Vacuum the SQLite database to optimize performance."""
    cursor = conn.cursor()
    try:
        cursor.execute("VACUUM;")
        print("Database vacuumed successfully.")
    except Exception as e:
        print(f"Error during vacuuming: {e}")
    finally:
        cursor.close()  # Close the cursor but keep the connection open


def execute_test_query_and_replace_placeholders(
    conn, test_query, main_query, filtering_columns
):
    """
    Execute a test query and replace placeholders in the main query with actual filtering values.
    Guarantees that no `None` values are used to replace placeholders.
    """
    cursor = conn.cursor()
    try:
        # Execute the test query
        print("Executing test query...")
        print("test_query:", test_query)
        cursor.execute(test_query)

        # Fetch all results
        results = cursor.fetchall()

        # Sort rows by COUNT(*) (last column) in descending order
        results.sort(key=lambda row: row[-1], reverse=True)

        # Now find the first row with valid (non-None) filtering values
        max_values = None
        for row in results:
            # count_value = row[-1]  # The last column is COUNT(*)
            if filtering_columns:
                potential_values = row[-(len(filtering_columns) + 1) : -1]
            else:
                potential_values = row[: -(len(filtering_columns) + 1)]

            # Check if all values are valid (not None)
            if all(value is not None for value in potential_values):
                max_values = potential_values
                break

        # If no valid result is found, return the main query unchanged
        if max_values is None:
            print(
                "No valid results found for test query. Returning original main query."
            )
            return main_query

        # Replace placeholders in the main query with actual values
        for value in max_values:
            main_query = main_query.replace(
                "?", f"'{value}'", 1
            )  # Replace one placeholder at a time

    except sqlite3.OperationalError as e:
        print(f"SQLite OperationalError: {e}")
        print("Returning the original main query without replacements.")
    finally:
        cursor.close()  # Ensure the cursor is closed after use

    return main_query


def run_query(db_file, query, result_queue):
    """Runs the query and stores the result in a queue."""
    conn = sqlite3.connect(db_file)  # Create connection inside the process
    try:
        df = pd.read_sql_query(query, conn)
        result_queue.put((df, "", ""))  # Send results back
    except Exception as e:
        result_queue.put((pd.DataFrame(), query, str(e)))
    finally:
        conn.close()


def execute_query(db_file, query, timeout=240):
    """
    Executes a SQLite query with a timeout by running it in a separate process.

    Parameters:
        db_file: Path to the SQLite database file.
        query: SQL query to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        DataFrame containing the results, or an empty DataFrame on timeout/error.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_query, args=(db_file, query, result_queue)
    )
    process.start()
    process.join(timeout)  # Wait for the process to finish within the timeout

    if process.is_alive():
        process.terminate()  # Kill the process if it exceeds the timeout
        process.join()
        print(f"Query execution timed out after {timeout} seconds.")
        return pd.DataFrame(), query, f"Query timed out after {timeout} seconds."

    return (
        result_queue.get()
        if not result_queue.empty()
        else (pd.DataFrame(), query, "Unknown error")
    )


def process_queries_for_rules(rule_data, db_id, database_path):
    """
    Process rule data by executing test queries and replacing placeholders in main queries.

    Parameters:
        rule_data: List of dictionaries containing subgraph and query information.
        db_id: Database ID to construct the database path.
        database_path: Path template for accessing SQLite databases.
    """
    for i, result in enumerate(rule_data, start=1):
        print(f"Processing subgraph {i}...")

        # Check for None values in necessary fields before processing
        test_query = result.get("test_query")
        main_query = result.get("main_query")
        filtering_columns = result.get("filtering_columns")

        # If any of the required fields are None, skip the processing for this result
        if test_query is None or main_query is None or filtering_columns is None:
            print("Skipping this subgraph due to missing data (None values).")
            continue

        # Open a new connection for each query
        with sqlite3.connect(database_path.format(db_id=db_id)) as conn:
            # Vacuum the database before processing queries
            vacuum_database(conn)

            # Execute the test query and replace placeholders in the main query
            final_query = execute_test_query_and_replace_placeholders(
                conn, test_query, main_query, filtering_columns
            )

        # Update the rule data with the final query (in-place update)
        result["main_query"] = final_query

        print(f"Updated main_query for Subgraph {i}:\n{final_query}\n")


def filter_valid_queries(rule_data, db_file):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - rule_data: List of dictionaries containing query information.
    - db_file: Path to the SQLite database file.

    Returns:
    - List of dictionaries with valid queries.
    """
    valid_data = []

    if any(element["main_query"] is None for element in rule_data):
        print(f"Total Queries: {0}")
    else:
        print(f"Total Queries: {len(rule_data)}")
    for element in rule_data:
        if element["main_query"] is None:
            return rule_data

        else:
            if "'Teferi's Protection'" in element["main_query"]:
                element["main_query"] = element["main_query"].replace(
                    "'Teferi's Protection'", "'Teferi''s Protection'"
                )
            if "Ancestor's Chosen" in element["main_query"]:
                element["main_query"] = element["main_query"].replace(
                    "Ancestor's Chosen", "Ancestor''s Chosen"
                )
            # Execute the query and check if it returns results
            result_df, _, _ = execute_query(db_file, element["main_query"])
            if result_df.empty:
                print("Query did not return any results. Skipping...")
                print(f"Main Query: {element['main_query']}")
            if not result_df.empty:  # Check if the query returned results
                print("Query executed successfully!")
                valid_data.append(element)
    return valid_data


def filter_valid_queries_2nd_version(rule_data, db_file):
    """
    Filters the given rule data to keep only valid queries.

    Parameters:
    - rule_data: List of dictionaries containing query information.
    - db_file: Path to the SQLite database file.

    Returns:
    - List of dictionaries with valid queries.
    """
    valid_data = []
    error_queries = []
    empty_queries = []

    print(f"Total Queries: {len(rule_data)}")
    if len(rule_data) == 0:
        return rule_data, [], []

    for pattern in rule_data:
        equivalent_queries = pattern.get("equivalent_queries", [])
        valid_equivalent_queries = []

        for query in equivalent_queries:
            new_query = query.get("new_query", "")

            result_df, error_query, error = execute_query(db_file, new_query)

            if result_df.empty and error_query != "":
                print("Query did not return any results. Skipping...")
                print(f"Main Query: {new_query}")
                empty_queries.append(
                    {
                        "db_id": query.get("db_id", ""),
                        "question": query.get("question", ""),
                        "SQL": query.get("SQL", ""),
                        "new_query": new_query,
                    }
                )
            else:
                print("Query executed successfully!")
                query["new_query"] = new_query
                valid_equivalent_queries.append(query)

            if len(error_query) > 0:
                error_queries.append(
                    {
                        "db_id": query.get("db_id", ""),
                        "question": query.get("question", ""),
                        "SQL": query.get("SQL", ""),
                        "new_query": error_query,
                        "error": error,
                    }
                )

        pattern["equivalent_queries"] = valid_equivalent_queries
        if len(valid_equivalent_queries) > 0:
            valid_data.append(pattern)

    return valid_data, error_queries, empty_queries

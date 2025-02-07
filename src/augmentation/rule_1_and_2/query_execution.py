import sqlite3
import pandas as pd


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
    """
    cursor = conn.cursor()
    try:
        # Execute the test query
        print("Executing test query...")
        print("test_query:", test_query)
        cursor.execute(test_query)

        # Fetch all results to determine the maximum COUNT(*) and corresponding filtering values
        results = cursor.fetchall()
        max_count = 0
        max_values = None

        for row in results:
            count_value = row[-1]  # The last column is COUNT(*)
            if count_value > max_count:
                max_count = count_value
                # Extract filtering values based on the number of filtering columns
                if filtering_columns:
                    max_values = row[-(len(filtering_columns) + 1) : -1]
                else:
                    max_values = row[: -(len(filtering_columns) + 1)]

        # If no result is found, return the main query unchanged
        if max_values is None:
            print(f"No results found for test query. Returning original main query.")
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


def execute_query(db_file, query):
    """
    Connects to the SQLite database and executes the given query.

    Parameters:
        db_file: Path to the SQLite database file.
        query: SQL query to execute.

    Returns:
        DataFrame containing the results of the query.
    """
    # Connect to the database
    conn = sqlite3.connect(db_file)

    try:
        # Execute the query and fetch results
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        # Close the connection
        conn.close()


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
            print(f"Skipping this subgraph due to missing data (None values).")
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

    if any(element["main_query"] == None for element in rule_data):
        print(f"Total Queries: {0}")
    else:
        print(f"Total Queries: {len(rule_data)}")
    for element in rule_data:
        if element["main_query"] == None:
            return rule_data

        else:
            # Execute the query and check if it returns results
            result_df = execute_query(db_file, element["main_query"])
            if not result_df.empty:  # Check if the query returned results
                print("Query executed successfully!")
                valid_data.append(element)
    return valid_data

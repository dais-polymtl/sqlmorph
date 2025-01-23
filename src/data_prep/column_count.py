import os
import json
import csv
import sqlglot
from collections import defaultdict, Counter
import logging

# Setting up logging for better traceability and debugging
logging.basicConfig(
    level=logging.INFO,  # You can set to DEBUG for more granular logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Function to sanitize SQL queries by replacing backticks with single quotes
def sanitize_sql(sql):
    """
    Replaces backticks with single quotes in the SQL query string.
    :param sql: The SQL query string to sanitize.
    :return: Sanitized SQL query string.
    """
    return sql.replace("`", "'")  # Replace backticks with single quotes


# Function to parse the SQL and extract projections and filtering columns
def parse_sql(sql, tables):
    """
    Parses SQL query to extract projection and filtering columns.
    :param sql: SQL query string.
    :param tables: List of table dictionaries containing the table name and aliases.
    :return: Dictionary with projections and filtering columns.
    """
    # Sanitize SQL and parse it using SQLGlot
    sql = sanitize_sql(sql)
    parsed = sqlglot.parse_one(sql)

    # Create a map of table aliases to their actual names (lowercased)
    alias_map = {
        table["table_alias"].lower(): table["table_name"].lower() for table in tables
    }

    projections = Counter()
    filtering = Counter()

    # Extract projections from SELECT clauses
    for select in parsed.find_all(sqlglot.expressions.Select):
        for projection in select.expressions:
            column = projection.find(sqlglot.expressions.Column) or projection
            if isinstance(column, sqlglot.expressions.Column):
                col_name = projection.alias_or_name
                table_alias = column.table.lower() if column.table else None
                column_name = (
                    column.name
                    if isinstance(column, sqlglot.expressions.Column)
                    else col_name
                )

                table_name = alias_map.get(table_alias, "unknown")
                projections[(table_name, column_name)] += 1
            else:
                projections[(None, projection.sql())] += (
                    1  # For operations like MAX(), SUM()
                )

    # Extract filtering columns from WHERE clause
    where_clause = parsed.find(sqlglot.expressions.Where)
    if where_clause:
        for column in where_clause.find_all(sqlglot.expressions.Column):
            if isinstance(column, sqlglot.expressions.Column):
                table_alias = column.table.lower() if column.table else None
                column_name = column.name

                table_name = alias_map.get(table_alias, "unknown")
                filtering[(table_name, column_name)] += 1

    return projections, filtering


# Function to process each query pattern and update the stats
def process_query_pattern(query_pattern, db_stats, db_row_count):
    """
    Processes each query pattern to extract and update projection and filtering statistics.
    :param query_pattern: The query pattern (dict) to process.
    :param db_stats: Dictionary to store statistics.
    :param db_row_count: Dictionary to store the number of rows per db_id.
    """
    db_id = query_pattern.get("db_id", "unknown")
    tables = query_pattern.get("Tables", [])
    sql = query_pattern.get("SQL", "")

    # Increment the count of rows for this db_id
    db_row_count[db_id] += 1

    # Parse the SQL query and extract projections and filtering
    projections, filtering = parse_sql(sql, tables)

    # Update the total conditions and projections count for this db_id
    db_stats[db_id]["total_conditions"] += sum(filtering.values())
    db_stats[db_id]["total_projections"] += sum(projections.values())

    # Update max/min conditions/projections
    db_stats[db_id]["max_conditions"] = max(
        db_stats[db_id]["max_conditions"], sum(filtering.values())
    )
    db_stats[db_id]["min_conditions"] = min(
        db_stats[db_id]["min_conditions"], sum(filtering.values())
    )
    db_stats[db_id]["max_projections"] = max(
        db_stats[db_id]["max_projections"], sum(projections.values())
    )
    db_stats[db_id]["min_projections"] = min(
        db_stats[db_id]["min_projections"], sum(projections.values())
    )

    # Update table-wise statistics
    for (table_name, col_name), count in projections.items():
        db_stats[db_id]["tables"][table_name]["projection"][col_name] += count
    for (table_name, col_name), count in filtering.items():
        db_stats[db_id]["tables"][table_name]["filtering"][col_name] += count

    logging.debug(f"Processed query pattern for db_id: {db_id}")


# Function to write statistics to a CSV file
def write_stats_to_csv(db_stats, db_row_count, output_csv_file):
    """
    Writes the computed statistics to a CSV file.
    :param db_stats: Dictionary with the statistics to write.
    :param db_row_count: Dictionary with the count of rows per db_id.
    :param output_csv_file: Path to the output CSV file.
    """
    with open(output_csv_file, "w", newline="") as csvfile:
        fieldnames = [
            "db_id",
            "num_queries",
            "table_name",
            "table_count",
            "projection_column",
            "projection_occurrences",
            "filtering_column",
            "filtering_occurrences",
            "avg_conditions_per_query",
            "avg_projections_per_query",
            "max_conditions_per_query",
            "min_conditions_per_query",
            "max_projections_per_query",
            "min_projections_per_query",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write rows for each db_id and table
        for db_id, stats in db_stats.items():
            total_queries = db_row_count[db_id]
            avg_conditions = (
                round(stats["total_conditions"] / total_queries)
                if total_queries > 0
                else 0
            )
            avg_projections = (
                round(stats["total_projections"] / total_queries)
                if total_queries > 0
                else 0
            )

            for table, table_stats in stats["tables"].items():
                table_count = table_stats["table_count"]

                # Write projection columns data to CSV
                for col_name, count in table_stats["projection"].items():
                    writer.writerow(
                        {
                            "db_id": db_id,
                            "num_queries": total_queries,
                            "table_name": table,
                            "table_count": table_count,
                            "projection_column": col_name,
                            "projection_occurrences": f"{count}/{total_queries}",
                            "filtering_column": "",
                            "filtering_occurrences": "",
                            "avg_conditions_per_query": avg_conditions,
                            "avg_projections_per_query": avg_projections,
                            "max_conditions_per_query": stats["max_conditions"],
                            "min_conditions_per_query": stats["min_conditions"],
                            "max_projections_per_query": stats["max_projections"],
                            "min_projections_per_query": stats["min_projections"],
                        }
                    )

                # Write filtering columns data to CSV
                for col_name, count in table_stats["filtering"].items():
                    writer.writerow(
                        {
                            "db_id": db_id,
                            "num_queries": total_queries,
                            "table_name": table,
                            "table_count": table_count,
                            "projection_column": "",
                            "projection_occurrences": "",
                            "filtering_column": col_name,
                            "filtering_occurrences": f"{count}/{total_queries}",
                            "avg_conditions_per_query": avg_conditions,
                            "avg_projections_per_query": avg_projections,
                            "max_conditions_per_query": stats["max_conditions"],
                            "min_conditions_per_query": stats["min_conditions"],
                            "max_projections_per_query": stats["max_projections"],
                            "min_projections_per_query": stats["min_projections"],
                        }
                    )

    logging.info(f"CSV file '{output_csv_file}' has been created successfully.")


# Function to process the folder containing JSON files and generate statistics
def generate_query_statistics(json_folder, output_csv_file):
    """
    Processes all the JSON files in the given folder and generates query statistics.
    :param json_folder: Path to the folder containing JSON files.
    :param output_csv_file: Path to the output CSV file.
    """
    logging.info(f"Starting to process files from folder: {json_folder}")

    # List all JSON files in the folder
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    logging.info(f"Found {len(json_files)} JSON files.")

    # Load all data from the JSON files
    data = []
    for file in json_files:
        with open(os.path.join(json_folder, file), "r") as f:
            data += json.load(f)

    # Initialize statistics dictionaries
    db_stats = defaultdict(
        lambda: {
            "total_conditions": 0,
            "total_projections": 0,
            "max_conditions": 0,
            "min_conditions": float("inf"),
            "max_projections": 0,
            "min_projections": float("inf"),
            "tables": defaultdict(
                lambda: {
                    "projection": Counter(),
                    "filtering": Counter(),
                    "table_count": 0,
                }
            ),
        }
    )
    db_row_count = defaultdict(int)

    # Process each query pattern
    for query_pattern in data:
        process_query_pattern(query_pattern, db_stats, db_row_count)

    # Write statistics to CSV
    write_stats_to_csv(db_stats, db_row_count, output_csv_file)

import json
import pandas as pd
import re
from tqdm import tqdm
import signal
from contextlib import contextmanager

from src.core.database.database_handler import DatabaseHandler
from src.core.database.database_handler import DBMS


@contextmanager
def timeout(seconds):
    """Context manager for timeout handling."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


def read_name_mapping(mapping_csv_path):
    """
    Read the name mapping CSV file and organize it by database ID.

    Args:
        mapping_csv_path (str): Path to the CSV file containing name mappings

    Returns:
        dict: Mapping by db_id with table and column name mappings
    """
    df = pd.read_csv(mapping_csv_path)

    db_mappings = {}

    for _, row in df.iterrows():
        db_id = row["db_id"]
        table_name = row["table_name"]
        column_name = row["column_name"]
        new_table_name = row["new_table_name"]
        new_column_name = row["new_column_name"]

        # Initialize mapping for this db_id if it doesn't exist
        if db_id not in db_mappings:
            db_mappings[db_id] = {
                "table_mappings": {},  # old_name -> new_name
                "column_mappings": {},  # old_column -> new_column
            }

        # Add table mapping if it changed
        if table_name != new_table_name:
            db_mappings[db_id]["table_mappings"][table_name] = new_table_name

        # Add column mapping if it changed
        if column_name != new_column_name:
            db_mappings[db_id]["column_mappings"][column_name] = new_column_name

    return db_mappings


def replace_sql_names(sql_query, table_mappings, column_mappings):
    """
    Replace table and column names in SQL query with new names.

    Args:
        sql_query (str): Original SQL query
        table_mappings (dict): Mapping of old table names to new table names
        column_mappings (dict): Mapping of old column names to new column names

    Returns:
        str: Updated SQL query with new names
    """
    updated_sql = sql_query

    # Replace table names
    for old_table, new_table in table_mappings.items():
        # Use word boundaries and case-insensitive matching
        # Handle both quoted and unquoted table names
        patterns = [
            rf"\b{re.escape(old_table)}\b",  # Unquoted table name
            rf"`{re.escape(old_table)}`",  # Backtick quoted
            rf'"{re.escape(old_table)}"',  # Double quoted
            rf"'{re.escape(old_table)}'",  # Single quoted
        ]

        replacements = [
            new_table,  # Unquoted replacement
            f"`{new_table}`",  # Backtick quoted replacement
            f'"{new_table}"',  # Double quoted replacement
            f"'{new_table}'",  # Single quoted replacement
        ]

        for pattern, replacement in zip(patterns, replacements):
            updated_sql = re.sub(pattern, replacement, updated_sql, flags=re.IGNORECASE)

    # Replace column names - but avoid aggregation functions and ORDER BY
    for old_column, new_column in column_mappings.items():
        # Define common SQL aggregation functions
        agg_functions = [
            "count",
            "sum",
            "avg",
            "min",
            "max",
            "group_concat",
            "total",
            "abs",
            "upper",
            "lower",
            "length",
            "substr",
            "trim",
            "round",
            "cast",
            "coalesce",
            "ifnull",
            "nullif",
        ]

        # Check if the column name is an aggregation function
        if old_column.lower() in agg_functions:
            # Skip replacement for aggregation functions
            continue

        # Special handling for 'order' - don't replace if followed by 'by'
        if old_column.lower() == "order":
            # Don't replace 'order' if it's part of 'ORDER BY'
            # Use a more specific pattern that excludes 'ORDER BY'
            patterns = [
                rf"\b{re.escape(old_column)}\b(?!\s*by)(?!\s*\()",  # Not followed by 'by' or '('
                rf"`{re.escape(old_column)}`(?!\s*by)(?!\s*\()",  # Backtick quoted, not followed by 'by' or '('
                rf'"{re.escape(old_column)}"(?!\s*by)(?!\s*\()',  # Double quoted, not followed by 'by' or '('
                rf"'{re.escape(old_column)}'",  # Single quoted (less common for columns)
            ]
        else:
            # Handle both quoted and unquoted column names
            patterns = [
                rf"\b{re.escape(old_column)}\b(?!\s*\()",  # Unquoted, not followed by (
                rf"`{re.escape(old_column)}`(?!\s*\()",  # Backtick quoted, not followed by (
                rf'"{re.escape(old_column)}"(?!\s*\()',  # Double quoted, not followed by (
                rf"'{re.escape(old_column)}'",  # Single quoted (less common for columns)
            ]

        replacements = [
            new_column,  # Unquoted replacement
            f"`{new_column}`",  # Backtick quoted replacement
            f'"{new_column}"',  # Double quoted replacement
            f"'{new_column}'",  # Single quoted replacement
        ]

        for pattern, replacement in zip(patterns, replacements):
            updated_sql = re.sub(pattern, replacement, updated_sql, flags=re.IGNORECASE)

    return updated_sql


def save_results_to_csv(validated_queries, csv_output_path):
    """
    Save validation results to CSV file with original and new SQL columns.

    Args:
        validated_queries (list): List of validated query data with EX field
        csv_output_path (str): Path to output CSV file
    """
    # Prepare data for CSV
    csv_data = []

    for query in validated_queries:
        csv_row = {
            "question_id": query.get("question_id"),
            "db_id": query.get("db_id"),
            "question": query.get("question"),
            "evidence": query.get("evidence", ""),
            "original_sql": query.get("original_sql", ""),  # Store original SQL
            "new_sql": query.get("SQL", ""),  # Store updated SQL
            "difficulty": query.get("difficulty", ""),
            "EX": query.get("EX", 0),
        }
        csv_data.append(csv_row)

    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output_path, index=False, encoding="utf-8")

    print(f"Results saved to CSV: {csv_output_path}")
    return df


def validate_sql_with_ex(
    original_queries,
    updated_queries,
    original_db_root,
    new_db_root,
    target_db_ids,
    timeout_seconds=60,
):
    """
    Validate SQL changes using execution comparison between old and new databases.

    Args:
        original_queries (list): List of original query data
        updated_queries (list): List of updated query data
        original_db_root (str): Root path to original databases
        new_db_root (str): Root path to new databases
        target_db_ids (list): List of database IDs to validate
        timeout_seconds (int): Timeout for each query evaluation

    Returns:
        list: Updated queries with EX field and original_sql field added
    """
    print("Starting EX metric validation...")

    validation_stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "timeout_queries": 0,
        "error_queries": 0,
        "db_results": {},
    }

    # Create mapping of original queries by question_id for quick lookup
    original_query_map = {q["question_id"]: q for q in original_queries}
    validated_queries = []

    for updated_query in tqdm(updated_queries, desc="Validating queries with EX"):
        db_id = updated_query.get("db_id")
        question_id = updated_query.get("question_id")

        # Initialize EX field and store original SQL
        updated_query_copy = updated_query.copy()
        updated_query_copy["EX"] = 0

        # Store original SQL for reference
        if question_id in original_query_map:
            updated_query_copy["original_sql"] = original_query_map[question_id]["SQL"]
        else:
            updated_query_copy["original_sql"] = ""

        # Skip if not in target databases
        if db_id not in target_db_ids:
            validated_queries.append(updated_query_copy)
            continue

        validation_stats["total_queries"] += 1

        # Initialize db results if not exists
        if db_id not in validation_stats["db_results"]:
            validation_stats["db_results"][db_id] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "timeout": 0,
                "errors": 0,
            }

        validation_stats["db_results"][db_id]["total"] += 1

        # Get original query
        if question_id not in original_query_map:
            print(f"Warning: Original query not found for question_id {question_id}")
            validated_queries.append(updated_query_copy)
            continue

        original_query = original_query_map[question_id]

        try:
            with timeout(timeout_seconds):
                gt_db_handler = DatabaseHandler(
                    dbms=DBMS.SQLITE,
                    connection_params={
                        "dbms": DBMS.SQLITE,
                        "db_path": f"{original_db_root}/{db_id}/{db_id}.sqlite",
                    },
                )

                new_db_handler = DatabaseHandler(
                    dbms=DBMS.SQLITE,
                    connection_params={
                        "dbms": DBMS.SQLITE,
                        "db_path": f"{new_db_root}/{db_id}/{db_id}.sqlite",
                    },
                )

                # Run original query on original database
                gt_cols, gt_rows = gt_db_handler.run_query(original_query["SQL"])

                # Run updated query on new database
                pred_cols, pred_rows = new_db_handler.run_query(updated_query["SQL"])

                # Calculate EX metric
                ex = 1 if set(gt_rows) == set(pred_rows) else 0
                updated_query_copy["EX"] = ex

                if ex == 1:
                    validation_stats["successful_queries"] += 1
                    validation_stats["db_results"][db_id]["successful"] += 1
                else:
                    validation_stats["failed_queries"] += 1
                    validation_stats["db_results"][db_id]["failed"] += 1

        except TimeoutError:
            validation_stats["timeout_queries"] += 1
            validation_stats["db_results"][db_id]["timeout"] += 1
            updated_query_copy["EX"] = 0
            print(f"Timeout for query {question_id} in {db_id}")

        except Exception as e:
            validation_stats["error_queries"] += 1
            validation_stats["db_results"][db_id]["errors"] += 1
            updated_query_copy["EX"] = 0
            print(f"Error evaluating query {question_id} in {db_id}: {str(e)}")

        validated_queries.append(updated_query_copy)

    return validated_queries, validation_stats


def print_ex_validation_summary(validation_stats):
    """
    Print comprehensive summary of EX validation results.

    Args:
        validation_stats (dict): Results from validate_sql_with_ex
    """
    print(f"\n{'='*70}")
    print("EX METRIC VALIDATION SUMMARY")
    print(f"{'='*70}")

    total = validation_stats["total_queries"]
    successful = validation_stats["successful_queries"]
    failed = validation_stats["failed_queries"]
    timeout = validation_stats["timeout_queries"]
    errors = validation_stats["error_queries"]

    success_rate = (successful / total * 100) if total > 0 else 0

    print("📊 Overall Statistics:")
    print(f"  • Total queries validated: {total}")
    print(f"  • Successful (EX=1): {successful} ({success_rate:.1f}%)")
    print(f"  • Failed (EX=0): {failed}")
    print(f"  • Timeout errors: {timeout}")
    print(f"  • Other errors: {errors}")

    print("\n📋 Per-Database Results:")
    for db_id, db_stats in validation_stats["db_results"].items():
        if db_stats["total"] > 0:
            db_success_rate = db_stats["successful"] / db_stats["total"] * 100

            print(f"  • {db_id}:")
            print(f"    - Total: {db_stats['total']}")
            print(
                f"    - Success (EX=1): {db_stats['successful']} ({db_success_rate:.1f}%)"
            )
            print(f"    - Failed (EX=0): {db_stats['failed']}")
            print(
                f"    - Timeouts: {db_stats['timeout']}, Errors: {db_stats['errors']}"
            )

    print("\n🎯 Final Assessment:")
    if success_rate >= 90:
        print(f"🎉 EXCELLENT: {success_rate:.1f}% perfect matches (EX=1)!")
    elif success_rate >= 70:
        print(f"✅ GOOD: {success_rate:.1f}% perfect matches (EX=1)")
    elif success_rate >= 50:
        print(f"⚠️  MODERATE: {success_rate:.1f}% perfect matches (EX=1)")
    else:
        print(f"❌ POOR: Only {success_rate:.1f}% perfect matches (EX=1)")

    print(f"{'='*70}")


def process_sql_queries(
    queries_json_path, output_json_path, mapping_csv_path, target_db_ids
):
    """
    Process SQL queries and update table/column names based on mappings.

    Args:
        queries_json_path (str): Path to input JSON file with queries
        output_json_path (str): Path to output JSON file
        mapping_csv_path (str): Path to CSV file with name mappings
        target_db_ids (list): List of database IDs to process
    """
    # Read name mappings
    print("Reading name mappings...")
    db_mappings = read_name_mapping(mapping_csv_path)

    # Read input JSON file
    print(f"Loading queries from {queries_json_path}...")
    with open(queries_json_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Track statistics
    stats = {
        "total_queries": len(queries),
        "processed_queries": 0,
        "changed_queries": 0,
        "skipped_queries": 0,
        "db_stats": {},
    }

    # Process each query
    print("Processing SQL queries...")
    updated_queries = []

    for query_data in tqdm(queries, desc="Processing queries"):
        db_id = query_data.get("db_id")
        original_sql = query_data.get("SQL", "")

        # Initialize db stats if not exists
        if db_id not in stats["db_stats"]:
            stats["db_stats"][db_id] = {"total": 0, "processed": 0, "changed": 0}

        stats["db_stats"][db_id]["total"] += 1

        # Check if this database should be processed
        if db_id not in target_db_ids:
            # Keep original query unchanged
            updated_queries.append(query_data)
            stats["skipped_queries"] += 1
            continue

        # Check if we have mappings for this database
        if db_id not in db_mappings:
            # Keep original query unchanged
            updated_queries.append(query_data)
            stats["skipped_queries"] += 1
            continue

        # Get mappings for this database
        table_mappings = db_mappings[db_id].get("table_mappings", {})
        column_mappings = db_mappings[db_id].get("column_mappings", {})

        # Update SQL query
        updated_sql = replace_sql_names(original_sql, table_mappings, column_mappings)

        # Create updated query data
        updated_query_data = query_data.copy()
        updated_query_data["SQL"] = updated_sql
        updated_queries.append(updated_query_data)

        # Update statistics
        stats["processed_queries"] += 1
        stats["db_stats"][db_id]["processed"] += 1

        if updated_sql != original_sql:
            stats["changed_queries"] += 1
            stats["db_stats"][db_id]["changed"] += 1

    # Save updated queries
    print(f"Saving updated queries to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(updated_queries, f, indent=2, ensure_ascii=False)

    return stats


def print_summary(stats):
    """
    Print a comprehensive summary of the processing results.

    Args:
        stats (dict): Statistics from processing
    """
    print(f"\n{'='*60}")
    print("SQL QUERY ALTERATION SUMMARY")
    print(f"{'='*60}")

    # Overall statistics
    overall_success_rate = (
        (stats["changed_queries"] / stats["processed_queries"] * 100)
        if stats["processed_queries"] > 0
        else 0
    )

    print("📊 Processing Statistics:")
    print(f"  • Total queries: {stats['total_queries']}")
    print(f"  • Processed queries: {stats['processed_queries']}")
    print(f"  • Skipped queries: {stats['skipped_queries']}")
    print(f"  • Changed queries: {stats['changed_queries']}")

    print("\n✅ Success Rate:")
    print(
        f"  • Queries modified: {stats['changed_queries']}/{stats['processed_queries']} ({overall_success_rate:.1f}%)"
    )

    # Per-database breakdown
    print("\n📋 Per-Database Breakdown:")
    for db_id, db_stats in stats["db_stats"].items():
        if db_stats["processed"] > 0:
            db_success_rate = (
                (db_stats["changed"] / db_stats["processed"] * 100)
                if db_stats["processed"] > 0
                else 0
            )
            print(
                f"  • {db_id}: {db_stats['changed']}/{db_stats['processed']} queries changed ({db_success_rate:.1f}%)"
            )
        elif db_stats["total"] > 0:
            print(f"  • {db_id}: {db_stats['total']} queries skipped (no mappings)")

    print("\n🎯 Final Result:")
    if overall_success_rate >= 50:
        print(
            f"🎉 SUCCESS: {overall_success_rate:.1f}% of processed queries were modified!"
        )
    elif overall_success_rate >= 25:
        print(
            f"⚠️  WARNING: Only {overall_success_rate:.1f}% of processed queries were modified"
        )
    else:
        print(
            f"❌ LOW IMPACT: Only {overall_success_rate:.1f}% of processed queries were modified"
        )

    print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    queries_json_path = "data/benchmarks/Bird/minidev/MINIDEV/mini_dev_sqlite.json"
    mapping_csv_path = "data/augmentation/snail/databases_naturalness_decreased.csv"
    results_csv_path = "data/augmentation/snail/new_sql_queries.csv"

    # EX validation configuration
    original_db_root = "data/benchmarks/Bird/minidev/MINIDEV/dev_databases"
    new_db_root = "data/augmentation/snail/new_dev_databases"
    timeout_seconds = 30  # Timeout for each query evaluation

    # List of database IDs to process
    target_db_ids = [
        "california_schools",
        "card_games",
        "codebase_community",
        "debit_card_specializing",
        "european_football_2",
        "financial",
        "formula_1",
        "student_club",
        "superhero",
        "thrombosis_prediction",
        "toxicology",
    ]

    print("Processing SQL queries...")
    print(f"Input JSON: {queries_json_path}")
    print(f"Output CSV: {results_csv_path}")
    print(f"Mappings CSV: {mapping_csv_path}")
    print(f"Target databases: {len(target_db_ids)} databases")
    print()

    # Process the queries
    temp_json_path = "temp_updated_queries.json"
    stats = process_sql_queries(
        queries_json_path, temp_json_path, mapping_csv_path, target_db_ids
    )

    # Print comprehensive summary
    print_summary(stats)

    # Run EX validation
    print(f"\n{'='*60}")
    print("STARTING EX METRIC VALIDATION")
    print(f"{'='*60}")
    print(f"Original DB root: {original_db_root}")
    print(f"New DB root: {new_db_root}")
    print(f"Timeout per query: {timeout_seconds}s")
    print()

    # Load original and updated queries
    with open(queries_json_path, "r", encoding="utf-8") as f:
        original_queries = json.load(f)

    with open(temp_json_path, "r", encoding="utf-8") as f:
        updated_queries = json.load(f)

    # Run validation and get queries with EX field
    validated_queries, validation_stats = validate_sql_with_ex(
        original_queries,
        updated_queries,
        original_db_root,
        new_db_root,
        target_db_ids,
        timeout_seconds,
    )

    # Save results to CSV
    results_df = save_results_to_csv(validated_queries, results_csv_path)

    # Print validation results
    print_ex_validation_summary(validation_stats)

    # Print summary statistics
    successful_count = len(results_df[results_df["EX"] == 1])
    failed_count = len(results_df[results_df["EX"] == 0])

    print(f"\n📁 All results saved to CSV: {results_csv_path}")
    print(f"📊 Total queries: {len(results_df)}")
    print(f"📊 Successful queries (EX=1): {successful_count}")
    print(f"📊 Failed queries (EX=0): {failed_count}")

    # Clean up temporary file
    import os

    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

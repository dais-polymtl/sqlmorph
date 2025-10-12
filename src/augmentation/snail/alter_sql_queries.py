import json
import pandas as pd
import re
from tqdm import tqdm


def read_name_mapping(csv_path):
    """
    Read the name mapping CSV file and organize it by database ID.

    Args:
        csv_path (str): Path to the CSV file containing name mappings

    Returns:
        dict: Mapping by db_id with table and column name mappings
    """
    df = pd.read_csv(csv_path)

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

    # Replace column names
    for old_column, new_column in column_mappings.items():
        # Handle both quoted and unquoted column names
        patterns = [
            rf"\b{re.escape(old_column)}\b",  # Unquoted column name
            rf"`{re.escape(old_column)}`",  # Backtick quoted
            rf'"{re.escape(old_column)}"',  # Double quoted
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


def process_sql_queries(input_json_path, output_json_path, csv_path, target_db_ids):
    """
    Process SQL queries and update table/column names based on mappings.

    Args:
        input_json_path (str): Path to input JSON file with queries
        output_json_path (str): Path to output JSON file
        csv_path (str): Path to CSV file with name mappings
        target_db_ids (list): List of database IDs to process
    """
    # Read name mappings
    print("Reading name mappings...")
    db_mappings = read_name_mapping(csv_path)

    # Read input JSON file
    print(f"Loading queries from {input_json_path}...")
    with open(input_json_path, "r", encoding="utf-8") as f:
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
    input_json_path = "data/benchmarks/Bird/bird_dev.json"
    output_json_path = "data/benchmarks/Bird/new_bird_dev.json"
    csv_path = "data/augmentation/snail/databases_naturalness_decreased.csv"

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
    print(f"Input JSON: {input_json_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Mappings CSV: {csv_path}")
    print(f"Target databases: {len(target_db_ids)} databases")
    print()

    # Process the queries
    stats = process_sql_queries(
        input_json_path, output_json_path, csv_path, target_db_ids
    )

    # Print comprehensive summary
    print_summary(stats)

    print(f"\n📁 Updated queries saved to: {output_json_path}")

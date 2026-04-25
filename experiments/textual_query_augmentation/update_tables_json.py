import json
import pandas as pd
from collections import defaultdict


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


def update_table_names(table_names_original, table_mappings):
    """
    Update table names using the mapping.

    Args:
        table_names_original (list): Original table names
        table_mappings (dict): Table name mappings

    Returns:
        list: Updated table names
    """
    updated_table_names = []

    for table_name in table_names_original:
        if table_name in table_mappings:
            updated_table_names.append(table_mappings[table_name])
        else:
            updated_table_names.append(table_name)

    return updated_table_names


def update_column_names(column_names_original, column_mappings):
    """
    Update column names using the mapping.

    Args:
        column_names_original (list): Original column names in format [[table_idx, column_name], ...]
        column_mappings (dict): Column name mappings

    Returns:
        list: Updated column names
    """
    updated_column_names = []

    for column_entry in column_names_original:
        table_idx, column_name = column_entry

        # Keep the special "*" column as is
        if column_name == "*":
            updated_column_names.append(column_entry)
        elif column_name in column_mappings:
            updated_column_names.append([table_idx, column_mappings[column_name]])
        else:
            updated_column_names.append(column_entry)

    return updated_column_names


def update_tables_json(json_input_path, json_output_path, csv_path):
    """
    Update the tables JSON file with new table and column names.

    Args:
        json_input_path (str): Path to input JSON file
        json_output_path (str): Path to output JSON file
        csv_path (str): Path to CSV file with name mappings
    """
    # Read name mappings
    print("Reading name mappings...")
    db_mappings = read_name_mapping(csv_path)

    # Read input JSON file
    print(f"Loading JSON from {json_input_path}...")
    with open(json_input_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # Track statistics
    stats = {
        "total_entries": len(tables_data),
        "updated_entries": 0,
        "updated_tables": 0,
        "updated_columns": 0,
        "db_stats": defaultdict(lambda: {"tables_changed": 0, "columns_changed": 0}),
    }

    # Process each entry
    updated_tables_data = []

    for entry in tables_data:
        db_id = entry.get("db_id")

        if db_id in db_mappings:
            print(f"Updating entry for database: {db_id}")

            # Get mappings for this database
            table_mappings = db_mappings[db_id].get("table_mappings", {})
            column_mappings = db_mappings[db_id].get("column_mappings", {})

            # Create updated entry
            updated_entry = entry.copy()

            # Update table names
            if "table_names_original" in entry:
                original_tables = entry["table_names_original"]
                updated_tables = update_table_names(original_tables, table_mappings)
                updated_entry["table_names_original"] = updated_tables

                # Count changes
                tables_changed = sum(
                    1 for old, new in zip(original_tables, updated_tables) if old != new
                )
                stats["updated_tables"] += tables_changed
                stats["db_stats"][db_id]["tables_changed"] = tables_changed

            # Update column names
            if "column_names_original" in entry:
                original_columns = entry["column_names_original"]
                updated_columns = update_column_names(original_columns, column_mappings)
                updated_entry["column_names_original"] = updated_columns

                # Count changes
                columns_changed = sum(
                    1
                    for old, new in zip(original_columns, updated_columns)
                    if old != new
                )
                stats["updated_columns"] += columns_changed
                stats["db_stats"][db_id]["columns_changed"] = columns_changed

            updated_tables_data.append(updated_entry)
            stats["updated_entries"] += 1
        else:
            # No mappings for this database, keep original
            updated_tables_data.append(entry)

    # Save updated JSON
    print(f"Saving updated JSON to {json_output_path}...")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(updated_tables_data, f, indent=2, ensure_ascii=False)

    return stats


def print_update_summary(stats):
    """
    Print comprehensive summary of the update results.

    Args:
        stats (dict): Statistics from update process
    """
    print(f"\n{'='*60}")
    print("TABLE JSON UPDATE SUMMARY")
    print(f"{'='*60}")

    print("📊 Processing Statistics:")
    print(f"  • Total entries: {stats['total_entries']}")
    print(f"  • Updated entries: {stats['updated_entries']}")
    print(f"  • Total table names changed: {stats['updated_tables']}")
    print(f"  • Total column names changed: {stats['updated_columns']}")

    if stats["db_stats"]:
        print("\n📋 Per-Database Breakdown:")
        for db_id, db_stats in stats["db_stats"].items():
            print(f"  • {db_id}:")
            print(f"    - Table names changed: {db_stats['tables_changed']}")
            print(f"    - Column names changed: {db_stats['columns_changed']}")

    print("\n🎯 Final Result:")
    if stats["updated_entries"] > 0:
        print(f"🎉 SUCCESS: Updated {stats['updated_entries']} database entries!")
    else:
        print("⚠️  WARNING: No entries were updated")

    print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    csv_path = "data/augmentation/decrease_naturalness/databases_naturalness_decreased_fixed.csv"
    json_input_path = "data/benchmarks/Bird/minidev/MINIDEV/dev_tables.json"
    json_output_path = "data/augmentation/decrease_naturalness/experiment_dev_sql_failed/new_dev_databases/new_dev_tables.json"

    print("Updating tables JSON file...")
    print(f"Input CSV: {csv_path}")
    print(f"Input JSON: {json_input_path}")
    print(f"Output JSON: {json_output_path}")
    print()

    # Process the update
    stats = update_tables_json(json_input_path, json_output_path, csv_path)

    # Print comprehensive summary
    print_update_summary(stats)

    print(f"\n📁 Updated tables JSON saved to: {json_output_path}")

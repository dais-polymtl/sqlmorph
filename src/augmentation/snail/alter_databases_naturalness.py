import os
import sqlite3
import pandas as pd
import shutil
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
                "column_mappings": {},  # (table_name, old_column) -> new_column
            }

        # Add table mapping if it changed
        if table_name != new_table_name:
            db_mappings[db_id]["table_mappings"][table_name] = new_table_name

        # Add column mapping if it changed
        if column_name != new_column_name:
            # Use new table name as the key for consistency
            final_table_name = (
                new_table_name if table_name != new_table_name else table_name
            )
            db_mappings[db_id]["column_mappings"][
                (final_table_name, column_name)
            ] = new_column_name

    return db_mappings


def copy_database_structure(db_ids, input_root, output_root):
    """
    Copy all databases maintaining the same structure.

    Args:
        db_ids (list): List of database IDs to copy
        input_root (str): Root directory of original databases
        output_root (str): Root directory for copied databases
    """
    print("Copying database structure...")

    for db_id in tqdm(db_ids, desc="Copying databases"):
        input_db_dir = os.path.join(input_root, db_id)
        output_db_dir = os.path.join(output_root, db_id)

        # Create output directory
        os.makedirs(output_db_dir, exist_ok=True)

        # Copy the entire database directory
        if os.path.exists(input_db_dir):
            # Copy all files in the database directory
            for file_name in os.listdir(input_db_dir):
                src_file = os.path.join(input_db_dir, file_name)
                dst_file = os.path.join(output_db_dir, file_name)

                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        else:
            print(f"Warning: Directory not found: {input_db_dir}")


def get_table_info(conn, table_name):
    """
    Get detailed information about a table including foreign keys.

    Args:
        conn: SQLite connection
        table_name (str): Name of the table

    Returns:
        dict: Table information
    """
    cursor = conn.cursor()

    # Get column information
    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
    columns = cursor.fetchall()

    # Get foreign key information
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
    foreign_keys = cursor.fetchall()

    return {"columns": columns, "foreign_keys": foreign_keys}


def alter_table_name(conn, old_name, new_name):
    """
    Rename a table using ALTER TABLE.

    Args:
        conn: SQLite connection
        old_name (str): Current table name
        new_name (str): New table name
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"ALTER TABLE `{old_name}` RENAME TO `{new_name}`")
        conn.commit()
        print(f"Renamed table: {old_name} -> {new_name}")
        return True
    except sqlite3.Error as e:
        print(f"Error renaming table {old_name} to {new_name}: {e}")
        return False


def alter_column_name(conn, table_name, old_column, new_column):
    """
    Rename a column using ALTER TABLE.

    Args:
        conn: SQLite connection
        table_name (str): Name of the table
        old_column (str): Current column name
        new_column (str): New column name
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"ALTER TABLE `{table_name}` RENAME COLUMN `{old_column}` TO `{new_column}`"
        )
        conn.commit()
        print(f"Renamed column in {table_name}: {old_column} -> {new_column}")
        return True
    except sqlite3.Error as e:
        print(
            f"Error renaming column {old_column} to {new_column} in table {table_name}: {e}"
        )
        return False


def alter_database_schema(db_path, mappings):
    """
    Alter the schema of a database using the provided mappings.

    Args:
        db_path (str): Path to the database file
        mappings (dict): Name mappings for tables and columns
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")  # Disable foreign keys during alteration

    try:
        table_mappings = mappings.get("table_mappings", {})
        column_mappings = mappings.get("column_mappings", {})

        # First, rename all columns (before renaming tables)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            # Check if this table will be renamed
            final_table_name = table_mappings.get(table_name, table_name)

            # Rename columns for this table
            for (mapped_table, old_column), new_column in column_mappings.items():
                if mapped_table == final_table_name:  # Use the final table name
                    alter_column_name(conn, table_name, old_column, new_column)

        # Then rename tables
        for old_table_name, new_table_name in table_mappings.items():
            alter_table_name(conn, old_table_name, new_table_name)

        conn.commit()
        print(f"Successfully altered database schema: {db_path}")

    except Exception as e:
        print(f"Error altering database {db_path}: {str(e)}")
        conn.rollback()

    finally:
        conn.execute("PRAGMA foreign_keys = ON")  # Re-enable foreign keys
        conn.close()


def process_databases(db_ids, input_root, output_root, csv_path):
    """
    Main function to copy databases and alter their schemas.

    Args:
        db_ids (list): List of database IDs to process
        input_root (str): Root directory of original databases
        output_root (str): Root directory for modified databases
        csv_path (str): Path to the CSV file with name mappings
    """
    # Ensure output root exists
    os.makedirs(output_root, exist_ok=True)

    # Step 1: Copy all databases maintaining structure
    copy_database_structure(db_ids, input_root, output_root)

    # Step 2: Read name mappings
    print("Reading name mappings...")
    db_mappings = read_name_mapping(csv_path)

    # Step 3: Alter schemas for databases that have mappings
    print("Altering database schemas...")

    for db_id in tqdm(db_ids, desc="Altering schemas"):
        db_path = os.path.join(output_root, db_id, f"{db_id}.sqlite")

        if os.path.exists(db_path):
            if db_id in db_mappings:
                print(f"Altering schema for {db_id}...")
                alter_database_schema(db_path, db_mappings[db_id])
            else:
                print(f"No mappings found for {db_id}, keeping original schema")
        else:
            print(f"Warning: Database file not found: {db_path}")


def validate_changes(db_ids, output_root, db_mappings):
    """
    Validate that the schema changes were applied correctly.

    Args:
        db_ids (list): List of database IDs
        output_root (str): Root directory of modified databases
        db_mappings (dict): Name mappings that were applied

    Returns:
        dict: Validation results summary
    """
    print("Validating schema changes...")

    validation_results = {
        "total_dbs": 0,
        "successful_dbs": 0,
        "total_tables": 0,
        "successful_tables": 0,
        "total_columns": 0,
        "successful_columns": 0,
        "failed_dbs": [],
        "failed_tables": [],
        "failed_columns": [],
    }

    for db_id in db_ids:
        if db_id not in db_mappings:
            continue

        validation_results["total_dbs"] += 1
        db_success = True

        db_path = os.path.join(output_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            validation_results["failed_dbs"].append(f"{db_id} (file not found)")
            continue

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Check that renamed tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            existing_tables = [row[0] for row in cursor.fetchall()]

            table_mappings = db_mappings[db_id].get("table_mappings", {})
            for old_name, new_name in table_mappings.items():
                validation_results["total_tables"] += 1
                if new_name in existing_tables:
                    print(
                        f"✓ Table {old_name} successfully renamed to {new_name} in {db_id}"
                    )
                    validation_results["successful_tables"] += 1
                else:
                    print(f"✗ Table {new_name} not found in {db_id}")
                    validation_results["failed_tables"].append(
                        f"{db_id}.{old_name} -> {new_name}"
                    )
                    db_success = False

            # Check that renamed columns exist
            column_mappings = db_mappings[db_id].get("column_mappings", {})
            for (table_name, old_column), new_column in column_mappings.items():
                validation_results["total_columns"] += 1
                try:
                    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                    columns = [col[1] for col in cursor.fetchall()]

                    if new_column in columns:
                        print(
                            f"✓ Column {old_column} successfully renamed to {new_column} in {db_id}.{table_name}"
                        )
                        validation_results["successful_columns"] += 1
                    else:
                        print(
                            f"✗ Column {new_column} not found in {db_id}.{table_name}"
                        )
                        validation_results["failed_columns"].append(
                            f"{db_id}.{table_name}.{old_column} -> {new_column}"
                        )
                        db_success = False
                except sqlite3.Error:
                    print(
                        f"✗ Could not check columns for table {table_name} in {db_id}"
                    )
                    validation_results["failed_columns"].append(
                        f"{db_id}.{table_name}.{old_column} -> {new_column} (table error)"
                    )
                    db_success = False

            if db_success:
                validation_results["successful_dbs"] += 1
            else:
                validation_results["failed_dbs"].append(db_id)

        finally:
            conn.close()

    return validation_results


if __name__ == "__main__":
    # Configuration
    input_root = "data/benchmarks/Bird/dev_databases"
    output_root = "data/benchmarks/Bird/new_dev_databases"
    csv_path = "data/augmentation/snail/databases_naturalness_decreased.csv"

    # List of database IDs to process
    db_ids = [
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

    print(f"Processing {len(db_ids)} databases...")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Mappings file: {csv_path}")
    print()

    # Process all databases
    process_databases(db_ids, input_root, output_root, csv_path)

    # Read mappings for validation
    db_mappings = read_name_mapping(csv_path)

    # Validate the changes
    validation_results = validate_changes(db_ids, output_root, db_mappings)

    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("DATABASE ALTERATION SUMMARY")
    print(f"{'='*60}")

    # Calculate success rates
    db_success_rate = (
        (validation_results["successful_dbs"] / validation_results["total_dbs"] * 100)
        if validation_results["total_dbs"] > 0
        else 0
    )
    table_success_rate = (
        (
            validation_results["successful_tables"]
            / validation_results["total_tables"]
            * 100
        )
        if validation_results["total_tables"] > 0
        else 0
    )
    column_success_rate = (
        (
            validation_results["successful_columns"]
            / validation_results["total_columns"]
            * 100
        )
        if validation_results["total_columns"] > 0
        else 0
    )

    total_changes = (
        validation_results["total_tables"] + validation_results["total_columns"]
    )
    successful_changes = (
        validation_results["successful_tables"]
        + validation_results["successful_columns"]
    )
    overall_success_rate = (
        (successful_changes / total_changes * 100) if total_changes > 0 else 0
    )

    print("📊 Processing Statistics:")
    print(f"  • Databases processed: {validation_results['total_dbs']}")
    print(f"  • Tables to rename: {validation_results['total_tables']}")
    print(f"  • Columns to rename: {validation_results['total_columns']}")
    print(f"  • Total schema changes: {total_changes}")

    print("\n✅ Success Statistics:")
    print(
        f"  • Databases successful: {validation_results['successful_dbs']}/{validation_results['total_dbs']} ({db_success_rate:.1f}%)"
    )
    print(
        f"  • Tables renamed: {validation_results['successful_tables']}/{validation_results['total_tables']} ({table_success_rate:.1f}%)"
    )
    print(
        f"  • Columns renamed: {validation_results['successful_columns']}/{validation_results['total_columns']} ({column_success_rate:.1f}%)"
    )
    print(
        f"  • Overall success: {successful_changes}/{total_changes} ({overall_success_rate:.1f}%)"
    )

    if (
        validation_results["failed_dbs"]
        or validation_results["failed_tables"]
        or validation_results["failed_columns"]
    ):
        print("\n❌ Failed Operations:")
        if validation_results["failed_dbs"]:
            print(
                f"  • Failed databases: {', '.join(validation_results['failed_dbs'])}"
            )
        if validation_results["failed_tables"]:
            print(
                f"  • Failed table renames: {len(validation_results['failed_tables'])}"
            )
        if validation_results["failed_columns"]:
            print(
                f"  • Failed column renames: {len(validation_results['failed_columns'])}"
            )

    print("\n🎯 Final Result:")
    if overall_success_rate == 100:
        print("🎉 SUCCESS: All schema changes applied successfully!")
    elif overall_success_rate >= 90:
        print(f"⚠️  WARNING: {100-overall_success_rate:.1f}% of changes failed")
    else:
        print(f"❌ FAILURE: {100-overall_success_rate:.1f}% of changes failed")

    print(f"\n📁 Output saved to: {output_root}")
    print(f"{'='*60}")

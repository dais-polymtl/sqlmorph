from naturalness_classifiers.snails_naturalness_classifier import (
    CanineIdentifierClassifier,
)
import os
import sqlite3
import csv


def extract_schema(db_path):
    """
    Extract schema from SQLite database.
    Returns a dictionary with table names as keys and lists of column names as values.
    """
    # Verify database file exists
    if not os.path.exists(db_path):
        print(f"Database file not found at path: {db_path}")
        return {}

    # Connect directly to the SQLite database to avoid wrapper issues
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        print(f"Tables found: {table_names}")

        schema = {}
        for table_name in table_names:
            print(f"Processing table: {table_name}")

            # Get columns for each table
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns_info = cursor.fetchall()

            # Extract column names (index 1 in SQLite PRAGMA table_info)
            columns = [col[1] for col in columns_info]
            schema[table_name] = columns

            print(f"Columns for {table_name}: {columns}")

        conn.close()
        return schema

    except Exception as e:
        print(f"Error extracting schema: {str(e)}")
        return {}


def classify_schema_naturalness(schema_dict, classifier):
    """
    Classify the naturalness of table and column names.
    Returns a dictionary with natural names mapped to original names.
    """
    # Dictionary to store results
    naturalness_dict = {}

    for table_name, columns in schema_dict.items():
        # Classify table name
        table_result = classifier.classify_identifier(table_name)

        # Add debugging to see the structure
        print(f"Debug - table result for {table_name}: {table_result}")

        # Handle result based on its actual structure
        if isinstance(table_result, list) and table_result:
            label = table_result[0].get("label", "unknown")
            score = table_result[0].get("score", 0.0)
        else:
            label = "unknown"
            score = 0.0

        # Store table classification information using the label and score variables we already determined
        naturalness_dict[table_name] = {
            "naturalness": label,
            "confidence": score,
            "columns": {},
        }

        # Classify column names
        for column in columns:
            column_result = classifier.classify_identifier(column)

            # Handle column result based on its structure
            if isinstance(column_result, list) and column_result:
                col_label = column_result[0].get("label", "unknown")
                col_score = column_result[0].get("score", 0.0)
            else:
                col_label = "unknown"
                col_score = 0.0

            # Store column classification information
            naturalness_dict[table_name]["columns"][column] = {
                "naturalness": col_label,
                "confidence": col_score,
            }

    return naturalness_dict


def write_naturalness_to_csv(
    naturalness_results, dataset, db_id, output_path, write_header=False
):
    """
    Write naturalness results to CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mode = "w" if write_header else "a"
    with open(output_path, mode, newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "dataset",
            "db_id",
            "table_name",
            "table_naturalness",
            "table_confidence",
            "column_name",
            "column_naturalness",
            "column_confidence",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for table_name, table_info in naturalness_results.items():
            for column_name, column_info in table_info["columns"].items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "db_id": db_id,
                        "table_name": table_name,
                        "table_naturalness": table_info["naturalness"],
                        "table_confidence": round(table_info["confidence"], 4),
                        "column_name": column_name,
                        "column_naturalness": column_info["naturalness"],
                        "column_confidence": round(column_info["confidence"], 4),
                    }
                )


def check_db_exists_in_csv(csv_path, dataset, db_id):
    """
    Check if database already exists in CSV file.
    """
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["dataset"] == dataset and row["db_id"] == db_id:
                return True
    return False


def process_databases(dataset, db_ids):
    """
    Process multiple databases and classify their schema naturalness.
    """
    # Determine dataset path
    if dataset == "bird/minidev":
        path = "Bird/minidev/MINIDEV/dev_databases"
    elif dataset == "bird/dev":
        path = "Bird/dev_databases"
    elif dataset == "bird/train":
        path = "Bird/train/train_databases"
    else:
        print(f"Unknown dataset: {dataset}")
        return

    print(f"Processing {len(db_ids)} databases: {db_ids}")

    # Single CSV file for all databases
    csv_output_path = "data/augmentation/snail/databases_naturalness.csv"
    write_header = not os.path.exists(csv_output_path)

    # Initialize the classifier once
    classifier = CanineIdentifierClassifier()

    # Process each database
    for i, db_id in enumerate(db_ids):
        print(f"\n{'='*50}")
        print(f"Processing database: {db_id} ({i+1}/{len(db_ids)})")
        print(f"{'='*50}")

        # Check if database already exists in CSV
        if check_db_exists_in_csv(csv_output_path, dataset, db_id):
            print(f"Database {db_id} already exists in CSV, skipping...")
            continue

        db_path = f"data/benchmarks/{path}/{db_id}/{db_id}.sqlite"

        print(f"Using database path: {db_path}")
        print(f"File exists: {os.path.exists(db_path)}")

        if not os.path.exists(db_path):
            print(f"Skipping {db_id} - database file not found")
            continue

        # Extract schema
        print("Extracting schema...")
        schema = extract_schema(db_path)
        print(f"Found {len(schema)} tables")

        if not schema:
            print(f"No tables found in {db_id}, skipping...")
            continue

        # Classify naturalness
        print("Classifying identifiers...")
        naturalness_results = classify_schema_naturalness(schema, classifier)

        # Print results
        print("\nSchema Naturalness Results:")
        for table, table_info in naturalness_results.items():
            print(
                f"Table: {table} - {table_info['naturalness']} (confidence: {table_info['confidence']:.4f})"
            )
            print("Columns:")
            for column, column_info in table_info["columns"].items():
                print(
                    f"  - {column}: {column_info['naturalness']} (confidence: {column_info['confidence']:.4f})"
                )
            print()

        # Write results to single CSV file
        write_naturalness_to_csv(
            naturalness_results, dataset, db_id, csv_output_path, write_header
        )
        write_header = False  # Only write header for the first database
        print(f"Results appended to: {csv_output_path}")

    print(f"\n{'='*50}")
    print(f"Completed processing {len(db_ids)} databases")
    print(f"All results saved to: {csv_output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Configuration parameters
    dataset = "bird/dev"  #
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

    # Output CSV file path
    csv_output_path = "data/augmentation/snail/databases_naturalness.csv"

    # Process databases
    process_databases(dataset, db_ids)

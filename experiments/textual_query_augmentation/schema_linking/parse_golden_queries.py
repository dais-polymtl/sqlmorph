"""
Extract schema linking information from Bird development dataset SQL queries.
"""

import json
import sqlglot
from sqlglot import expressions as exp
import re
from pathlib import Path


def extract_schema_linking(sql_query, dialect="sqlite"):
    """
    Extract schema linking information from a SQL query.

    Args:
        sql_query: SQL query string
        dialect: SQL dialect (default: sqlite)

    Returns:
        Dictionary mapping table names to lists of column names
    """
    try:
        # Clean up any ANSI escape codes
        clean_query = re.sub(r"\x1b\[[0-9;]*m", "", sql_query)
        parsed = sqlglot.parse_one(clean_query, dialect=dialect)

        schema_linking = {}
        tables = list(parsed.find_all(exp.Table))

        # --- Logic for single-table queries ---
        if len(tables) == 1:
            # Get the actual table name, ignoring any alias it might have
            table_name = tables[0].name
            schema_linking[table_name] = set()

            # Find all column references
            for column in parsed.find_all(exp.Column):
                # Add every column found, as it must belong to the single table
                schema_linking[table_name].add(column.name)
        else:
            # --- Original logic for multi-table queries (JOINs) ---
            # This part requires columns to be explicitly qualified (e.g., table.column)
            alias_to_table = {t.alias_or_name: t.name for t in tables}

            for column in parsed.find_all(exp.Column):
                if column.table:  # Only consider columns with explicit table references
                    table_alias = column.table
                    table_name = alias_to_table.get(table_alias)
                    if table_name:
                        if table_name not in schema_linking:
                            schema_linking[table_name] = set()
                        schema_linking[table_name].add(column.name)

        # Convert sets to lists for the final output
        return {table: list(columns) for table, columns in schema_linking.items()}
    except Exception as e:
        print(f"Error parsing query: {sql_query[:100]}... Error: {e}")
        return {}


def process_bird_dev_dataset(input_path, output_path):
    """
    Process Bird development dataset and extract schema linking for each query.

    Args:
        input_path: Path to bird_dev.json
        output_path: Path to save the output JSON file
    """
    # Load the input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each question and extract schema linking
    results = {}
    for item in data:
        question_id = str(item["question_id"])
        sql_query = item["SQL"]

        # Extract schema linking
        schema_linking = extract_schema_linking(sql_query, dialect="sqlite")
        results[question_id] = schema_linking

        # Print progress every 100 items
        if int(question_id) % 100 == 0:
            print(f"Processed question_id: {question_id}")

    # Save results to output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete! Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    input_path = "data/benchmarks/Bird/bird_dev.json"
    output_path = "data/augmentation/decrease_naturalness/schema_linking/dev_golden_sql_parsed.json"

    process_bird_dev_dataset(input_path, output_path)

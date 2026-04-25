import pandas as pd
import json
from pathlib import Path


def join_data_with_golden(
    csv_path: Path, json_path: Path, mapping_csv_path: Path, output_path: Path
):
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    if output_path.exists():
        print(f"Output file {output_path} already exists. Skipping processing.")
        return

    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")

    # Read JSON file
    with open(json_path, "r") as f:
        golden_schema = json.load(f)
    print(f"Loaded {len(golden_schema)} entries from JSON")

    # Read mapping CSV
    mapping_df = pd.read_csv(mapping_csv_path)
    print(f"Loaded {len(mapping_df)} rows from mapping CSV")

    # Create mapping dictionaries for each db_id
    # Structure: {db_id: {table_name: {new_table_name: str, columns: {old_col: new_col}}}}
    db_mappings = {}
    for _, row in mapping_df.iterrows():
        db_id = row["db_id"]
        if db_id not in db_mappings:
            db_mappings[db_id] = {}

        table_name = row["table_name"]
        if table_name not in db_mappings[db_id]:
            db_mappings[db_id][table_name] = {
                "new_table_name": row["new_table_name"],
                "columns": {},
            }

        column_name = row["column_name"]
        new_column_name = row["new_column_name"]
        db_mappings[db_id][table_name]["columns"][column_name] = new_column_name

    print(f"Created mappings for {len(db_mappings)} databases")

    # Convert question_id to string for matching with JSON keys
    df["original_gold_schema"] = (
        df["question_id"]
        .astype(str)
        .apply(
            lambda qid: (
                json.dumps(golden_schema.get(qid, "N/A"))
                if qid in golden_schema
                else "N/A"
            )
        )
    )

    # Create new_gold_schema column by transforming original_gold_schema using mappings
    def transform_schema(row):
        if row["original_gold_schema"] == "N/A":
            return "N/A"

        db_id = row["db_id"]
        if db_id not in db_mappings:
            return "N/A"

        try:
            original_schema = json.loads(row["original_gold_schema"])
            new_schema = {}

            for old_table, old_columns in original_schema.items():
                if old_table in db_mappings[db_id]:
                    new_table = db_mappings[db_id][old_table]["new_table_name"]
                    new_columns = []

                    for old_col in old_columns:
                        if old_col in db_mappings[db_id][old_table]["columns"]:
                            new_col = db_mappings[db_id][old_table]["columns"][old_col]
                            new_columns.append(new_col)
                        else:
                            # If column not found in mapping, keep original
                            new_columns.append(old_col)

                    new_schema[new_table] = new_columns
                else:
                    # If table not found in mapping, keep original
                    new_schema[old_table] = old_columns

            return json.dumps(new_schema)
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"Error transforming schema for question_id {row['question_id']}: {e}"
            )
            return "N/A"

    df["new_gold_schema"] = df.apply(transform_schema, axis=1)

    # Reorder columns to put difficulty and EX at the end
    cols = df.columns.tolist()
    # Remove difficulty and EX from their current positions
    cols.remove("difficulty")
    cols.remove("EX")
    # Add them at the end
    cols = cols + ["difficulty", "EX"]
    df = df[cols]

    # Ensure EX is numeric and keep only rows with EX == 1
    df["EX"] = pd.to_numeric(df["EX"], errors="coerce").fillna(0)
    df = df[df["EX"] == 1]

    # Write to new CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote {len(df)} rows to {output_path}")

    # Print summary statistics
    num_matched_original = (df["original_gold_schema"] != "N/A").sum()
    num_unmatched_original = (df["original_gold_schema"] == "N/A").sum()
    num_matched_new = (df["new_gold_schema"] != "N/A").sum()
    num_unmatched_new = (df["new_gold_schema"] == "N/A").sum()

    print(
        f"\nOriginal Gold Schema - Matched: {num_matched_original}, Unmatched: {num_unmatched_original}"
    )
    print(
        f"New Gold Schema - Matched: {num_matched_new}, Unmatched: {num_unmatched_new}"
    )


if __name__ == "__main__":
    # Define file paths
    csv_path = Path(
        "data/augmentation/decrease_naturalness/experiment_dev_sql_failed/experiments_dev_new_sql_nl_queries_fixed.csv"
    )
    json_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking_dev/dev_golden_sql_parsed.json"
    )
    mapping_csv_path = Path(
        "data/augmentation/decrease_naturalness/databases_naturalness_decreased_fixed.csv"
    )
    output_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking_dev_sql_failed/new_sql_nl_schema_queries.csv"
    )

    join_data_with_golden(csv_path, json_path, mapping_csv_path, output_path)

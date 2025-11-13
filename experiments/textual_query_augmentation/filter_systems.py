import pandas as pd
import json
import ast
import os


def read_systems_data(csv_path):
    """
    Read systems data from CSV file.

    Args:
        csv_path (str): Path to the systems data CSV file

    Returns:
        pd.DataFrame: Systems data
    """
    df = pd.read_csv(csv_path)
    return df


def extract_ex_from_execution_accuracy(execution_accuracy_str):
    """
    Extract EX value from EXECUTION_ACCURACY column string.

    Args:
        execution_accuracy_str (str): String representation of execution accuracy dict

    Returns:
        int: EX value (0 or 1)
    """
    try:
        # Parse the string as a dictionary
        exec_dict = ast.literal_eval(execution_accuracy_str)
        return exec_dict.get("EX", 0)
    except (ValueError, SyntaxError):
        return 0


def filter_system_successful_questions(systems_df, system_name):
    """
    Filter system entries where EXECUTION_ACCURACY has EX == 1.

    Args:
        systems_df (pd.DataFrame): Systems data DataFrame
        system_name (str): Name of the system to filter

    Returns:
        set: Set of question_ids that passed
    """
    # Filter for specified system
    system_df = systems_df[systems_df["system"] == system_name].copy()

    # Extract EX values from EXECUTION_ACCURACY column
    system_df["EX"] = system_df["EXECUTION_ACCURACY"].apply(
        extract_ex_from_execution_accuracy
    )

    # Filter for EX == 1
    successful_system = system_df[system_df["EX"] == 1]

    # Get unique question_ids
    successful_question_ids = set(successful_system["question_id"].tolist())

    print(f"Total {system_name} entries: {len(system_df)}")
    print(f"Successful {system_name} entries (EX=1): {len(successful_system)}")
    print(f"Unique successful question_ids: {len(successful_question_ids)}")

    return successful_question_ids


def filter_json_by_question_ids(json_path, question_ids):
    """
    Filter JSON file by question_ids.

    Args:
        json_path (str): Path to the JSON file
        question_ids (set): Set of question_ids to keep

    Returns:
        list: Filtered list of entries
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Filter entries by question_id
    filtered_data = [
        entry for entry in json_data if entry.get("question_id") in question_ids
    ]

    print(f"Original JSON entries: {len(json_data)}")
    print(f"Filtered JSON entries: {len(filtered_data)}")

    return filtered_data


def save_filtered_json(filtered_data, output_path):
    """
    Save filtered data to JSON file.

    Args:
        filtered_data (list): Filtered data
        output_path (str): Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered data saved to: {output_path}")


def read_sql_nl_queries(csv_path):
    """
    Read SQL NL queries CSV file and filter by EX == 1.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Filtered dataframe with EX == 1
    """
    df = pd.read_csv(csv_path)

    # Filter for EX == 1
    successful_df = df[df["EX"] == 1].copy()

    print(f"Total queries in CSV: {len(df)}")
    print(f"Successful queries (EX=1): {len(successful_df)}")

    return successful_df


def create_json_variants(filtered_df, system_name, output_dir):
    """
    Create four JSON variants from the filtered dataframe.

    Args:
        filtered_df (pd.DataFrame): Filtered dataframe
        system_name (str): Name of the system
        output_dir (str): Base output directory
    """
    # Create system-specific directory
    system_output_dir = os.path.join(output_dir, system_name)
    os.makedirs(system_output_dir, exist_ok=True)

    variants = [
        {
            "suffix": "original_nl_original_sql",
            "question_col": "original_question",
            "evidence_col": "original_evidence",
            "sql_col": "original_sql",
        },
        {
            "suffix": "original_nl_new_sql",
            "question_col": "original_question",
            "evidence_col": "original_evidence",
            "sql_col": "new_sql",
        },
        {
            "suffix": "new_nl_original_sql",
            "question_col": "new_question",
            "evidence_col": "new_evidence",
            "sql_col": "original_sql",
        },
        {
            "suffix": "new_nl_new_sql",
            "question_col": "new_question",
            "evidence_col": "new_evidence",
            "sql_col": "new_sql",
        },
    ]

    for variant in variants:
        # Create JSON data for this variant
        json_data = []

        for _, row in filtered_df.iterrows():
            # Handle NaN values by converting to empty string
            evidence_value = row[variant["evidence_col"]]
            if pd.isna(evidence_value):
                evidence_value = ""

            entry = {
                "question_id": row["question_id"],
                "db_id": row["db_id"],
                "question": row[variant["question_col"]],
                "evidence": evidence_value,
                "SQL": row[variant["sql_col"]],
                "difficulty": row["difficulty"],
            }
            json_data.append(entry)

        # Create output file path in system directory
        output_file = os.path.join(system_output_dir, f"{variant['suffix']}.json")

        # Save JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Created {variant['suffix']}: {output_file} ({len(json_data)} entries)")


def write_summary_to_file(systems, sql_nl_df, output_dir, system_stats):
    """
    Write processing summary to a text file.

    Args:
        systems (list): List of systems processed
        sql_nl_df (pd.DataFrame): SQL NL queries dataframe
        output_dir (str): Output directory
        system_stats (dict): Statistics for each system
    """
    summary_file = os.path.join(output_dir, "processing_summary.txt")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("SYSTEMS FILTERING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("Processing Statistics:\n")
        f.write(f"  • Systems processed: {len(systems)}\n")
        f.write(f"  • SQL NL queries with EX=1: {len(sql_nl_df)}\n")
        f.write("  • JSON variants generated per system: 4\n")
        f.write(f"  • Output directory: {output_dir}\n\n")

        f.write("Per-System Results:\n")
        for system_name, stats in system_stats.items():
            f.write(f"\n{system_name.upper()}:\n")
            f.write(f"  • Successful questions: {stats['successful_questions']}\n")
            f.write(f"  • Final filtered entries: {stats['filtered_entries']}\n")
            if stats["filtered_entries"] > 0:
                f.write(
                    f"  • Generated 4 JSON variants in: {os.path.join(output_dir, system_name)}\n"
                )
                f.write("  • JSON files created:\n")
                for variant in [
                    "original_nl_original_sql",
                    "original_nl_new_sql",
                    "new_nl_original_sql",
                    "new_nl_new_sql",
                ]:
                    f.write(f"    - {variant}.json\n")
            else:
                f.write("  • No matching queries found\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    # Configuration
    systems = ["din-sql", "mac-sql", "chess"]  # List of systems to process

    # File paths
    systems_csv_path = "data/metrics/experiments/system_level_comparison/systems_data_with_metrics-with_penalty.csv"
    sql_nl_csv_path = "data/augmentation/snail/new_sql_nl_queries.csv"
    output_dir = "data/augmentation/snail/experiments"  # Base output directory

    print("Reading systems data...")
    systems_df = read_systems_data(systems_csv_path)

    print("\nReading SQL NL queries CSV...")
    sql_nl_df = read_sql_nl_queries(sql_nl_csv_path)

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track system statistics for summary
    system_stats = {}

    # Process each system
    for system_name in systems:
        print(f"\n{'='*60}")
        print(f"PROCESSING SYSTEM: {system_name.upper()}")
        print(f"{'='*60}")

        print(f"Filtering {system_name} system successful questions...")
        successful_question_ids = filter_system_successful_questions(
            systems_df, system_name
        )

        print("Filtering queries by successful question_ids...")
        # Filter SQL NL queries by successful question_ids
        filtered_queries = sql_nl_df[
            sql_nl_df["question_id"].isin(successful_question_ids)
        ].copy()

        print(
            f"Queries matching successful {system_name} questions: {len(filtered_queries)}"
        )

        # Store statistics
        system_stats[system_name] = {
            "successful_questions": len(successful_question_ids),
            "filtered_entries": len(filtered_queries),
        }

        if len(filtered_queries) > 0:
            print(f"Creating JSON variants for {system_name}...")
            create_json_variants(filtered_queries, system_name, output_dir)
        else:
            print(f"No matching queries found for {system_name}")

        print(f"\n📊 {system_name.upper()} Summary:")
        print(f"  • Successful questions: {len(successful_question_ids)}")
        print(f"  • Final filtered entries: {len(filtered_queries)}")
        if len(filtered_queries) > 0:
            print(
                f"  • Generated 4 JSON variants in: {os.path.join(output_dir, system_name)}"
            )

    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print("📊 Processing Statistics:")
    print(f"  • Systems processed: {len(systems)}")
    print(f"  • SQL NL queries with EX=1: {len(sql_nl_df)}")
    print("  • JSON variants generated per system: 4")
    print(f"  • Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Write summary to file
    write_summary_to_file(systems, sql_nl_df, output_dir, system_stats)

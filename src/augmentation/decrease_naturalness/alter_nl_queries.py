import pandas as pd
import os
import json
from tqdm import tqdm
from collections import defaultdict
from src.core.model_manager import ModelManager, ModelType, ModelProvider, OpenAIModel
from src.core.model_manager.utils import compose_chat_messages
from src.core.prompt_renderer import PromptRenderer


def read_schema_mapping(mapping_csv_path):
    """
    Read the name mapping CSV file and organize it by database ID.

    Args:
        mapping_csv_path (str): Path to the CSV file containing name mappings

    Returns:
        dict: Mapping by db_id with complete table and column structure
    """
    df = pd.read_csv(mapping_csv_path)

    # Organize mappings hierarchically by db_id, table, and columns
    db_mappings = defaultdict(lambda: defaultdict(dict))

    # First pass: collect all tables and their mappings
    for _, row in df.iterrows():
        db_id = row["db_id"]
        table_name = row["table_name"]
        new_table_name = row["new_table_name"]

        # Initialize column mappings dictionary for this table
        if "columns" not in db_mappings[db_id][table_name]:
            db_mappings[db_id][table_name]["new_name"] = new_table_name
            db_mappings[db_id][table_name]["columns"] = {}

    # Second pass: collect all columns for each table
    for _, row in df.iterrows():
        db_id = row["db_id"]
        table_name = row["table_name"]
        column_name = row["column_name"]
        new_column_name = row["new_column_name"]

        # Add column mapping
        db_mappings[db_id][table_name]["columns"][column_name] = new_column_name

    return db_mappings


def format_schema_mapping(db_schema_mapping):
    """
    Format schema mapping for prompt display with tables and columns organized.

    Args:
        db_schema_mapping (dict): Schema mapping for a database

    Returns:
        str: Formatted schema mapping string
    """
    if not db_schema_mapping:
        return "No schema mappings available for this database."

    formatted_lines = []

    # Process each table
    for table_name, table_info in db_schema_mapping.items():
        new_table_name = table_info.get("new_name", table_name)

        # Add table header with arrow if it changed
        if table_name == new_table_name:
            formatted_lines.append(f"Table {table_name}:")
        else:
            formatted_lines.append(f"Table {table_name} -> {new_table_name}:")

        # Add column mappings
        columns = table_info.get("columns", {})
        if not columns:
            formatted_lines.append("(No columns)")
        else:
            for col_name, new_col_name in columns.items():
                if col_name == new_col_name:
                    formatted_lines.append(f"{col_name}")
                else:
                    formatted_lines.append(f"{col_name} -> {new_col_name}")

        # Add blank line between tables
        formatted_lines.append("")

    return "\n".join(formatted_lines).rstrip()


def load_existing_results(output_csv_path):
    """
    Load existing results from output CSV if it exists.

    Args:
        output_csv_path (str): Path to output CSV file

    Returns:
        dict: Dictionary of processed entries by (db_id, question_id)
    """
    if not os.path.exists(output_csv_path):
        return {}

    try:
        existing_df = pd.read_csv(output_csv_path)
        processed_entries = {}

        for _, row in existing_df.iterrows():
            key = (row["db_id"], row["question_id"])
            processed_entries[key] = {
                "new_question": row.get("new_question", ""),
                "new_evidence": row.get("new_evidence", ""),
            }

        print(f"Found {len(processed_entries)} existing processed entries")
        return processed_entries

    except Exception as e:
        print(f"Error loading existing results: {str(e)}")
        return {}


def write_result_to_csv(output_csv_path, row_data, is_first_write=False):
    """
    Write a single result row to CSV file.

    Args:
        output_csv_path (str): Path to output CSV file
        row_data (dict): Row data to write
        is_first_write (bool): Whether this is the first write (include header)
    """
    # Convert row_data to DataFrame
    df_row = pd.DataFrame([row_data])

    # Write to CSV
    if is_first_write:
        df_row.to_csv(output_csv_path, index=False, encoding="utf-8", mode="w")
    else:
        df_row.to_csv(
            output_csv_path, index=False, encoding="utf-8", mode="a", header=False
        )


def generate_less_natural_nl(
    question,
    evidence,
    schema_mapping,
    model_name=OpenAIModel.GPT_4O,
    seed=42,
    temperature=0,
):
    """
    Generate less natural question and evidence using LLM prompting.

    Args:
        question (str): Original question
        evidence (str): Original evidence
        schema_mapping (dict): Schema mapping for the database
        model_name: Model to use for generation
        seed (int): Random seed for reproducibility
        temperature (float): Temperature for generation

    Returns:
        tuple: (new_question, new_evidence)
    """
    # Initialize prompt renderer
    prompt_renderer = PromptRenderer(
        templates_dir_path="src/augmentation/snail/prompts_templates/"
    )

    # Initialize LLM
    llm = ModelManager.create_model(
        model_provider=ModelProvider.OPENAI,
        model_type=ModelType.COMPLETION,
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY", None),
    )

    try:
        # Format schema mapping for prompt
        formatted_schema_mapping = format_schema_mapping(schema_mapping)

        # Render prompt
        prompt_context = {
            "_SCHEMA_MAPPING_": formatted_schema_mapping,
            "_QUESTION_": question,
            "_EVIDENCE_": evidence,
        }
        prompt = prompt_renderer.render(
            "fewshot-decrease-nl-naturalness", prompt_context
        )

        # Compose messages and get LLM response
        messages = compose_chat_messages(user_messages=[prompt])
        llm_response = llm.get_chat_completion(
            messages=messages,
            seed=seed,
            temperature=temperature,
        )

        raw_response = llm_response["completion_content"][0].strip()

        # Remove code block markers if they exist
        if raw_response.startswith("```json"):
            raw_response = raw_response.replace("```json", "", 1)
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]

        raw_response = raw_response.strip()

        # Parse JSON response
        try:
            response_json = json.loads(raw_response)
            new_question = response_json.get("updated_question", question)
            new_evidence = response_json.get("updated_evidence", evidence)

            print("✓ Generated NL for question")
            return new_question, new_evidence

        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON response: {raw_response}")
            print(f"Error: {str(e)}")
            return question, evidence

    except Exception as e:
        print(f"✗ Error generating NL: {str(e)}")
        return question, evidence


def process_nl_queries(
    queries_csv_path,
    mapping_csv_path,
    output_csv_path,
    model_name=OpenAIModel.GPT_4O,
    seed=42,
    temperature=0,
):
    """
    Process natural language queries to make them less natural.
    Resume from existing progress if output file exists.
    """
    # Read schema mappings
    print("Reading schema mappings...")
    db_mappings = read_schema_mapping(mapping_csv_path)

    # Read queries CSV
    print(f"Loading queries from {queries_csv_path}...")
    df = pd.read_csv(queries_csv_path)

    # Load existing processed results
    existing_results = load_existing_results(output_csv_path)

    # Rename existing columns to original_* if not already renamed
    if "original_question" not in df.columns:
        df = df.rename(
            columns={"question": "original_question", "evidence": "original_evidence"}
        )

    # Initialize new columns if they don't exist
    if "new_question" not in df.columns:
        df["new_question"] = ""
    if "new_evidence" not in df.columns:
        df["new_evidence"] = ""

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Track statistics
    stats = {
        "total_queries": len(df),
        "processed_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "skipped_existing": 0,
        "db_stats": {},
    }

    # Check if we need to write header (first time)
    is_first_write = not os.path.exists(output_csv_path) or len(existing_results) == 0

    # Process each row
    print("Processing natural language queries...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        db_id = row["db_id"]
        question_id = row["question_id"]
        original_question = row["original_question"]
        original_evidence = row["original_evidence"]

        # Initialize db stats if not exists
        if db_id not in stats["db_stats"]:
            stats["db_stats"][db_id] = {
                "total": 0,
                "processed": 0,
                "successful": 0,
                "skipped": 0,
            }

        stats["db_stats"][db_id]["total"] += 1

        # Check if this entry is already processed
        entry_key = (db_id, question_id)
        if entry_key in existing_results:
            print(
                f"⏭️  Skipping already processed entry: {db_id}, question_id {question_id}"
            )
            stats["skipped_existing"] += 1
            stats["db_stats"][db_id]["skipped"] += 1
            continue

        stats["processed_queries"] += 1

        # Get schema mapping for this database
        schema_mapping = db_mappings.get(db_id, {})

        # Prepare row data for writing
        row_data = {
            "question_id": question_id,
            "db_id": db_id,
            "original_question": original_question,
            "original_evidence": original_evidence,
            "new_question": original_question,  # Default to original
            "new_evidence": original_evidence,  # Default to original
        }

        # Copy other columns if they exist
        for col in df.columns:
            if col not in [
                "question_id",
                "db_id",
                "original_question",
                "original_evidence",
                "new_question",
                "new_evidence",
            ]:
                row_data[col] = row[col]

        if not schema_mapping:
            print(f"⚠️  No schema mapping found for {db_id}, keeping original NL")
        else:
            # Generate less natural NL
            new_question, new_evidence = generate_less_natural_nl(
                original_question,
                original_evidence,
                schema_mapping,
                model_name,
                seed,
                temperature,
            )

            # Update row data with generated results
            row_data["new_question"] = new_question
            row_data["new_evidence"] = new_evidence

            stats["db_stats"][db_id]["processed"] += 1

            # Check if generation was successful (different from original)
            if new_question != original_question or new_evidence != original_evidence:
                stats["successful_queries"] += 1
                stats["db_stats"][db_id]["successful"] += 1
            else:
                stats["failed_queries"] += 1

        # Write result immediately to CSV
        write_result_to_csv(output_csv_path, row_data, is_first_write)
        is_first_write = False  # Only first write includes header

        print(f"✅ Processed and saved: {db_id}, question_id {question_id}")

    print(f"Processed queries saved to: {output_csv_path}")
    return stats


def print_nl_processing_summary(stats):
    """
    Print comprehensive summary of NL processing results.
    """
    print(f"\n{'='*70}")
    print("NATURAL LANGUAGE PROCESSING SUMMARY")
    print(f"{'='*70}")

    total = stats["total_queries"]
    processed = stats["processed_queries"]
    successful = stats["successful_queries"]
    failed = stats["failed_queries"]
    skipped = stats["skipped_existing"]

    success_rate = (successful / processed * 100) if processed > 0 else 0

    print("📊 Processing Statistics:")
    print(f"  • Total queries: {total}")
    print(f"  • Skipped (already processed): {skipped}")
    print(f"  • Newly processed queries: {processed}")
    print(f"  • Successfully modified: {successful} ({success_rate:.1f}%)")
    print(f"  • Failed/unchanged: {failed}")

    print("\n📋 Per-Database Results:")
    for db_id, db_stats in stats["db_stats"].items():
        if db_stats["total"] > 0:
            db_success_rate = (
                (db_stats["successful"] / db_stats["processed"] * 100)
                if db_stats["processed"] > 0
                else 0
            )

            print(f"  • {db_id}:")
            print(f"    - Total: {db_stats['total']}")
            print(f"    - Skipped: {db_stats.get('skipped', 0)}")
            print(f"    - Newly processed: {db_stats['processed']}")
            print(f"    - Modified: {db_stats['successful']} ({db_success_rate:.1f}%)")

    print("\n🎯 Final Assessment:")
    if success_rate >= 80:
        print(f"🎉 EXCELLENT: {success_rate:.1f}% of queries successfully modified!")
    elif success_rate >= 60:
        print(f"✅ GOOD: {success_rate:.1f}% of queries successfully modified")
    elif success_rate >= 40:
        print(f"⚠️  MODERATE: {success_rate:.1f}% of queries successfully modified")
    else:
        print(f"❌ POOR: Only {success_rate:.1f}% of queries successfully modified")

    print(f"{'='*70}")


if __name__ == "__main__":
    # Configuration
    queries_csv_path = "data/augmentation/snail/new_sql_queries.csv"
    mapping_csv_path = "data/augmentation/snail/databases_naturalness_decreased.csv"
    output_csv_path = "data/augmentation/snail/new_sql_nl_queries.csv"

    # Model configuration
    model_name = OpenAIModel.GPT_4O  # Model to use for generation
    seed = 42  # Random seed for reproducibility
    temperature = 0  # Temperature for generation

    print("Processing natural language queries...")
    print(f"Input queries CSV: {queries_csv_path}")
    print(f"Schema mappings CSV: {mapping_csv_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print()

    # Process the queries
    stats = process_nl_queries(
        queries_csv_path,
        mapping_csv_path,
        output_csv_path,
        model_name,
        seed,
        temperature,
    )

    # Print comprehensive summary
    print_nl_processing_summary(stats)

    print(f"\n📁 Updated natural language queries saved to: {output_csv_path}")

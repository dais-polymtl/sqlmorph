import pandas as pd
import os
from src.core.model_manager import ModelManager, ModelType, ModelProvider, OpenAIModel
from src.core.model_manager.utils import compose_chat_messages
from src.core.prompt_renderer import PromptRenderer


def generate_less_natural_name(
    current_name,
    current_naturalness,
    model_name=OpenAIModel.GPT_4O,
    seed=42,
    temperature=0,
):
    """
    Generate a less natural name using LLM prompting.
    """
    # If already N3, return original
    if current_naturalness == "N3":
        return current_name, "N3"

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

    # Select appropriate template based on current naturalness
    if current_naturalness == "N1":
        template_name = "fewshot-N1-to-N3"
    elif current_naturalness == "N2":
        template_name = "fewshot-N2-to-N3"
    else:
        # Fallback case - shouldn't happen if logic is correct
        return current_name, "N3"

    try:
        # Render prompt
        prompt = prompt_renderer.render(template_name, {"_IDENTIFIER_": current_name})

        # Compose messages and get LLM response
        messages = compose_chat_messages(user_messages=[prompt])
        llm_response = llm.get_chat_completion(
            messages=messages,
            seed=seed,
            temperature=temperature,
        )

        raw_response = llm_response["completion_content"][0].strip()

        # Sanity check: if response contains "->" extract the second part
        if "->" in raw_response:
            parts = raw_response.split("->")
            if len(parts) >= 2:
                new_name = parts[1].strip()
            else:
                new_name = raw_response
        else:
            new_name = raw_response

        print(f"{current_name} ({current_naturalness}) -> {new_name} (N3)")

        return new_name, "N3"

    except Exception as e:
        print(f"Error for '{current_name}': {str(e)}")
        return "error", "error"


def process_naturalness_decrease(
    input_csv_path,
    output_csv_path,
    model_name=OpenAIModel.GPT_4O,
    seed=42,
    temperature=0,
):
    """
    Process the naturalness CSV to decrease naturalness levels to N3.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # Initialize new columns
    df["new_table_name"] = ""
    df["new_table_naturalness"] = ""
    df["new_column_name"] = ""
    df["new_column_naturalness"] = ""

    # Cache for table names per database to avoid redundant LLM calls
    table_cache = (
        {}
    )  # Format: {(db_id, table_name, table_naturalness): (new_name, new_naturalness)}

    # Process each row
    for idx, row in df.iterrows():
        db_id = row["db_id"]

        # Process table naturalness with caching
        table_cache_key = (db_id, row["table_name"], row["table_naturalness"])

        if row["table_naturalness"] == "N3":
            # Already N3, keep original
            df.at[idx, "new_table_name"] = row["table_name"]
            df.at[idx, "new_table_naturalness"] = "N3"
        elif table_cache_key in table_cache:
            # Use cached result
            cached_table_name, cached_table_naturalness = table_cache[table_cache_key]
            df.at[idx, "new_table_name"] = cached_table_name
            df.at[idx, "new_table_naturalness"] = cached_table_naturalness
            print(f"Using cached table: {row['table_name']} -> {cached_table_name}")
        else:
            # Need to decrease naturalness to N3
            new_table_name, new_table_naturalness = generate_less_natural_name(
                row["table_name"], row["table_naturalness"], model_name, seed
            )
            # Cache the result
            table_cache[table_cache_key] = (new_table_name, new_table_naturalness)

            df.at[idx, "new_table_name"] = (
                new_table_name if new_table_name else row["table_name"]
            )
            df.at[idx, "new_table_naturalness"] = (
                new_table_naturalness if new_table_naturalness else "N3"
            )

        # Process column naturalness (no caching needed as column names are typically unique)
        if row["column_naturalness"] == "N3":
            # Already N3, keep original
            df.at[idx, "new_column_name"] = row["column_name"]
            df.at[idx, "new_column_naturalness"] = "N3"
        else:
            # Need to decrease naturalness to N3
            new_column_name, new_column_naturalness = generate_less_natural_name(
                row["column_name"],
                row["column_naturalness"],
                model_name,
                seed,
                temperature,
            )
            df.at[idx, "new_column_name"] = (
                new_column_name if new_column_name else row["column_name"]
            )
            df.at[idx, "new_column_naturalness"] = (
                new_column_naturalness if new_column_naturalness else "N3"
            )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save the processed dataframe
    df.to_csv(output_csv_path, index=False)
    print(f"Processed naturalness data saved to: {output_csv_path}")

    # Print summary statistics
    tables_changed = len(df[df["table_name"] != df["new_table_name"]])
    columns_changed = len(df[df["column_name"] != df["new_column_name"]])
    unique_tables_processed = len(table_cache)

    print("Summary:")
    print(f"- Total rows processed: {len(df)}")
    print(f"- Unique tables processed: {unique_tables_processed}")
    print(f"- Table names changed: {tables_changed}")
    print(f"- Column names changed: {columns_changed}")

    return df


if __name__ == "__main__":
    # File paths
    input_path = "data/augmentation/snail/databases_naturalness.csv"
    output_path = "data/augmentation/snail/databases_naturalness_decreased.csv"

    # Configuration parameters
    model_name = OpenAIModel.GPT_4O  # Model to use for generation
    seed = 42  # Random seed for reproducibility
    temperature = 0  # Temperature for generation (fixed in function)

    # Process the naturalness decrease
    processed_df = process_naturalness_decrease(
        input_path, output_path, model_name, seed, temperature
    )

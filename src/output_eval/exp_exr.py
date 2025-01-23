import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
from dotenv import load_dotenv


load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate a single embedding for the given text using the specified model.
    """
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_aggregated_embedding(text, model="text-embedding-ada-002", chunk_size=2000):
    """
    Generate an aggregated embedding for a large text by splitting it into chunks.
    - Splits text into smaller chunks of size `chunk_size`.
    - Computes embeddings for each chunk and averages them.
    """
    if not text.strip():
        return np.zeros(1536).tolist()  # Return zero vector for empty input.

    # Split the text into manageable chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = []

    # Generate embeddings for each chunk
    for chunk in chunks:
        try:
            embedding = get_embedding(chunk, model)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for chunk: {chunk[:100]}... - {e}")

    if not embeddings:
        return np.zeros(1536).tolist()  # Return zero vector if no embeddings are generated.

    # Return the average embedding of all chunks
    return np.mean(embeddings, axis=0).tolist()


def compute_column_similarity_with_embeddings(df1, df2, threshold=0.8):
    """
    Compute similarity between columns of two DataFrames using embeddings.
    - Matches columns where similarity exceeds the threshold.
    """
    similarity_scores = []

    # Compare all columns in df1 with all columns in df2
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Combine all values in each column into a single string
            text1 = " ".join(map(str, df1[col1].dropna().values))
            text2 = " ".join(map(str, df2[col2].dropna().values))

            # Get embeddings for each column
            embedding1 = get_aggregated_embedding(text1)
            embedding2 = get_aggregated_embedding(text2)

            if np.isnan(embedding1).any() or np.isnan(embedding2).any():
                continue  # Skip columns with invalid embeddings.

            # Compute cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]

            if similarity > threshold:
                similarity_scores.append((col1, col2, similarity))

    # Sort matches by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    # Match columns without duplication
    matched_columns = []
    used_columns_pred = set()
    used_columns_truth = set()

    for col1, col2, sim in similarity_scores:
        if col1 not in used_columns_truth and col2 not in used_columns_pred:
            matched_columns.append((col1, col2))
            used_columns_truth.add(col1)
            used_columns_pred.add(col2)

    return matched_columns


def calculate_field_metrics(df1, df2, threshold=0.8):
    """
    Calculate field-level metrics (EXP_fields, EXR_fields) based on similarity between columns of two DataFrames.
    """
    # Find similar columns based on embeddings
    similar_columns = compute_column_similarity_with_embeddings(df1, df2, threshold)

    # Initialize counters for field matches
    correctly_predicted_fields = 0
    extra_predicted_fields = 0
    missing_fields = 0

    # Get all ground truth and predicted columns
    ground_truth_columns = set(df1.columns)
    predicted_columns = set(df2.columns)


    # For fields, we compare column names and match based on similarity
    matched_columns_truth = {pair[0] for pair in similar_columns}
    matched_columns_pred = {pair[1] for pair in similar_columns}

    # Correctly predicted fields
    correctly_predicted_fields = len(matched_columns_truth)

    # Extra predicted fields (predicted columns with no match)
    extra_predicted_fields = len(predicted_columns - matched_columns_truth)

    # Missing fields (ground truth columns with no match)
    missing_fields = len(ground_truth_columns - matched_columns_pred)
    print("correctly_predicted_fields", correctly_predicted_fields)
    print("extra_predicted_fields", extra_predicted_fields)
    print("missing_fields", missing_fields)

    # Calculate EXP_fields and EXR_fields
    EXP_fields = correctly_predicted_fields / (correctly_predicted_fields + extra_predicted_fields) if correctly_predicted_fields + extra_predicted_fields else 0
    EXR_fields = correctly_predicted_fields / (correctly_predicted_fields + missing_fields) if correctly_predicted_fields + missing_fields else 0

    print("EXP_fields", EXP_fields)
    print("EXR_fields", EXR_fields)
    return EXP_fields, EXR_fields


def calculate_row_metrics(df1, df2, threshold=0.8):
    """
    Calculate row-level metrics (EXP_rows, EXR_rows) based on matching rows in projections of DataFrames.
    """
    # Compute field projections
    similar_columns = compute_column_similarity_with_embeddings(df1, df2, threshold)
    shared_columns_truth = [pair[0] for pair in similar_columns]
    shared_columns_pred = [pair[1] for pair in similar_columns]

    # Projection of DataFrames onto similar columns
    truth_proj = df1[shared_columns_truth]
    pred_proj = df2[shared_columns_pred]

    # Initialize counters for row matches
    correctly_predicted_rows = 0
    extra_predicted_rows = 0
    missing_rows = 0

    # Iterate through rows to match based on the projections
    R_truth_similar = set(tuple(row) for row in truth_proj.itertuples(index=False))
    R_pred_similar = set(tuple(row) for row in pred_proj.itertuples(index=False))

    correctly_predicted_rows = len(R_truth_similar & R_pred_similar)
    extra_predicted_rows = len(R_pred_similar - R_truth_similar)
    missing_rows = len(R_truth_similar - R_pred_similar)

    print("correctly_predicted_rows", correctly_predicted_rows)
    print("extra_predicted_rows", extra_predicted_rows)
    print("missing_rows", missing_rows)

    # Calculate EXP_rows and EXR_rows
    EXP_rows = correctly_predicted_rows / (correctly_predicted_rows + extra_predicted_rows) if correctly_predicted_rows + extra_predicted_rows else 0
    EXR_rows = correctly_predicted_rows / (correctly_predicted_rows + missing_rows) if correctly_predicted_rows + missing_rows else 0

    print("EXP_rows", EXP_rows)
    print("EXR_rows", EXR_rows)
    return EXP_rows, EXR_rows


def calculate_metrics(csv1, csv2, threshold=0.8):
    """
    Calculate evaluation metrics (EXP, EXR) based on similarity between two CSV files.
    - EXP_fields, EXR_fields for fields.
    - EXP_rows, EXR_rows for rows.
    """
    df1 = pd.read_csv(csv1)  # Load ground truth CSV
    df2 = pd.read_csv(csv2)  # Load predicted CSV

    # Calculate field-level metrics
    EXP_fields, EXR_fields = calculate_field_metrics(df1, df2, threshold)

    # Calculate row-level metrics
    EXP_rows, EXR_rows = calculate_row_metrics(df1, df2, threshold)

    return {
        "EXP_fields": round(EXP_fields * 100, 1),
        "EXR_fields": round(EXR_fields * 100, 1),
        "EXP_rows": round(EXP_rows * 100, 1),
        "EXR_rows": round(EXR_rows * 100, 1),
    }



def process_all_rules(base_folder, threshold=0.8):
    """
    Process all rule folders under the base folder:
    - Reads JSON and query subfolders.
    - Computes metrics for each query.
    - Saves results as a new JSON file with '_metrics' appended to the original name.
    """
    for rule_folder in sorted(os.listdir(base_folder)):
        rule_path = os.path.join(base_folder, rule_folder)

        if not os.path.isdir(rule_path):
            continue  # Skip files, only process directories.

        # Find the JSON file in the rule folder
        json_files = [f for f in os.listdir(rule_path) if f.endswith(".json") and not f.endswith("metrics.json")]
        if len(json_files) != 1:
            print(f"Skipping {rule_folder}: Could not identify a single JSON file.")
            continue

        rule_json_path = os.path.join(rule_path, json_files[0])

        # Load the rule JSON file
        with open(rule_json_path, "r") as f:
            rule_data = json.load(f)

        results = []
        total_exp_fields = 0
        total_exr_fields = 0
        total_exp_rows = 0
        total_exr_rows = 0
        query_count = 0

        # Process each query subfolder
        for query_folder in sorted(os.listdir(rule_path)):
            query_path = os.path.join(rule_path, query_folder)

            # Skip non-query folders
            if not os.path.isdir(query_path) or not query_folder.startswith("query_"):
                continue

            query_index = int(query_folder.replace("query_", ""))
            csv1 = os.path.join(query_path, "ground_truth.csv")
            csv2 = os.path.join(query_path, "predicted.csv")

            # Check if both required CSV files exist
            if os.path.exists(csv1) and os.path.exists(csv2):
                # Calculate metrics for the query
                metrics = calculate_metrics(csv1, csv2, threshold)

                # Aggregate metrics for averaging later
                total_exp_fields += metrics.get("EXP_fields", 0)
                total_exr_fields += metrics.get("EXR_fields", 0)
                total_exp_rows += metrics.get("EXP_rows", 0)
                total_exr_rows += metrics.get("EXR_rows", 0)
                query_count += 1

                # Store query-level results
                result = {
                    "query_id": query_index,
                    "question": rule_data[query_index]["question"],
                    "db_id": rule_data[query_index]["db_id"],
                    "ground_truth_query": rule_data[query_index]["ground_truth_query"],
                    "predicted_query": rule_data[query_index]["predicted_query"],
                    "EX_value": rule_data[query_index]["EX_value"],
                    "metrics": metrics
                }
                results.append(result)

        # Sort results by query ID
        results.sort(key=lambda x: x["query_id"])

        # Compute average metrics across all queries
        average_exp_fields = total_exp_fields / query_count if query_count > 0 else 0
        average_exr_fields = total_exr_fields / query_count if query_count > 0 else 0
        average_exp_rows = total_exp_rows / query_count if query_count > 0 else 0
        average_exr_rows = total_exr_rows / query_count if query_count > 0 else 0

        # Prepare output JSON data
        output_data = {
            "results": results,
            "average_EXP_fields": round(average_exp_fields, 1),
            "average_EXR_fields": round(average_exr_fields, 1),
            "average_EXP_rows": round(average_exp_rows, 1),
            "average_EXR_rows": round(average_exr_rows, 1),
        }

        # Save results to a new JSON file with '_metrics' suffix
        output_json_path = os.path.join(
            rule_path, f"{os.path.splitext(json_files[0])[0]}_metrics.json"
        )
        with open(output_json_path, "w") as out_file:
            json.dump(output_data, out_file, indent=4)

        print(f"Processed {rule_folder} - Results saved to {output_json_path}")


# Example usage
if __name__ == "__main__":
    base_folder = "rules_predictions"  # Base folder containing rule subfolders
    process_all_rules(base_folder, threshold=0.8)

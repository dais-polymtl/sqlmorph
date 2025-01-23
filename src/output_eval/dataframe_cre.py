import os
import sqlite3
import pandas as pd
import json

# Base paths
input_base_path = "rules_predictions"
base_db_path = "dev_databases"

# Traverse through all subfolders under `rules_predictions`
for folder_name in os.listdir(input_base_path):
    folder_path = os.path.join(input_base_path, folder_name)
    input_file = os.path.join(folder_path, f"{folder_name}.json")
    
    # Check if folder exists and contains the JSON file
    if os.path.isdir(folder_path) and os.path.isfile(input_file):
        print(f"Processing folder: {folder_name}")

        # Read the JSON file
        with open(input_file, "r") as file:
            data = json.load(file)

        # Process each query in the JSON file
        for idx, entry in enumerate(data):
            db_id = entry["db_id"]
            ground_truth_query = entry["ground_truth_query"]
            predicted_query = entry["predicted_query"]

            # Generate database path
            db_path = os.path.join(base_db_path, db_id, f"{db_id}.sqlite")
            
            # Ensure the database file exists
            if not os.path.isfile(db_path):
                print(f"Database file not found: {db_path}")
                continue

            # Create output folder for this query within the current folder
            query_folder = os.path.join(folder_path, f"query_{idx}")
            os.makedirs(query_folder, exist_ok=True)

            # Function to execute query and save to CSV
            def execute_and_save_query(query, filename):
                if not query.strip():
                    print(f"No query provided for query_{idx}, skipping...")
                    return
                
                try:
                    with sqlite3.connect(db_path) as conn:
                        df = pd.read_sql_query(query, conn)
                        output_file = os.path.join(query_folder, filename)
                        df.to_csv(output_file, index=False)
                        print(f"Saved results to {output_file}")
                except Exception as e:
                    print(f"Error executing query in query_{idx}: {e}")

            # Execute and save ground truth query
            execute_and_save_query(ground_truth_query, "ground_truth.csv")

            # Execute and save predicted query
            execute_and_save_query(predicted_query, "predicted.csv")

print("Processing complete.")

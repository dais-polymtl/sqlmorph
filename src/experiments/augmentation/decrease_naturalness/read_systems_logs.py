import json
import os

import pandas as pd


def process_predicted_sql(predicted_sql: str) -> str:
    """Remove newlines and extra spaces from predicted SQL."""
    return " ".join(predicted_sql.split())


def read_chess_data(chess_dir_path: str) -> pd.DataFrame:
    """Read data from chess system directory containing JSON files."""
    data_rows = []

    if not os.path.exists(chess_dir_path):
        print(f"Warning: Directory {chess_dir_path} does not exist")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )

    for filename in os.listdir(chess_dir_path):
        if filename.endswith(".json"):
            try:
                # Extract question_id (first number) and db_name from filename
                first_underscore = filename.find("_")
                if first_underscore != -1:
                    question_id = int(filename[:first_underscore])
                    db_name = filename[first_underscore + 1 : filename.rfind(".json")]
                else:
                    question_id = 0  # Default if no underscore
                    db_name = filename[:-5]  # Remove .json

                # Read JSON file
                file_path = os.path.join(chess_dir_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Search through all items in the list to find final_SQL
                if isinstance(data, list):
                    final_sql_found = False
                    for item in data:
                        if "final_SQL" in item:
                            final_sql = item["final_SQL"]

                            row = {
                                "system": "chess",
                                "db_name": db_name,
                                "question_id": question_id,
                                "question": final_sql.get("Question", ""),
                                "gold_sql": final_sql.get("GOLD_SQL", ""),
                                "predicted_sql": process_predicted_sql(
                                    final_sql.get("PREDICTED_SQL", "")
                                ),
                            }
                            data_rows.append(row)
                            final_sql_found = True
                            break

                    if not final_sql_found:
                        print(f"Warning: No 'final_SQL' key found in {filename}")
                else:
                    print(f"Warning: Invalid data structure in {filename}")

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

    # Create DataFrame and sort by question_id
    df = pd.DataFrame(data_rows)
    if not df.empty:
        df = df.sort_values("question_id").reset_index(drop=True)

    return df


def read_mac_sql_data(mac_sql_dir_path: str) -> pd.DataFrame:
    """Read data from mac-sql system directory."""
    data_rows = []

    if not os.path.exists(mac_sql_dir_path):
        print(f"Warning: Directory {mac_sql_dir_path} does not exist")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )

    json_path = os.path.join(mac_sql_dir_path, "output_bird.json")

    if not os.path.exists(json_path):
        print(f"Warning: output_bird.json not found in {mac_sql_dir_path}")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )

    try:
        # Read JSON file line by line (JSONL format)
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            data_row = {
                                "system": "mac-sql",
                                "db_name": item.get("db_id", ""),
                                "question_id": item.get("idx", 0),
                                "question": item.get("query", ""),
                                "gold_sql": item.get("ground_truth", ""),
                                "predicted_sql": process_predicted_sql(
                                    item.get("final_sql", "")
                                ),
                            }
                            data_rows.append(data_row)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in mac-sql JSON: {str(e)}")
                        continue

        # Create DataFrame and sort by question_id
        df = pd.DataFrame(data_rows)
        if not df.empty:
            df = df.sort_values("question_id").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error processing mac-sql data: {str(e)}")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )


def read_din_sql_data(din_sql_dir_path: str) -> pd.DataFrame:
    """Read data from din-sql system directory."""
    data_rows = []

    if not os.path.exists(din_sql_dir_path):
        print(f"Warning: Directory {din_sql_dir_path} does not exist")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )

    csv_path = os.path.join(din_sql_dir_path, "logs.csv")

    if not os.path.exists(csv_path):
        print(f"Warning: logs.csv not found in {din_sql_dir_path}")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )

    try:
        # Read CSV file
        csv_data = pd.read_csv(csv_path)

        # Process data from CSV
        for index, row in csv_data.iterrows():
            data_row = {
                "system": "din-sql",
                "db_name": row["db_id"],
                "question_id": index,  # Use index as question_id
                "question": row["question"],
                "gold_sql": row["gold_query"],
                "predicted_sql": process_predicted_sql(row["final_query"]),
            }
            data_rows.append(data_row)

        # Create DataFrame and sort by question_id
        df = pd.DataFrame(data_rows)
        if not df.empty:
            df = df.sort_values("question_id").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error processing din-sql data: {str(e)}")
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )


def combine_all_data(
    chess_dir: str, mac_sql_dir: str = None, din_sql_dir: str = None
) -> pd.DataFrame:
    """Combine data from all three systems into a single DataFrame."""
    all_dfs = []

    if chess_dir:
        chess_df = read_chess_data(chess_dir)
        if not chess_df.empty:
            all_dfs.append(chess_df)

    if mac_sql_dir:
        mac_sql_df = read_mac_sql_data(mac_sql_dir)
        if not mac_sql_df.empty:
            all_dfs.append(mac_sql_df)

    if din_sql_dir:
        din_sql_df = read_din_sql_data(din_sql_dir)
        if not din_sql_df.empty:
            all_dfs.append(din_sql_df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Sort by system first, then by question_id
        return combined_df.sort_values(["system", "question_id"]).reset_index(drop=True)
    else:
        return pd.DataFrame(
            columns=[
                "system",
                "db_name",
                "question_id",
                "question",
                "gold_sql",
                "predicted_sql",
            ]
        )


if __name__ == "__main__":
    ROOT_DIR = "data/augmentation/snail/experiments/"
    data_mode = "new_nl_original_sql_results"
    chess_directory = ROOT_DIR + "chess/" + data_mode
    din_sql_directory = ROOT_DIR + "din-sql/" + data_mode
    mac_sql_directory = ROOT_DIR + "mac-sql/" + data_mode

    # Test individual functions
    chess_data = read_chess_data(chess_directory)
    mac_sql_data = read_mac_sql_data(mac_sql_directory)
    din_sql_data = read_din_sql_data(din_sql_directory)
    # # combine mac and din data
    # combined_mac_din = pd.concat([mac_sql_data, din_sql_data], ignore_index=True).sort_values(["system", "question_id"]).reset_index(drop=True)
    # combined_mac_din.to_csv(ROOT_DIR + f"mac_din_sql_data_{data_mode}.csv", index=False)

    print(f"Chess data shape: {chess_data.shape}")
    print(f"Mac-SQL data shape: {mac_sql_data.shape}")
    print(f"Din-SQL data shape: {din_sql_data.shape}")

    # # Combine all data
    all_data = combine_all_data(chess_directory, mac_sql_directory, din_sql_directory)
    all_data.to_csv(ROOT_DIR + f"systems_data_{data_mode}.csv", index=False)
    print("DONE!")

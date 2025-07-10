import os
import json
import pickle

from src.core.logger.logger import Logger


logger = Logger(name="__name__")


def save_query_first(queries, output_file, status, mode):
    rule_outputs_base = os.getenv("RULE_OUTPUTS_BASE")
    data_folder = os.getenv("DATA_FOLDER")
    benchmark_path = os.path.join(data_folder, "benchmarks", "Bird")
    with open(os.path.join(benchmark_path, "bird_dev.json"), "r") as f:
        benchmark_data = json.load(f)

    benchmark_dict = {entry["question_id"]: entry for entry in benchmark_data}

    original_json_data = []
    extended_json_data = []
    sql_queries = []

    for query in queries["queries"]:
        match = benchmark_dict.get(query["ext_id"])
        if match:
            original_json_data.append(match)
        extended_json_data.append(
            {
                "question_id": query["id"],
                "db_id": queries["db_id"],
                "question": query.get("question", ""),
                "evidence": query["evidence"],
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        sql_queries.append(query["SQL"] + "\t" + queries["db_id"])

    output_folder = os.path.join(rule_outputs_base, status, mode)

    os.makedirs(output_folder, exist_ok=True)

    json_output_file_original = os.path.join(output_folder, f"{output_file}_ori.json")
    json_output_file_extended = os.path.join(output_folder, f"{output_file}_aug.json")
    sql_output_file = os.path.join(output_folder, f"{output_file}.sql")

    with open(json_output_file_original, "w") as f_json:
        json.dump(original_json_data, f_json, indent=4)
    with open(json_output_file_extended, "w") as f_json:
        json.dump(extended_json_data, f_json, indent=4)
    with open(sql_output_file, "w") as f_sql:
        f_sql.write("\n".join(sql_queries))


def save_graph_first(queries, output_file, status, mode):
    rule_outputs_base = os.getenv("RULE_OUTPUTS_BASE")
    explicit_json_data = []
    dev_set_like_json_data = []
    sql_queries = []

    for query in queries["graph_first"]:
        explicit_json_data.append(
            {
                "question_id": query["id"],
                "db_id": queries["db_id"],
                "question": query.get("explicit_question", ""),
                "evidence": "",
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        dev_set_like_json_data.append(
            {
                "question_id": query["id"],
                "db_id": queries["db_id"],
                "question": query.get("question", ""),
                "evidence": query.get("evidence", ""),
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        sql_queries.append(query["SQL"] + "\t" + queries["db_id"])

    # Include status in the path
    output_folder = os.path.join(rule_outputs_base, status, mode)

    os.makedirs(output_folder, exist_ok=True)

    json_output_file_explicit = os.path.join(output_folder, f"{output_file}_exp.json")
    json_output_file_dev = os.path.join(output_folder, f"{output_file}_dev.json")
    sql_output_file = os.path.join(output_folder, f"{output_file}.sql")

    with open(json_output_file_explicit, "w") as f_json:
        json.dump(explicit_json_data, f_json, indent=4)
    with open(json_output_file_dev, "w") as f_json:
        json.dump(dev_set_like_json_data, f_json, indent=4)
    with open(sql_output_file, "w") as f_sql:
        f_sql.write("\n".join(sql_queries))


def update_augmentation_log(db_id, i, logged_extensions):
    """
    Loads previous logged extensions from augmentation_log.pickle,
    appends the new ones, and writes back the updated list.
    """
    rule_outputs_base = os.getenv("RULE_OUTPUTS_BASE")
    log_dir = os.path.join(rule_outputs_base, "aug_log")
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "augmentation_log.pickle")

    # Step 1: Load existing log if it exists
    if os.path.exists(log_file_path):
        with open(log_file_path, "rb") as f:
            try:
                existing_log = pickle.load(f)
            except Exception as e:
                print(f"[WARNING] Could not load existing log: {e}")
                existing_log = []
    else:
        existing_log = []

    # Step 2: Extend the log
    existing_log.extend(logged_extensions)

    # Step 3: Save updated log
    with open(log_file_path, "wb") as f:
        pickle.dump(existing_log, f)
        logger.log(
            level="info",
            action="update_augmentation_log",
            details={
                "db_id": db_id,
                "num_tables": i + 1,
                "extensions_count": len(logged_extensions),
                "log_file": log_file_path,
            },
        )

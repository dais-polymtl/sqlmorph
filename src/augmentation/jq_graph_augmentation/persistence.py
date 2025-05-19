import os
import json


def save_query_first(queries, output_file):
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
                "question": query["question"] or "",
                "evidence": query["evidence"],
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        sql_queries.append(query["SQL"] + "\t" + queries["db_id"])

    json_output_file_original = os.path.join(
        rule_outputs_base, output_file + "_original.json"
    )
    json_output_file_extended = os.path.join(
        rule_outputs_base, output_file + "_extended.json"
    )

    sql_output_file = os.path.join(rule_outputs_base, output_file + ".sql")

    os.makedirs(os.path.dirname(json_output_file_original), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_file_extended), exist_ok=True)

    os.makedirs(os.path.dirname(sql_output_file), exist_ok=True)

    with open(json_output_file_original, "w") as f_json:
        json.dump(original_json_data, f_json, indent=4)
    with open(json_output_file_extended, "w") as f_json:
        json.dump(extended_json_data, f_json, indent=4)
    with open(sql_output_file, "w") as f_sql:
        f_sql.write("\n".join(sql_queries))


def save_graph_first(
    queries,
    output_file,
):
    rule_outputs_base = os.getenv("RULE_OUTPUTS_BASE")
    explicit_json_data = []
    dev_set_like_json_data = []
    sql_queries = []

    for query in queries["graph_first"]:
        explicit_json_data.append(
            {
                "question_id": query["id"],
                "db_id": queries["db_id"],
                "question": query["explicit_question"] or "",
                "evidence": "",
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        dev_set_like_json_data.append(
            {
                "question_id": query["id"],
                "db_id": queries["db_id"],
                "question": query["question"] or "",
                "evidence": query["evidence"] or "",
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        sql_queries.append(query["SQL"] + "\t" + queries["db_id"])

    json_output_file_explicit = os.path.join(
        rule_outputs_base, output_file + "_explicit.json"
    )
    json_output_file_dev = os.path.join(
        rule_outputs_base, output_file + "_dev_set_like.json"
    )
    sql_output_file = os.path.join(rule_outputs_base, output_file + ".sql")

    os.makedirs(os.path.dirname(json_output_file_explicit), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_file_dev), exist_ok=True)
    os.makedirs(os.path.dirname(sql_output_file), exist_ok=True)
    with open(json_output_file_explicit, "w") as f_json:
        json.dump(explicit_json_data, f_json, indent=4)
    with open(json_output_file_dev, "w") as f_json:
        json.dump(dev_set_like_json_data, f_json, indent=4)
    with open(sql_output_file, "w") as f_sql:
        f_sql.write("\n".join(sql_queries))

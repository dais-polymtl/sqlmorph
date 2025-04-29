import os
import json


def save_query_first(queries, output_file):
    rule_outputs_base = os.getenv("RULE_OUTPUTS_BASE")
    original_json_data = []
    extended_json_data = []
    sql_queries = []

    for i, query in enumerate(queries):
        original_json_data.append(
            {
                "question_id": i,
                "db_id": query["db_id"],
                "question": query["question"],
                "evidence": query["evidence"],
                "SQL": query["SQL"],
                "difficulty": query["difficulty"],
            }
        )
        extended_json_data.append(
            {
                "question_id": i,
                "db_id": query["db_id"],
                "question": query["query_first"]["new_question"] or "",
                "evidence": query["evidence"],
                "SQL": query["query_first"]["new_query"],
                "difficulty": query["difficulty"],
            }
        )
        sql_queries.append(query["query_first"]["new_query"] + "\t" + query["db_id"])

    json_output_file = os.path.join(rule_outputs_base, output_file + ".json")
    sql_output_file = os.path.join(rule_outputs_base, output_file + ".sql")

    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(sql_output_file), exist_ok=True)

    with open(json_output_file, "w") as f_json:
        json.dump(original_json_data, f_json, indent=4)
    with open(json_output_file, "w") as f_json:
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

    for i, query in enumerate(queries):
        explicit_json_data.append(
            {
                "question_id": i,
                "db_id": query["db_id"],
                "question": query["graph_first"]["explicit_question"] or "",
                "evidence": "",
                "SQL": query["graph_first"]["main_query"],
                "difficulty": query["difficulty"],
            }
        )
        dev_set_like_json_data.append(
            {
                "question_id": i,
                "db_id": query["db_id"],
                "question": query["graph_first"]["dev_set_like_question"] or "",
                "evidence": query["graph_first"]["new_evidence"] or "",
                "SQL": query["graph_first"]["main_query"],
                "difficulty": query["difficulty"],
            }
        )
        sql_queries.append(query["graph_first"]["main_query"] + "\t" + query["db_id"])

    json_output_file = os.path.join(rule_outputs_base, output_file + ".json")
    sql_output_file = os.path.join(rule_outputs_base, output_file + ".sql")

    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(sql_output_file), exist_ok=True)
    with open(json_output_file, "w") as f_json:
        json.dump(explicit_json_data, f_json, indent=4)
    with open(json_output_file, "w") as f_json:
        json.dump(dev_set_like_json_data, f_json, indent=4)
    with open(sql_output_file, "w") as f_sql:
        f_sql.write("\n".join(sql_queries))

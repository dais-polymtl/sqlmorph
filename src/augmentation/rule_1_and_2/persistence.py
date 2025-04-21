import os
import json
from pathlib import Path

import networkx as nx

from .llm_inference import (
    generate_explicit_question,
    generate_dev_set_like_question,
    generate_evidence,
    generate_new_extended_question,
)
from .graph_processing import calculate_subgraph_centrality, is_cyclic
from .data_retrieval import load_schema
from src.data_prep.database_schemas.bird_schema import BIRD_Schema


def compute_graph_metrics(subgraph, schema_graph):
    """Compute graph-based metrics."""
    centrality = nx.degree_centrality(schema_graph)
    return {
        "num_nodes": subgraph.number_of_nodes(),
        "num_connections": subgraph.number_of_edges(),
        "cyclic": is_cyclic(subgraph),
        "centrality_score": calculate_subgraph_centrality(subgraph, centrality),
    }


def get_schema_object(db_id):
    """Load BIRD_Schema object."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    schema_data_file = project_root / "data" / "benchmarks" / "Bird" / "dev_tables.json"
    return BIRD_Schema(db_id, schema_data_file)


def write_outputs(json_output_file, sql_output_file, json_data, sql_queries):
    """Save JSON and SQL output files."""
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(sql_output_file), exist_ok=True)

    with open(json_output_file, "w") as f_json:
        json.dump(json_data, f_json, indent=4)

    with open(sql_output_file, "w") as f_sql:
        f_sql.write("\n".join(sql_queries))


def save_new_queries(
    rules_list,
    graph_data_folder,
    json_output_file,
    sql_output_file,
    question_type,
    extension_type
):
    global_rule_data = []

    for rule_data in rules_list:
        db_id = rule_data["db_id"]
        schema_graph = load_schema(
            os.path.join(graph_data_folder, f"{db_id}_graph.pkl")
        )
        metrics = compute_graph_metrics(rule_data["extended_subgraph"], schema_graph)

        global_rule_data.append(
            {
                "db_id": db_id,
                "main_query": rule_data["graph_first"]["main_query"],
                **metrics,
            }
        )

    global_rule_data_sorted = sorted(
        global_rule_data,
        key=lambda x: (
            x["num_nodes"],
            x["num_connections"],
            x["cyclic"],
            x["centrality_score"],
        ),
        reverse=True,
    )

    json_data, sql_queries = [], []

    for idx, entry in enumerate(global_rule_data_sorted):
        schema = get_schema_object(entry["db_id"])

        if extension_type == "excluded":
            question, evidence = "", ""

        elif extension_type == "pruned":
            if question_type == "explicit":
                question = generate_explicit_question(entry["main_query"], schema=schema)
                evidence = ""
            elif question_type == "dev_set_like":
                question = generate_dev_set_like_question(entry["main_query"], schema=schema)
                evidence = generate_evidence(entry["main_query"], schema=schema, question=question)

        json_data.append(
            {
                "question_id": idx,
                "db_id": entry["db_id"],
                "question": question,
                "evidence": evidence,
                "SQL": entry["main_query"],
                "difficulty": "challenging",
            }
        )
        sql_queries.append(f"{entry['main_query']}\t{entry['db_id']}")

    write_outputs(json_output_file, sql_output_file, json_data, sql_queries)


def save_old_extended_queries(
    rule_list,
    graph_data_folder,
    json_output_file,
    sql_output_file,
    query_type,
    extension_type
):
    global_rule_data = []

    for rule_data in rule_list:
        db_id = rule_data["db_id"]
        schema_graph = load_schema(
            os.path.join(graph_data_folder, f"{db_id}_graph.pkl")
        )
        metrics = compute_graph_metrics(rule_data["extended_subgraph"], schema_graph)

        for query in rule_data["query_first"]:
            global_rule_data.append(
                {"db_id": db_id, "equivalent_query": query, **metrics}
            )

    global_rule_data_sorted = sorted(
        global_rule_data,
        key=lambda x: (
            x["num_nodes"],
            x["num_connections"],
            x["cyclic"],
            x["centrality_score"],
        ),
        reverse=True,
    )

    json_data, sql_queries = [], []

    for idx, entry in enumerate(global_rule_data_sorted):
        query = entry["equivalent_query"]
        schema = get_schema_object(query["db_id"])

        if extension_type == "excluded":
            if query_type == "original":
                question = query["question"]
                sql_query = query["SQL"]
            elif query_type == "new":
                question = ""
                sql_query = query["new_query"]

        elif extension_type == "pruned":
            if query_type == "original":
                question = query["question"]
                sql_query = query["SQL"]
            elif query_type == "new":
                question = generate_new_extended_question(
                query["SQL"], query["new_query"], schema, query["question"]
            )
            sql_query = query["new_query"]

        json_data.append(
            {
                "question_id": idx,
                "db_id": query["db_id"],
                "question": question,
                "evidence": query.get("evidence", ""),
                "SQL": sql_query,
                "difficulty": query.get("difficulty", "challenging"),
            }
        )
        sql_queries.append(f"{sql_query}\t{query['db_id']}")

    write_outputs(json_output_file, sql_output_file, json_data, sql_queries)

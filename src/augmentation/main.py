import os
import pandas as pd

from rule_1_and_2 import (
    retrieve_all_dev_patterns,
    load_schema,
    extend_and_filter_subgraphs,
    ensure_directory,
)
from rule_1_and_2.persistence import save_rule_data, process_rule_folder
from rule_1_and_2.query_execution import process_queries_for_rules, filter_valid_queries
from rule_1_and_2.query_generation import generate_inner_join_query_with_test
from rule_1_and_2.visualization import save_pre_rule_subgraphs


def run_rule_1_and_2():
    # Configuration
    data_folder = "data/"

    db_ids = [
        "california_schools",
        "card_games",
        "codebase_community",
        "debit_card_specializing",
        "european_football_2",
        "financial",
        "formula_1",
        "student_club",
        "superhero",
        "thrombosis_prediction",
        "toxicology",
    ]
    rule_inputs_base = os.path.join(data_folder, "rule_inputs", "essay_bird_subgraphs")
    graph_data_base = os.path.join(data_folder, "graph_data", "bird_graphs", "pickles")
    rule_outputs_base = os.path.join(data_folder, "rule_outputs", "rules_1_2")
    images_output_base = os.path.join(rule_outputs_base, "images")
    ensure_directory(images_output_base)

    # Load query statistics once as it's common for all databases
    query_stats_path = os.path.join(rule_inputs_base, "query_statistics.csv")
    if not os.path.exists(query_stats_path):
        print(
            f"Query statistics file not found at {query_stats_path}. Please check the path."
        )
        return
    df = pd.read_csv(query_stats_path)

    for db_id in db_ids:
        print(f"\nProcessing Database: {db_id}")

        # Define paths specific to the current db_id
        rule_inputs_folder = os.path.join(rule_inputs_base, db_id)
        graph_data_folder = graph_data_base
        rule_outputs_folder = os.path.join(rule_outputs_base)
        images_output_dir = os.path.join(images_output_base, db_id)
        ensure_directory(images_output_dir)

        # Retrieve all subgraphs
        if not os.path.exists(rule_inputs_folder):
            print(
                f"Rule inputs folder not found at {rule_inputs_folder}. Skipping {db_id}."
            )
            continue

        database_subgraphs = retrieve_all_dev_patterns(rule_inputs_folder)
        print(f"Found {len(database_subgraphs)} subgraphs for {db_id}.")

        # Load the schema
        schema_path = os.path.join(graph_data_folder, f"{db_id}_graph.pkl")
        if not os.path.exists(schema_path):
            print(f"Schema file not found at {schema_path}. Skipping {db_id}.")
            continue
        schema = load_schema(schema_path)

        # Save pre-rule subgraphs visualization
        save_pre_rule_subgraphs(
            schema=schema,
            subgraphs=database_subgraphs,
            rule_name="rule_1",
            db_id=db_id,
            output_base_dir=images_output_dir,
        )
        save_pre_rule_subgraphs(
            schema=schema,
            subgraphs=database_subgraphs,
            rule_name="rule_2",
            db_id=db_id,
            output_base_dir=images_output_dir,
        )

        # Extend and filter subgraphs based on rules
        rule_1_patterns, rule_2_patterns = extend_and_filter_subgraphs(
            database_subgraphs, schema
        )
        print(f"Rule 1 Patterns for {db_id}: {len(rule_1_patterns)}")
        print(f"Rule 2 Patterns for {db_id}: {len(rule_2_patterns)}")

        # Generate queries for Rule 1 and Rule 2
        rule_1_data = []
        rule_2_data = []

        for pattern in rule_1_patterns:
            result = generate_inner_join_query_with_test(schema, pattern, db_id, df)
            if db_id == "financial":
                if "order" in result["main_query"]:
                    result["main_query"] = result["main_query"].replace(
                        "order", "'order'"
                    )
                    result["test_query"] = result["test_query"].replace(
                        "order", "'order'"
                    )

            rule_1_data.append(
                {
                    "subgraph": pattern,
                    "main_query": result["main_query"],
                    "test_query": result["test_query"],
                    "projection_columns": result["projection_columns"],
                    "filtering_columns": result["filtering_columns"],
                }
            )

        for pattern in rule_2_patterns:
            result = generate_inner_join_query_with_test(schema, pattern, db_id, df)
            if db_id == "financial":
                if "order" in result["main_query"]:
                    result["main_query"] = result["main_query"].replace(
                        "order", "'order'"
                    )
                    result["test_query"] = result["test_query"].replace(
                        "order", "'order'"
                    )

            rule_2_data.append(
                {
                    "subgraph": pattern,
                    "main_query": result["main_query"],
                    "test_query": result["test_query"],
                    "projection_columns": result["projection_columns"],
                    "filtering_columns": result["filtering_columns"],
                }
            )

        database_path_template = os.path.join(
            data_folder, "benchmarks", "Bird", "dev_databases", db_id, f"{db_id}.sqlite"
        )

        # Execute and update queries
        if not os.path.exists(database_path_template):
            print(
                f"Database file not found at {database_path_template}. Skipping execution for {db_id}."
            )
            continue

        process_queries_for_rules(rule_1_data, db_id, database_path_template)
        process_queries_for_rules(rule_2_data, db_id, database_path_template)

        # rule_1_data = filter_valid_queries(rule_1_data, database_path_template)
        # rule_2_data = filter_valid_queries(rule_2_data, database_path_template)

        # Save outputs
        save_rule_data(rule_1_data, "rule_1", rule_outputs_folder, db_id)
        save_rule_data(rule_2_data, "rule_2", rule_outputs_folder, db_id)
        print(f"Data saved successfully for {db_id}.")

    # Global ordering and final output generation for all databases
    for rule_name in ["rule_1", "rule_2"]:
        for question_type in ["explicit", "dev_set_like"]:
            print(
                f"\nProcessing Global Outputs for {rule_name} with {question_type} questions..."
            )
            json_output = os.path.join(
                rule_outputs_base, f"dev_{rule_name.split('_')[1]}_{question_type}.json"
            )
            sql_output = os.path.join(
                rule_outputs_base, f"dev_{rule_name.split('_')[1]}_{question_type}.sql"
            )
            process_rule_folder(
                rule_outputs_folder,
                graph_data_base,
                rule_name,
                json_output,
                sql_output,
                question_type,
            )
            print(
                f"Files {os.path.basename(json_output)} and {os.path.basename(sql_output)} generated successfully for {rule_name}."
            )

    print("\nAll databases processed successfully.")


if __name__ == "__main__":
    run_rule_1_and_2()

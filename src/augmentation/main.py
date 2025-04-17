import csv
import json
import os

import pandas as pd

from rule_1_and_2 import (
    retrieve_all_dev_patterns,
    load_schema,
    extend_and_filter_subgraphs,
    ensure_directory,
)
from rule_1_and_2.persistence import (
    save_rule_data,
    process_rule_folder,
    process_rule_folder_2nd_version,
)
from rule_1_and_2.query_execution import (
    process_queries_for_rules,
    filter_valid_queries,
    filter_valid_queries_2nd_version,
)
from rule_1_and_2.query_generation import (
    generate_inner_join_query_with_test,
    modify_rule_1_queries,
)
from rule_1_and_2.visualization import save_pre_rule_subgraphs
from rule_3 import (
    retrieve_all_dev_patterns as r3_retrieve_all_dev_patterns,
    split_queries,
    filter_subgraphs,
    process_dataset,
    hide_tables_with_synonym_replacement,
    hide_tables_with_backtranslation,
    hide_tables_with_contextual_augmentation,
    process_and_save_data_for_set,
)
from rule_3.utils import set_random_seed


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
    # db_ids = [
    #     "california_schools",
    #     "card_games"
    # ]

    rule_inputs_base = os.path.join(data_folder, "rule_inputs", "rules_1_2")
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

    augmentation_numbers = []
    rule_1_error_queries_for_databases = []
    rule_2_error_queries_for_databases = []
    rule_1_empty_queries_for_databases = []
    rule_2_empty_queries_for_databases = []

    rule_outputs_folder = os.path.join(rule_outputs_base)
    rule_1_error_queries_path = os.path.join(
        rule_outputs_folder, "rule_1_error_queries.json"
    )
    rule_2_error_queries_path = os.path.join(
        rule_outputs_folder, "rule_2_error_queries.json"
    )
    rule_1_empty_queries_path = os.path.join(
        rule_outputs_folder, "rule_1_empty_queries.json"
    )
    rule_2_empty_queries_path = os.path.join(
        rule_outputs_folder, "rule_2_empty_queries.json"
    )

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

        (
            rule_1_patterns,
            rule_2_patterns,
            r1_before,
            r2_before,
            r1_after,
            r2_after,
            r1_dev_before,
            r2_dev_before,
            r1_dev_after,
            r2_dev_after,
        ) = extend_and_filter_subgraphs(database_subgraphs, schema)
        print(f"Rule 1 Patterns for {db_id}: {len(rule_1_patterns)}")
        print(f"Rule 2 Patterns for {db_id}: {len(rule_2_patterns)}")

        augmentation_stat = {
            "db_id": db_id,
            "rule_1_without_pruning": r1_before,
            "rule_2_without_pruning": r2_before,
            "rule_1_with_pruning": r1_after,
            "rule_2_with_pruning": r2_after,
            "rule_1_dev_before": r1_dev_before,
            "rule_2_dev_before": r2_dev_before,
            "rule_1_dev_after": r1_dev_after,
            "rule_2_dev_after": r2_dev_after,
        }
        augmentation_numbers.append(augmentation_stat)

        # Generate queries for Rule 1 and Rule 2
        rule_1_data = []
        rule_2_data = []
        rule_1_data_2nd_version = []
        rule_2_data_2nd_version = []

        for pattern in rule_1_patterns:
            # result = generate_inner_join_query_with_test(schema, pattern['extended_subgraph'], db_id, df)
            result = generate_inner_join_query_with_test(
                schema, pattern["extended_subgraph"], db_id, df
            )
            result_2nd_version = modify_rule_1_queries(pattern)

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
            rule_1_data_2nd_version.append(result_2nd_version)

        for pattern in rule_2_patterns:
            # result = generate_inner_join_query_with_test(schema, pattern['extended_subgraph'], db_id, df)
            result = generate_inner_join_query_with_test(
                schema, pattern["extended_subgraph"], db_id, df
            )
            result_2nd_version = modify_rule_1_queries(pattern)
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
            rule_2_data_2nd_version.append(result_2nd_version)

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

        rule_1_data = filter_valid_queries(rule_1_data, database_path_template)
        rule_2_data = filter_valid_queries(rule_2_data, database_path_template)

        rule_1_data_2nd_version, rule_1_error_queries, rule_1_empty_queries = (
            filter_valid_queries_2nd_version(
                rule_1_data_2nd_version, database_path_template
            )
        )
        rule_2_data_2nd_version, rule_2_error_queries, rule_2_empty_queries = (
            filter_valid_queries_2nd_version(
                rule_2_data_2nd_version, database_path_template
            )
        )

        rule_1_error_queries_for_databases.extend(rule_1_error_queries)
        rule_2_error_queries_for_databases.extend(rule_2_error_queries)
        rule_1_empty_queries_for_databases.extend(rule_1_empty_queries)
        rule_2_empty_queries_for_databases.extend(rule_2_empty_queries)

        # Save outputs
        save_rule_data(rule_1_data, "rule_1", rule_outputs_folder, db_id)
        save_rule_data(rule_2_data, "rule_2", rule_outputs_folder, db_id)

        save_rule_data(
            rule_1_data_2nd_version, "rule_1_2nd_version", rule_outputs_folder, db_id
        )
        save_rule_data(
            rule_2_data_2nd_version, "rule_2_2nd_version", rule_outputs_folder, db_id
        )
        print(f"Data saved successfully for {db_id}.")

    # Save augmentation statistics
    augmentation_stats_output = os.path.join(
        rule_outputs_base, "augmentation_stats.csv"
    )
    fieldnames = [
        "db_id",
        "rule_1_without_pruning",
        "rule_2_without_pruning",
        "rule_1_with_pruning",
        "rule_2_with_pruning",
    ]
    fieldnames += [
        "rule_1_dev_before",
        "rule_2_dev_before",
        "rule_1_dev_after",
        "rule_2_dev_after",
    ]
    with open(augmentation_stats_output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(augmentation_numbers)
    print(f"Augmentation statistics saved to {augmentation_stats_output}.")

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

    print("\nAll databases processed successfully for the first version.")

    for rule_name in ["rule_1_2nd_version", "rule_2_2nd_version"]:
        for query_type in ["original", "new"]:
            print(
                f"\nProcessing Global Outputs for {rule_name} with {question_type} questions..."
            )
            first_part = rule_name.replace("rule_", "")
            json_output = os.path.join(
                rule_outputs_base, f"dev_{first_part}_{query_type}.json"
            )
            sql_output = os.path.join(
                rule_outputs_base, f"dev_{first_part}_{query_type}.sql"
            )
            process_rule_folder_2nd_version(
                rule_outputs_folder,
                graph_data_base,
                rule_name,
                json_output,
                sql_output,
                query_type,
            )
            print(
                f"Files {os.path.basename(json_output)} and {os.path.basename(sql_output)} generated successfully for {rule_name}."
            )

    with open(rule_1_error_queries_path, "w") as f:
        json.dump(rule_1_error_queries_for_databases, f, indent=4)
    with open(rule_2_error_queries_path, "w") as f:
        json.dump(rule_2_error_queries_for_databases, f, indent=4)
    with open(rule_1_empty_queries_path, "w") as f:
        json.dump(rule_1_empty_queries_for_databases, f, indent=4)
    with open(rule_2_empty_queries_path, "w") as f:
        json.dump(rule_2_empty_queries_for_databases, f, indent=4)


def run_rule_3():
    print("Starting Rule 3: Hiding Path Information")

    # Configuration
    data_folder = "data/"
    rule_inputs_base = os.path.join(data_folder, "rule_inputs", "rules_1_2")

    # Define database IDs
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

    # Set up output directories
    output_base_dir = os.path.join(data_folder, "rule_outputs", "rule_3")
    train_output_base_dir = os.path.join(output_base_dir, "train")
    dev_output_base_dir = os.path.join(output_base_dir, "dev")
    test_output_base_dir = os.path.join(output_base_dir, "test")

    for dir_path in [
        output_base_dir,
        train_output_base_dir,
        dev_output_base_dir,
        test_output_base_dir,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Processing {len(db_ids)} databases...")

    # Load database subgraphs
    database_subgraphs = {
        db_id: r3_retrieve_all_dev_patterns(os.path.join(rule_inputs_base, db_id))
        for db_id in db_ids
    }

    # Split queries into train, dev, and test sets
    set_random_seed(42)  # Ensuring reproducibility
    train_queries, dev_queries, test_queries = split_queries(database_subgraphs)

    # Filter subgraphs for each set
    train_subgraphs = filter_subgraphs(database_subgraphs, train_queries)
    dev_subgraphs = filter_subgraphs(database_subgraphs, dev_queries)
    test_subgraphs = filter_subgraphs(database_subgraphs, test_queries)

    print(
        f"Train: {len(train_queries)} queries, {sum(len(subgraphs) for subgraphs in train_subgraphs.values())} subgraphs"
    )
    print(
        f"Dev: {len(dev_queries)} queries, {sum(len(subgraphs) for subgraphs in dev_subgraphs.values())} subgraphs"
    )
    print(
        f"Test: {len(test_queries)} queries, {sum(len(subgraphs) for subgraphs in test_subgraphs.values())} subgraphs"
    )

    # Process datasets to identify central tables
    print("Processing datasets to identify central tables...")
    train_subgraphs_with_central_tables = process_dataset(train_subgraphs)
    dev_subgraphs_with_central_tables = process_dataset(dev_subgraphs)
    test_subgraphs_with_central_tables = process_dataset(test_subgraphs)

    # Apply hiding techniques to train and dev sets
    print("Applying hiding techniques to train and dev sets...")

    # Synonym Replacement
    print("Applying Synonym Replacement...")
    train_queries_with_synonym_replacement = hide_tables_with_synonym_replacement(
        train_subgraphs_with_central_tables
    )
    dev_queries_with_synonym_replacement = hide_tables_with_synonym_replacement(
        dev_subgraphs_with_central_tables
    )

    # Back Translation
    print("Applying Back Translation...")
    train_queries_with_backtranslation = hide_tables_with_backtranslation(
        train_subgraphs_with_central_tables
    )
    dev_queries_with_backtranslation = hide_tables_with_backtranslation(
        dev_subgraphs_with_central_tables
    )

    # Contextual Augmentation
    print("Applying Contextual Augmentation...")
    train_queries_with_contextual_augmentation = (
        hide_tables_with_contextual_augmentation(train_subgraphs_with_central_tables)
    )
    dev_queries_with_contextual_augmentation = hide_tables_with_contextual_augmentation(
        dev_subgraphs_with_central_tables
    )

    # Save the train and dev data
    print("Saving train and dev data...")

    train_data = {
        "synonym_replacement": train_queries_with_synonym_replacement,
        "backtranslation": train_queries_with_backtranslation,
        "contextual_augmentation": train_queries_with_contextual_augmentation,
    }

    dev_data = {
        "synonym_replacement": dev_queries_with_synonym_replacement,
        "backtranslation": dev_queries_with_backtranslation,
        "contextual_augmentation": dev_queries_with_contextual_augmentation,
    }

    process_and_save_data_for_set(train_data, train_output_base_dir, "train")
    process_and_save_data_for_set(dev_data, dev_output_base_dir, "dev")

    # Now generate and save data for the test set
    print("Generating and saving data for the test set...")
    test_queries_with_synonym_replacement = hide_tables_with_synonym_replacement(
        test_subgraphs_with_central_tables
    )
    test_queries_with_backtranslation = hide_tables_with_backtranslation(
        test_subgraphs_with_central_tables
    )
    test_queries_with_contextual_augmentation = (
        hide_tables_with_contextual_augmentation(test_subgraphs_with_central_tables)
    )

    test_data = {
        "synonym_replacement": test_queries_with_synonym_replacement,
        "backtranslation": test_queries_with_backtranslation,
        "contextual_augmentation": test_queries_with_contextual_augmentation,
    }

    process_and_save_data_for_set(test_data, test_output_base_dir, "test")

    print("Rule 3 processing complete!")


if __name__ == "__main__":
    # run_rule_1_and_2()
    run_rule_3()

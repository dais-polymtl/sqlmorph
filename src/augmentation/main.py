import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.logger.logger import Logger
from src.core.database.adapters.duckdb_adapter import DuckDBAdapter
from rule_1_and_2 import (
    retrieve_all_dev_subgraphs,
    load_schema,
    extend_and_filter_subgraphs,
)
from rule_1_and_2.persistence import save_new_queries, save_old_extended_queries

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

logger = Logger(__name__)


def run_rule_1_and_2():
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

    rule_inputs_base = os.path.join(data_folder, "rule_inputs", "rules_1_2")
    graph_data_base = os.path.join(data_folder, "graph_data", "bird_graphs", "pickles")
    rule_outputs_base = os.path.join(data_folder, "rule_outputs", "rules_1_2")

    query_stats_path = os.path.join(rule_inputs_base, "query_statistics.csv")
    if not os.path.exists(query_stats_path):
        logger.log(
            level="error",
            action="missing_query_stats_file",
            details={"path": query_stats_path},
        )
        return

    df = pd.read_csv(query_stats_path)

    # Accumulate across DBs
    all_rule_1_pruned, all_rule_2_pruned = [], []
    all_rule_1_excluded, all_rule_2_excluded = [], []

    for db_id in db_ids:
        logger.log(
            level="info",
            action="Starting the augmentation for the following database.",
            details={"db_id": db_id},
        )

        rule_inputs_folder = os.path.join(rule_inputs_base, db_id)
        schema_path = os.path.join(graph_data_base, f"{db_id}_graph.pkl")
        db_path = os.path.join(
            data_folder, "benchmarks", "Bird", "dev_databases", db_id, f"{db_id}.sqlite"
        )

        if not os.path.exists(rule_inputs_folder):
            logger.log(
                "warning",
                "The augmentation input files are not found!",
                {"path": rule_inputs_folder, "db_id": db_id},
            )
            continue
        if not os.path.exists(schema_path):
            logger.log(
                "warning",
                "The schema file is not found!",
                {"path": schema_path, "db_id": db_id},
            )
            continue
        if not os.path.exists(db_path):
            logger.log(
                "warning",
                "The database file is not found!",
                {"path": db_path, "db_id": db_id},
            )
            continue

        subgraphs = retrieve_all_dev_subgraphs(rule_inputs_folder, logger)
        logger.log(
            "info",
            "Successful retrieval of the database subgraphs.",
            {"db_id": db_id, "Number of subgraphs": len(subgraphs)},
        )

        schema = load_schema(schema_path)
        adapter = DuckDBAdapter({"db_path": db_path})

        r1_pruned, r2_pruned, r1_excluded, r2_excluded = extend_and_filter_subgraphs(
            subgraphs, db_id, df, schema, adapter
        )
        logger.log(
            "info",
            "rule_patterns_count",
            {
                "db_id": db_id,
                "rule_1_patterns_count": len(r1_pruned),
                "rule_2_patterns_count": len(r2_pruned),
            },
        )

        all_rule_1_pruned.extend(r1_pruned)
        all_rule_2_pruned.extend(r2_pruned)
        all_rule_1_excluded.extend(r1_excluded)
        all_rule_2_excluded.extend(r2_excluded)

        logger.log("info", "queries_generated_successfully", {"db_id": db_id})

    # Save new & extended queries for Rule 1 & 2 — PRUNED
    for rule_id, pruned in [(1, all_rule_1_pruned), (2, all_rule_2_pruned)]:
        extension_type = "pruned"
        rule_folder = os.path.join(rule_outputs_base, extension_type, f"rule_{rule_id}")

        # Graph-first: explicit and dev_set_like
        for question_type in ["explicit", "dev_set_like"]:
            json_path = os.path.join(rule_folder, "graph_first", f"dev_{rule_id}_{question_type}.json")
            sql_path = os.path.join(rule_folder, "graph_first", f"dev_{rule_id}_{question_type}.sql")

            save_new_queries(pruned, graph_data_base, json_path, sql_path, question_type, extension_type)


        # Query-first: original and new
        for query_type in ["original", "new"]:
            logger.log("info", f"Processing Global Outputs for rule {rule_id} with {query_type} questions...")

            json_path = os.path.join(rule_folder, "query_first", f"dev_{rule_id}_{query_type}_extended_queries.json")
            sql_path = os.path.join(rule_folder, "query_first", f"dev_{rule_id}_{query_type}_extended_queries.sql")

            save_old_extended_queries(pruned, graph_data_base, json_path, sql_path, query_type, extension_type)

            print(
                f"Files {os.path.basename(json_path)} and {os.path.basename(sql_path)} generated successfully for rule_{rule_id}."
            )

    logger.log("info", "all_databases_processed_successfully", {"version": "first"})

    # Save new & extended queries for Rule 1 & 2 — EXCLUDED
    for rule_id, excluded in [(1, all_rule_1_excluded), (2, all_rule_2_excluded)]:
        extension_type = "excluded"
        rule_folder = os.path.join(rule_outputs_base, extension_type, f"rule_{rule_id}")

        # Graph-first: explicit and dev_set_like
        for question_type in ["explicit", "dev_set_like"]:
            json_path = os.path.join(rule_folder, "graph_first", f"dev_{rule_id}_{question_type}.json")
            sql_path = os.path.join(rule_folder, "graph_first", f"dev_{rule_id}_{question_type}.sql")

            save_new_queries(excluded, graph_data_base, json_path, sql_path, question_type, extension_type)

            logger.log(
                "info",
                "files_generated_successfully",
                {
                    "json_file": os.path.basename(json_path),
                    "sql_file": os.path.basename(sql_path),
                    "rule": rule_id,
                },
            )

        # Query-first: original and new
        for query_type in ["original", "new"]:
            logger.log("info", f"Processing Global Outputs for rule {rule_id} with {query_type} questions...")

            json_path = os.path.join(rule_folder, "query_first", f"dev_{rule_id}_{query_type}_extended_queries.json")
            sql_path = os.path.join(rule_folder, "query_first", f"dev_{rule_id}_{query_type}_extended_queries.sql")

            save_old_extended_queries(excluded, graph_data_base, json_path, sql_path, query_type, extension_type)

            print(
                f"Files {os.path.basename(json_path)} and {os.path.basename(sql_path)} generated successfully for rule_{rule_id}."
            )

    logger.log("info", "all_databases_processed_successfully", {"version": "second"})

def run_rule_3():
    print("Starting Rule 3: Hiding Path Information")

    # Configuration
    data_folder = "data/"
    rule_inputs_base = os.path.join(data_folder, "rule_inputs", "rules_1_2")

    # Define database IDs
    # db_ids = [
    #     "california_schools",
    #     "card_games",
    #     "codebase_community",
    #     "debit_card_specializing",
    #     "european_football_2",
    #     "financial",
    #     "formula_1",
    #     "student_club",
    #     "superhero",
    #     "thrombosis_prediction",
    #     "toxicology",
    # ]
    db_ids = [
        "card_games"
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
    run_rule_1_and_2()
    # run_rule_3()

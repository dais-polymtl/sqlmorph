import os
import sys
import pandas as pd
from dotenv import load_dotenv
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.logger.logger import Logger
from src.core.database.database_handler import DatabaseHandler, DBMS
from src.core.prompt_renderer.prompt_renderer import PromptRenderer
from src.core.model_manager.model_manager import ModelManager, ModelProvider, ModelType
from src.core.model_manager.openai_model import OpenAIModel

from jq_graph_augmentation import (
    retrieve_all_dev_subgraphs,
    load_schema,
    extend_and_filter_subgraphs,
)
from jq_graph_augmentation.persistence import (
    save_new_queries,
    save_old_extended_queries,
)

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
load_dotenv()


def get_environment_config():
    keys = [
        "DATABASE_IDS",
        "DATA_FOLDER",
        "RULE_INPUTS_BASE",
        "GRAPH_DATA_BASE",
        "RULE_OUTPUTS_BASE",
        "PROMPT_TEMPLATE_PATH",
        "OPENAI_API_KEY",
    ]
    config = {key: os.getenv(key) for key in keys}
    config["DATABASE_IDS"] = (
        config["DATABASE_IDS"].split(",") if config["DATABASE_IDS"] else []
    )

    if not all(config.values()):
        logger.log("error", "Missing one or more environment variables.", config)
        sys.exit(1)
    return config


def process_db(
    db_id,
    i,
    config,
    args,
    mode="n",  # mode="n" for queries_with_n_tables, "lt_n" for queries_with_lt_n_tables
):
    rule_inputs_folder = os.path.join(config["RULE_INPUTS_BASE"], db_id)
    schema_path = os.path.join(config["GRAPH_DATA_BASE"], f"{db_id}_graph.pkl")
    db_path = os.path.join(
        config["DATA_FOLDER"],
        "benchmarks",
        "Bird",
        "dev_databases",
        db_id,
        f"{db_id}.sqlite",
    )

    if not all(map(os.path.exists, [rule_inputs_folder, schema_path, db_path])):
        logger.log(
            "warning",
            f"Missing input files for {db_id}.",
            {
                "rule_inputs_folder": rule_inputs_folder,
                "schema_path": schema_path,
                "db_path": db_path,
            },
        )
        return [], [], [], []

    subgraphs = retrieve_all_dev_subgraphs(rule_inputs_folder, logger)
    logger.log("info", "Subgraphs retrieved", {"db_id": db_id, "count": len(subgraphs)})

    schema = load_schema(schema_path)
    adapter = DatabaseHandler(DBMS.DUCKDB, {"db_path": db_path})
    prompt_renderer = PromptRenderer(config["PROMPT_TEMPLATE_PATH"])
    model = ModelManager().create_model(
        model_provider=ModelProvider.OPENAI,
        model_type=ModelType.COMPLETION,
        model_name=OpenAIModel.GPT_4O,
        openai_api_key=config["OPENAI_API_KEY"],
    )

    query_first = args.query_first_augmentation
    graph_first = args.graph_first_augmentation
    n_tables = args.n_tables
    query_stats_path = os.path.join(config["RULE_INPUTS_BASE"], "query_statistics.csv")
    query_col_stats = pd.read_csv(query_stats_path)

    kept_n_tables, kept_lt_n_tables, excluded_n_tables, excluded_lt_n_tables = (
        extend_and_filter_subgraphs(
            subgraphs,
            db_id,
            query_col_stats,
            schema,
            adapter,
            logger,
            query_first,
            graph_first,
            n_tables[i],
            mode,
        )
    )
    if mode == "n":
        return kept_n_tables, excluded_n_tables, prompt_renderer, model
    if mode == "lt_n":
        return kept_lt_n_tables, excluded_lt_n_tables, prompt_renderer, model


def save_outputs(rule_id, items, extension_type, config, args, prompt_renderer, model):
    rule_folder = os.path.join(
        config["RULE_OUTPUTS_BASE"], extension_type, f"rule_{rule_id}"
    )

    if args.graph_first_augmentation:
        for question_type in ["explicit", "dev_set_like"]:
            json_path = os.path.join(
                rule_folder, "graph_first", f"dev_{rule_id}_{question_type}.json"
            )
            sql_path = os.path.join(
                rule_folder, "graph_first", f"dev_{rule_id}_{question_type}.sql"
            )
            save_new_queries(
                items,
                config["GRAPH_DATA_BASE"],
                json_path,
                sql_path,
                question_type,
                extension_type,
                prompt_renderer,
                model,
            )

    if args.query_first_augmentation:
        for query_type in ["original", "new"]:
            logger.log(
                "info", f"Saving query-first outputs for rule {rule_id} ({query_type})"
            )
            json_path = os.path.join(
                rule_folder,
                "query_first",
                f"dev_{rule_id}_{query_type}_extended_queries.json",
            )
            sql_path = os.path.join(
                rule_folder,
                "query_first",
                f"dev_{rule_id}_{query_type}_extended_queries.sql",
            )
            save_old_extended_queries(
                items,
                config["GRAPH_DATA_BASE"],
                json_path,
                sql_path,
                query_type,
                extension_type,
                prompt_renderer,
                model,
            )


def queries_with_n_tables(args):
    config = get_environment_config()
    kept_n, excluded_n = [], []

    for i, db_id in enumerate(config["DATABASE_IDS"]):
        kept_queries_n, excluded_queries_n, prompt_renderer, model = process_db(
            db_id, i, config, args, mode="n"
        )
        kept_n.extend(kept_queries_n)
        excluded_n.extend(excluded_queries_n)

    save_outputs(1, kept_n, "pruned", config, args, prompt_renderer, model)
    save_outputs(1, excluded_n, "excluded", config, args, prompt_renderer, model)

    logger.log("info", "All databases processed successfully", {"mode": "n_tables"})


def queries_with_lt_n_tables(args):
    config = get_environment_config()
    kept_lt_n, excluded_lt_n = [], []

    for i, db_id in enumerate(config["DATABASE_IDS"]):
        kept_queries_lt_n, excluded_queries_lt_n, prompt_renderer, model = process_db(
            db_id, i, config, args, mode="lt_n"
        )
        kept_lt_n.extend(kept_queries_lt_n)
        excluded_lt_n.extend(excluded_queries_lt_n)

    save_outputs(2, kept_lt_n, "pruned", config, args, prompt_renderer, model)
    save_outputs(2, excluded_lt_n, "excluded", config, args, prompt_renderer, model)

    logger.log("info", "All databases processed successfully", {"mode": "lt_n_tables"})


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
    db_ids = ["card_games"]

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


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    # def str2bool(v):
    #     return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description="Run augmentation code.")
    parser.add_argument(
        "--n_tables",
        type=int,
        nargs="+",
        help="List of table counts to generate queries up to n tables for each database.",
    )
    parser.add_argument(
        "--query_first_augmentation",
        type=str2bool,
        help="Run query first augmentation.",
    )
    parser.add_argument(
        "--graph_first_augmentation",
        type=str2bool,
        help="Run graph first augmentation.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    queries_with_n_tables(args)
    queries_with_lt_n_tables(args)
    # run_rule_3()

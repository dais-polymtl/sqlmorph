import os
import sys
from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.logger.logger import Logger


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


if __name__ == "__main__":
    run_rule_3()

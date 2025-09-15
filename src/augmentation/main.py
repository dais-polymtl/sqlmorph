import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from tabulate import tabulate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.logger.logger import Logger


from rule_3 import (
    retrieve_jqgs,
    compute_node_distribution_per_db,
    cluster_db_ids_by_node_dist,
    split_cluster_jqgs,
    process_dataset,
    visualize_clusters,
    hide_tables_with_synonym_replacement,
    hide_tables_with_backtranslation,
    hide_tables_with_contextual_augmentation,
    hiding_success_scores,
    process_and_save_data_for_set,
)

logger = Logger(__name__)
load_dotenv()


def run_rule_3():
    print("Starting Rule 3: Hiding Path Information")

    data_folder = Path(os.getenv("DATA_FOLDER"))
    rule_inputs_base = data_folder / "rule_inputs" / "lt_elimination"
    output_base_dir = data_folder / "rule_outputs" / "lt_elimination"

    db_ids = [f.name for f in rule_inputs_base.iterdir() if f.is_dir()]
    logger.log("info", f"Found {len(db_ids)} databases in rule inputs.")
    for split in ["train", "dev", "test"]:
        os.makedirs(output_base_dir / split, exist_ok=True)

    # Load and process dataset
    database_jqgs = [
        jqg for db in db_ids for jqg in retrieve_jqgs(rule_inputs_base / db)
    ]
    database_jqgs_with_lt = process_dataset(database_jqgs)

    logger.log(
        "info", f"Loaded {len(database_jqgs)} queries with possible link tables."
    )
    logger.log(
        "info", f"Processed {len(database_jqgs_with_lt)} queries with link tables."
    )

    # Cluster and split
    db_node_dist = compute_node_distribution_per_db(database_jqgs_with_lt)
    clustered_db_ids = cluster_db_ids_by_node_dist(db_node_dist, n_clusters=2)
    for cid, ids in clustered_db_ids.items():
        logger.log("info", f"Cluster {cid} contains {len(ids)} databases.")
        visualize_clusters(
            db_node_dist=db_node_dist,
            clustered=clustered_db_ids,
            output_path=output_base_dir / f"clusters_{cid}.png",
        )

    largest_cluster = max(clustered_db_ids, key=lambda cid: len(clustered_db_ids[cid]))

    train_set, dev_set, test_set = split_cluster_jqgs(
        database_jqgs_with_lt,
        clustered_db_ids[largest_cluster],
        test_size=100,
        dev_ratio=0.2,
        seed=35,
    )

    for split in ["train", "dev", "test"]:
        folder_path = output_base_dir / split
        os.makedirs(folder_path, exist_ok=True)
        file_path = folder_path / f"{split}_jqgs_with_lt.pkl"
        if split == "train":
            data_to_save = train_set
        elif split == "dev":
            data_to_save = dev_set
        elif split == "test":
            data_to_save = test_set
        else:
            raise ValueError(f"Unknown split: {split}")
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)
    logger.log(
        "info",
        f"Saved {len(train_set)} train, {len(dev_set)} dev, and {len(test_set)} test sets to {output_base_dir}.",
    )

    np.random.seed(35)
    train_set = train_set[:50]  # limit to 50 for faster testing
    np.random.shuffle(train_set)
    np.random.shuffle(dev_set)
    np.random.shuffle(test_set)

    train_sr = hide_tables_with_synonym_replacement(train_set)
    train_bt = hide_tables_with_backtranslation(train_set)
    train_ca = hide_tables_with_contextual_augmentation(train_set)

    dev_sr = hide_tables_with_synonym_replacement(dev_set)
    dev_bt = hide_tables_with_backtranslation(dev_set)
    dev_ca = hide_tables_with_contextual_augmentation(dev_set)

    test_sr = hide_tables_with_synonym_replacement(test_set)
    test_bt = hide_tables_with_backtranslation(test_set)
    test_ca = hide_tables_with_contextual_augmentation(test_set)

    train_sr_results = hiding_success_scores(train_sr)
    train_bt_results = hiding_success_scores(train_bt)
    train_ca_results = hiding_success_scores(train_ca)
    # scores
    sr_results = hiding_success_scores(dev_sr)
    bt_results = hiding_success_scores(dev_bt)
    ca_results = hiding_success_scores(dev_ca)

    test_sr_results = hiding_success_scores(test_sr)
    test_bt_results = hiding_success_scores(test_bt)
    test_ca_results = hiding_success_scores(test_ca)

    train_data = {
        "syn_rep": train_sr_results["queries_w_scores"],
        "backtrans": train_bt_results["queries_w_scores"],
        "context_aug": train_ca_results["queries_w_scores"],
    }

    dev_data = {
        "syn_rep": sr_results["queries_w_scores"],
        "backtrans": bt_results["queries_w_scores"],
        "context_aug": ca_results["queries_w_scores"],
    }

    test_data = {
        "syn_rep": test_sr_results["queries_w_scores"],
        "backtrans": test_bt_results["queries_w_scores"],
        "context_aug": test_ca_results["queries_w_scores"],
    }

    print("\nTraining Set Hiding Success Scores:")
    headers = [
        "Technique",
        "Q Score",
        "E Score",
    ]
    table = []
    table.append(
        [
            "Synonym Replacement",
            round(train_sr_results["averages"]["question_score"], 3),
            round(train_sr_results["averages"]["evidence_score"], 3),
        ]
    )
    table.append(
        [
            "Backtranslation",
            round(train_bt_results["averages"]["question_score"], 3),
            round(train_bt_results["averages"]["evidence_score"], 3),
        ]
    )
    table.append(
        [
            "Contextual Augmentation",
            round(train_ca_results["averages"]["question_score"], 3),
            round(train_ca_results["averages"]["evidence_score"], 3),
        ]
    )
    print(tabulate(table, headers=headers, tablefmt="grid"))

    print("\nHiding Success Scores:")
    headers = [
        "Technique",
        "Q Score",
        "E Score",
    ]
    table = []
    table.append(
        [
            "Synonym Replacement",
            round(sr_results["averages"]["question_score"], 3),
            round(sr_results["averages"]["evidence_score"], 3),
        ]
    )
    table.append(
        [
            "Backtranslation",
            round(bt_results["averages"]["question_score"], 3),
            round(bt_results["averages"]["evidence_score"], 3),
        ]
    )
    table.append(
        [
            "Contextual Augmentation",
            round(ca_results["averages"]["question_score"], 3),
            round(ca_results["averages"]["evidence_score"], 3),
        ]
    )

    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("\nTest Set Hiding Success Scores:")
    headers = [
        "Technique",
        "Q Score",
        "E Score",
    ]

    table = []
    table.append(
        [
            "Synonym Replacement",
            (
                round(test_sr_results["averages"].get("question_score") or 0.0, 3)
                if test_sr_results["averages"].get("question_score") is not None
                else "N/A"
            ),
            (
                round(test_sr_results["averages"].get("evidence_score") or 0.0, 3)
                if test_sr_results["averages"].get("evidence_score") is not None
                else "N/A"
            ),
        ]
    )
    table.append(
        [
            "Backtranslation",
            (
                round(test_bt_results["averages"].get("question_score") or 0.0, 3)
                if test_bt_results["averages"].get("question_score") is not None
                else "N/A"
            ),
            (
                round(test_bt_results["averages"].get("evidence_score") or 0.0, 3)
                if test_bt_results["averages"].get("evidence_score") is not None
                else "N/A"
            ),
        ]
    )
    table.append(
        [
            "Contextual Augmentation",
            (
                round(test_ca_results["averages"].get("question_score") or 0.0, 3)
                if test_ca_results["averages"].get("question_score") is not None
                else "N/A"
            ),
            (
                round(test_ca_results["averages"].get("evidence_score") or 0.0, 3)
                if test_ca_results["averages"].get("evidence_score") is not None
                else "N/A"
            ),
        ]
    )

    print(tabulate(table, headers=headers, tablefmt="grid"))

    # Save results
    process_and_save_data_for_set(train_data, output_base_dir / "train", "train")
    process_and_save_data_for_set(dev_data, output_base_dir / "dev", "dev")
    process_and_save_data_for_set(test_data, output_base_dir / "test", "test")


if __name__ == "__main__":
    run_rule_3()

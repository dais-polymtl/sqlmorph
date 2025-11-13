import os
from pathlib import Path
from typing import List, Tuple
import json

from src.augmentation.join_query_expansion.sql_query_gen import (
    aug_n_table_sql_queries,
)
from src.augmentation.join_query_expansion.nl_query_gen import (
    gen_nl,
)

from src.augmentation.join_query_expansion.persistence import (
    save_graph_first,
    save_query_first,
    update_augmentation_log,
)


def store_sql_nl_pairs(filtered_aug, discarded_aug, file_name, graph_first=False):
    save_query_first(
        queries=filtered_aug,
        output_file=file_name + "qf_filtered",
        status="filtered",
        mode="query_first",
    )

    save_query_first(
        queries=discarded_aug,
        output_file=file_name + "qf_discarded",
        status="discarded",
        mode="query_first",
    )

    if graph_first:
        save_graph_first(
            queries=filtered_aug,
            output_file=file_name + "gf_filtered",
            status="filtered",
            mode="graph_first",
        )

        save_graph_first(
            queries=discarded_aug,
            output_file=file_name + "gf_discarded",
            status="discarded",
            mode="graph_first",
        )


def generate_aug_queries(db_ids):
    for db_id, max_tables in db_ids:
        for i in range(1, max_tables):
            filtered_aug, discarded_aug, all_extensions = aug_n_table_sql_queries(
                db_id=db_id,
                num_tables=i + 1,
                graph_first=True,
            )
            filtered_aug = gen_nl(
                filtered_aug,
                graph_first=True,
            )
            discarded_aug = gen_nl(
                discarded_aug,
                graph_first=True,
            )
            store_sql_nl_pairs(
                filtered_aug,
                discarded_aug,
                file_name=f"{db_id}_{i + 1}t_",
                graph_first=True,
            )
            update_augmentation_log(db_id, i, all_extensions)


def collect_augmented_sql_jsons(db_ids: List[Tuple[str, str]]) -> None:
    """
    Collects JSON list contents from filtered and discarded SQL-NL augmented files (query-first and graph-first),
    and aggregates them into consolidated JSON array files.

    Args:
        db_ids (List[Tuple[str, str]]): List of (db_id, db_name) tuples.
    """
    base_output_path = Path(os.getenv("RULE_OUTPUTS_BASE"))
    data_folder_path = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation"

    aggregated_data = {
        "graph_first": {
            "dev": {"filtered": [], "discarded": []},
            "exp": {"filtered": [], "discarded": []},
        },
        "query_first": {"filtered": [], "discarded": []},
    }

    for db_id, _ in db_ids:
        for status in ["filtered", "discarded"]:
            for mode, short in [("graph_first", "gf"), ("query_first", "qf")]:
                folder_path = base_output_path / status / mode

                if mode == "graph_first":
                    for split in ["dev", "exp"]:
                        pattern = f"{db_id}_*t_{short}_{status}_{split}.json"
                        for file in folder_path.glob(pattern):
                            try:
                                data = json.load(open(file))
                                aggregated_data[mode][split][status].extend(data)
                            except Exception as e:
                                print(f"[ERROR] Failed to parse {file}: {e}")
                else:  # query_first
                    pattern = f"{db_id}_*t_{short}_{status}_aug.json"
                    for file in folder_path.glob(pattern):
                        try:
                            data = json.load(open(file))
                            aggregated_data[mode][status].extend(data)
                        except Exception as e:
                            print(f"[ERROR] Failed to parse {file}: {e}")
    # Write graph_first outputs
    gf_path = data_folder_path / "graph_first"
    gf_path.mkdir(parents=True, exist_ok=True)
    for split in ["dev", "exp"]:
        for status in ["filtered", "discarded"]:
            out_file = gf_path / f"bird_gf_{status}_{split}.json"
            with open(out_file, "w") as f:
                json.dump(aggregated_data["graph_first"][split][status], f, indent=4)

    # Write query_first outputs
    qf_path = data_folder_path / "query_first"
    qf_path.mkdir(parents=True, exist_ok=True)
    for status in ["filtered", "discarded"]:
        out_file = qf_path / f"bird_qf_{status}.json"
        with open(out_file, "w") as f:
            json.dump(aggregated_data["query_first"][status], f, indent=4)


def main(db_ids):
    # generate_aug_queries(db_ids)
    collect_augmented_sql_jsons(db_ids)


if __name__ == "__main__":
    db_ids = [
        ("california_schools", 2),
        ("card_games", 3),
        ("codebase_community", 4),
        ("debit_card_specializing", 3),
        ("european_football_2", 4),
        ("financial", 5),
        ("formula_1", 4),
        ("student_club", 4),
        ("superhero", 4),
        ("thrombosis_prediction", 3),
        ("toxicology", 4),
    ]

    main(db_ids)

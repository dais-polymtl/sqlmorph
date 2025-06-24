from src.augmentation.jq_graph_augmentation.sql_query_gen import (
    aug_n_table_sql_queries,
)
from src.augmentation.jq_graph_augmentation.nl_query_gen import (
    gen_nl,
)

from src.augmentation.jq_graph_augmentation.persistence import (
    save_graph_first,
    save_query_first,
)


def store_sql_nl_pairs(filtered_aug, discarded_aug, file_name, graph_first=False):
    save_query_first(
        queries=filtered_aug, output_file=file_name + "query_first_filtered"
    )

    save_query_first(
        queries=discarded_aug, output_file=file_name + "query_first_discarded"
    )

    if graph_first:
        save_graph_first(
            queries=filtered_aug, output_file=file_name + "filtered_graph_first"
        )

        save_graph_first(
            queries=discarded_aug, output_file=file_name + "discarded_graph_first"
        )


def generate_aug_queries(db_ids):
    for db_id, max_tables in db_ids:
        # for i in range(1, max_tables):
        filtered_aug, discarded_aug = aug_n_table_sql_queries(
            db_id=db_id,
            num_tables=max_tables,
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
            file_name=f"{db_id}_queires_{max_tables}_tables_",
            graph_first=True,
        )


def main(db_ids):
    generate_aug_queries(db_ids)


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
        ("toxicology", 3),
    ]
    main(db_ids)

import os
import sys
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.logger.logger import Logger


from .sql_query_gen import (
    aug_n_table_sql_queries,
)
from .nl_query_gen import (
    gen_nl,
)
from .persistence import (
    save_graph_first,
    save_query_first,
)

logger = Logger(__name__)


def store_sql_nl_pairs(filtered_aug, discarded_aug, graph_first=False):
    save_query_first(queries=filtered_aug, output_file="query_first_filtered")

    save_query_first(queries=discarded_aug, output_file="query_first_discarded")

    if graph_first:
        save_graph_first(queries=filtered_aug, output_file="filtered_graph_first")

        save_graph_first(queries=discarded_aug, output_file="discarded_graph_first")


def main(args):
    filtered_aug, discarded_aug, all_extensions = aug_n_table_sql_queries(
        db_id=args.db_id,
        num_tables=args.num_tables,
        graph_first=args.graph_first,
    )
    # filtered_aug = gen_nl(
    #     filtered_aug,
    #     graph_first=args.graph_first,
    # )
    # discarded_aug = gen_nl(
    #     discarded_aug,
    #     graph_first=args.graph_first,
    # )
    # store_sql_nl_pairs(filtered_aug, discarded_aug, graph_first=args.graph_first)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run augmentation code.")
    parser.add_argument(
        "db_id",
        type=str,
        help="The database id for which to augment queries.",
    )
    parser.add_argument(
        "num_tables",
        type=int,
        help="Number of tables involved in the augmented queries.",
    )
    parser.add_argument(
        "-g",
        "--graph_first",
        type=str2bool,
        default=False,
        help="Run graph first augmentation.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    main(args)

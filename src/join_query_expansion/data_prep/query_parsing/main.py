from query_loader import load_queries, split_nested_flat_queries
from query_parser import parse_queries
from persistence import save_interm_queries, save_non_flattened_queries
from jq_graph_creation import create_jq_graphs
from src.core.logger.logger import Logger

logger = Logger(__name__)


def main(dataset: str, force_parsing: bool, jq_graph: bool) -> None:
    if not force_parsing and not jq_graph:
        logger.log("info", f"Starting query flattening pipeline for: {dataset}")

        all_qs = load_queries(dataset)
        nested_qs, flat_qs = split_nested_flat_queries(all_qs)
        logger.log(
            "info",
            f"Loaded {len(all_qs)} queries: {len(flat_qs)} flat, {len(nested_qs)} nested",
        )

        parsed_qs = parse_queries(flat_qs)

        # auto_flat_qs, still_nested = flatten_queries_automatically(nested_qs, dataset)
        # logger.log(
        #     "info",
        #     f"Auto-flattened {len(auto_flat_qs)} queries, {len(still_nested)} left for manual",
        # )

        # parsed_auto = parse_queries(auto_flat_qs)

        # manual_flat_qs, non_flattened = flatten_queries_manually(still_nested, dataset)
        # logger.log("info", f"Manually flattened {len(manual_flat_qs)} queries")

        # parsed_manual = parse_queries(manual_flat_qs)

        # for db_id, qs in parsed_auto.items():
        #     parsed_qs.setdefault(db_id, []).extend(qs)
        # for db_id, qs in parsed_manual.items():
        #     parsed_qs.setdefault(db_id, []).extend(qs)

        save_interm_queries(parsed_qs, dataset)
        save_non_flattened_queries(nested_qs, dataset)

        # logger.log(
        #     "info",
        #     f"Pipeline completed. Total queries saved: {sum(len(q) for q in parsed_qs.values())}",
        # )

    # elif force_parsing:
    #     enter_tables_joins_manually(dataset)
    #     logger.log("info", f"Manual entry for tables and joins completed for {dataset}")

    elif jq_graph:
        create_jq_graphs(dataset)


if __name__ == "__main__":
    main("Bird", force_parsing=False, jq_graph=True)

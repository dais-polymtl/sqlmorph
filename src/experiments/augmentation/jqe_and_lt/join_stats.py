import os
import re
import json
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlglot
import sqlglot.expressions as exp
import networkx as nx
from sqlglot.expressions import EQ, GT, GTE, LT, LTE, NEQ
from src.core.logger.logger import Logger

logger = Logger(__name__)

COMPARISON_OPS = (EQ, GT, GTE, LT, LTE, NEQ)
AGG_FUNCTIONS = {"max", "min", "sum", "count", "avg"}


def count_conditions(where_expr: Optional[exp.Expression]) -> int:
    """
    Count the number of comparison conditions in the WHERE clause.

    Args:
        where_expr (Optional[exp.Expression]): The WHERE clause expression.

    Returns:
        int: Number of comparison conditions.
    """
    if where_expr is None:
        return 0
    return sum(1 for expr in where_expr.walk() if isinstance(expr, COMPARISON_OPS))


def has_aggregation(parsed: exp.Expression) -> bool:
    """
    Determine if the parsed SQL query contains aggregation functions.

    Args:
        parsed (exp.Expression): Parsed SQL expression.

    Returns:
        bool: True if any aggregation function is present, False otherwise.
    """
    return any(
        expr.__class__.__name__.lower() in AGG_FUNCTIONS for expr in parsed.walk()
    )


def join_details(
    pruned_aug: List[Dict[str, Any]], generated_aug: List[Dict[str, Any]]
) -> None:
    """
    Collect and log statistics about join graphs for pruned, generated, and original queries.

    Args:
        pruned_aug (List[Dict[str, Any]]): Pruned query augmentation entries.
        generated_aug (List[Dict[str, Any]]): Generated query augmentation entries.
    """
    base = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "query_first"
    )
    pruned_aug_data = json.load(open(base / "bird_qf_filtered.json", encoding="utf-8"))
    generated_aug_data = json.load(
        open(base / "bird_qf_discarded.json", encoding="utf-8")
    )
    rule_input_base = Path(os.getenv("RULE_INPUTS_BASE"))

    def stats(
        g: nx.Graph, qid: str, dbid: str, query: str, set_type: str
    ) -> Dict[str, Any]:
        """
        Extract join graph statistics from a query and its join graph.

        Args:
            g (nx.Graph): Join graph.
            qid (str): Question ID.
            dbid (str): Database ID.
            query (str): SQL query string.
            set_type (str): Set label ('original', 'pruned', or 'generated').

        Returns:
            Dict[str, Any]: Dictionary of statistics for the query.
        """
        n = g.number_of_nodes()
        d = list(dict(g.degree()).values())
        c = nx.cycle_basis(g)

        join_count = len(re.findall(r"\bJOIN\b", query, flags=re.IGNORECASE))
        parsed = sqlglot.parse_one(query, dialect="mysql")

        projection_count = len(parsed.expressions)
        where = parsed.args.get("where")
        condition_count = count_conditions(where)

        group_by = parsed.args.get("group") is not None
        having = parsed.args.get("having") is not None
        aggregation = has_aggregation(parsed)

        return {
            "id": qid,
            "db_id": dbid,
            "num_nodes": n,
            "num_edges": g.number_of_edges(),
            "num_joins": join_count,
            "avg_degree": sum(d) / n if n else 0,
            "is_cyclic": bool(c),
            "num_cycles": len(c),
            "num_conditions": condition_count,
            "num_projections": projection_count,
            "has_group_by": group_by,
            "has_having": having,
            "has_aggregation": aggregation,
            "set": set_type,
        }

    def get_id(q: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        """
        Find the question ID from the dataset matching the query.

        Args:
            q (Dict[str, Any]): The query entry.
            data (List[Dict[str, Any]]): Dataset to search.

        Returns:
            str: Matching question ID.
        """
        return next(i["question_id"] for i in data if i["SQL"] == q["query_first"])

    pruned_details = [
        stats(
            q["ext_jq_graph"],
            get_id(q, pruned_aug_data),
            q["db_id"],
            q["query_first"],
            "pruned",
        )
        for q in pruned_aug
    ]

    generated_details = [
        stats(
            q["ext_jq_graph"],
            get_id(q, generated_aug_data),
            q["db_id"],
            q["query_first"],
            "generated",
        )
        for q in generated_aug
    ]

    augmented_details = pruned_details + generated_details

    original_details: List[Dict[str, Any]] = []
    original_queries_path = rule_input_base / "jq_augmentation"

    for folder in original_queries_path.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.glob("*.pkl"):
            with open(file, "rb") as f:
                entries = pickle.load(f)
                for d in entries:
                    query = d.get("flattened_query") or d.get("SQL")
                    original_details.append(
                        stats(
                            d["jq_graph"],
                            d["question_id"],
                            d["db_id"],
                            query,
                            "original",
                        )
                    )

    output_dir = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "id",
        "db_id",
        "num_nodes",
        "num_edges",
        "num_joins",
        "avg_degree",
        "is_cyclic",
        "num_cycles",
        "set",
        "num_conditions",
        "num_projections",
        "has_group_by",
        "has_having",
        "has_aggregation",
    ]

    def log_distribution_for_number_edges(
        name: str, data: List[Dict[str, Any]]
    ) -> None:
        """
        Log edge distribution for a given dataset.

        Args:
            name (str): Label for the dataset.
            data (List[Dict[str, Any]]): Join details.
        """
        edge_dist: Dict[int, int] = {}
        for d in data:
            e = d["num_edges"]
            edge_dist[e] = edge_dist.get(e, 0) + 1
        logger.log(
            "info", f"Edge distribution for {name}: {dict(sorted(edge_dist.items()))}"
        )

    for name, data in [
        ("augmented", augmented_details),
        ("original", original_details),
    ]:
        path = output_dir / f"{name}_join_details.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        logger.log("info", f"{name.capitalize()} augmentation details saved to {path}")
        log_distribution_for_number_edges(name, data)


def read_queries_w_jqg():
    rule_outputs_base = Path(os.getenv("RULE_OUTPUTS_BASE"))
    aug_log_path = rule_outputs_base / "aug_log" / "augmentation_log.pickle"

    if not aug_log_path.exists():
        logger.log("error", f"Augmentation log file not found: {aug_log_path}")
        return []
    try:
        with open(aug_log_path, "rb") as f:
            logged_extensions = pickle.load(f)
    except Exception as e:
        logger.log("error", f"Error reading augmentation log file: {e}")
        return [], []

    generated_aug = [log for log in logged_extensions if log["status"] == "generated"]
    pruned_aug = [log for log in logged_extensions if log["status"] == "pruned"]

    return pruned_aug, generated_aug


def main():
    """
    Main function to execute the join details collection and logging.
    """
    pruned_aug, generated_aug = read_queries_w_jqg()
    print(f"Pruned Augmentation Entries: {len(pruned_aug)}")
    print(f"Generated Augmentation Entries: {len(generated_aug)}")
    print(f"Total Augmentation Entries: {len(pruned_aug) + len(generated_aug)}")
    if not pruned_aug or not generated_aug:
        logger.log("error", "No augmentation data found.")
        return

    join_details(pruned_aug, generated_aug)
    logger.log("info", "Join details collection completed.")


if __name__ == "__main__":
    main()

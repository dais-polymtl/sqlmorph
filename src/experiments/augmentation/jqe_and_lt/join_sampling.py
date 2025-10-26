from pathlib import Path
import os
import json
import csv
from collections import defaultdict
from itertools import product
from random import sample
from typing import Dict, List, Tuple, Set, DefaultDict, Union, Optional
from src.core.logger.logger import Logger

logger = Logger(name=__name__)


def extract_query_stats() -> Dict[
    str,
    Union[
        DefaultDict[Tuple[int, int], List[int]],
        DefaultDict[Tuple[int, int], int],
        DefaultDict[Tuple[int, int], List[int]],
        DefaultDict[Tuple[int, int], DefaultDict[Tuple[int, int], int]],
        DefaultDict[Tuple[int, int], int],
        DefaultDict[Tuple[int, int], int],
        DefaultDict[Tuple[int, int], int],
        DefaultDict[
            Tuple[int, int],
            DefaultDict[Tuple[int, int], DefaultDict[str, List[int]]],
        ],
    ],
]:
    """
    Group queries by node/edge structure and collect statistics on SQL query features.
    """
    path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "augmented_join_details.csv"
    )

    node_edge_groups: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    cyclic_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    acyclic_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    cycle_counts: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    condition_projection_combinations: DefaultDict[
        Tuple[int, int], DefaultDict[Tuple[int, int], int]
    ] = defaultdict(lambda: defaultdict(int))
    groupby_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    having_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    aggregation_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    combo_dbids: DefaultDict[
        Tuple[int, int], DefaultDict[Tuple[int, int], DefaultDict[str, List[int]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["num_nodes"]), int(row["num_edges"]))
            qid = int(row["id"])
            dbid = row["db_id"].strip()
            node_edge_groups[key].append(qid)

            is_cyclic = row["is_cyclic"].strip().lower() == "true"
            (cyclic_counts if is_cyclic else acyclic_counts)[key] += 1
            cycle_counts[key].append(int(row["num_cycles"]))

            num_conditions = int(row["num_conditions"])
            num_projections = int(row["num_projections"])
            combo = (num_conditions, num_projections)
            condition_projection_combinations[key][combo] += 1
            combo_dbids[key][combo][dbid].append(qid)

            if row["has_group_by"].strip().lower() == "true":
                groupby_counts[key] += 1
            if row["has_having"].strip().lower() == "true":
                having_counts[key] += 1
            if row["has_aggregation"].strip().lower() == "true":
                aggregation_counts[key] += 1

    return {
        "node_edge_groups": node_edge_groups,
        "cyclic_counts": cyclic_counts,
        "acyclic_counts": acyclic_counts,
        "cycle_counts": cycle_counts,
        "condition_projection_combinations": condition_projection_combinations,
        "groupby_counts": groupby_counts,
        "having_counts": having_counts,
        "aggregation_counts": aggregation_counts,
        "combo_dbids": combo_dbids,
    }


def write_query_summary(output_file: Union[str, Path] = "join_group_stats.csv") -> None:
    """
    Save a CSV summarizing the grouped query statistics.
    """
    stats = extract_query_stats()

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Nodes",
                "Edges",
                "Conditions",
                "Projections",
                "Total",
                "Cyclic",
                "Acyclic",
                "AvgCycles",
                "GroupBy",
                "Having",
                "Aggregation",
                "DBs",
            ]
        )

        for key in sorted(stats["node_edge_groups"]):
            total = len(stats["node_edge_groups"][key])
            cyclic = stats["cyclic_counts"].get(key, 0)
            acyclic = stats["acyclic_counts"].get(key, 0)
            avg_cycles = sum(stats["cycle_counts"][key]) / total if total else 0
            group_by = stats["groupby_counts"].get(key, 0)
            having = stats["having_counts"].get(key, 0)
            agg = stats["aggregation_counts"].get(key, 0)
            combos = stats["condition_projection_combinations"][key]
            combo_dbids = stats["combo_dbids"]

            for (cond, proj), count in sorted(combos.items()):
                dbs = combo_dbids[key][(cond, proj)]
                db_counts = [
                    f"{dbid}({len(qids)})"
                    for dbid, qids in sorted(dbs.items(), key=lambda x: -len(x[1]))
                ]
                db_string = ", ".join(db_counts)

                writer.writerow(
                    [
                        key[0],
                        key[1],
                        cond,
                        proj,
                        count,
                        cyclic,
                        acyclic,
                        f"{avg_cycles:.2f}",
                        group_by,
                        having,
                        agg,
                        db_string,
                    ]
                )


def generate_balanced_plan(
    summary_file: Union[str, Path] = "join_group_stats.csv",
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    Create a sampling plan to ensure balanced coverage across target join graph patterns.
    """
    node_edge_targets: Set[Tuple[int, int]] = {(2, 1), (3, 2), (4, 3), (5, 4)}
    combo_targets: Set[Tuple[int, int]] = set(product(range(7), range(7)))
    plan: DefaultDict[Tuple[int, int], DefaultDict[Tuple[int, int], Dict[str, int]]] = (
        defaultdict(lambda: defaultdict(dict))
    )

    with open(summary_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes = int(row["Nodes"])
            edges = int(row["Edges"])
            cond = int(row["Conditions"])
            proj = int(row["Projections"])

            if (nodes, edges) not in node_edge_targets or (
                cond,
                proj,
            ) not in combo_targets:
                continue

            for db_entry in row["DBs"].split(","):
                db_entry = db_entry.strip()
                if "(" not in db_entry:
                    continue
                dbid, count_str = db_entry[:-1].split("(")
                plan[(cond, proj)][(nodes, edges)][dbid] = int(count_str)

    sampling_limits: DefaultDict[Tuple[int, int], Dict[str, int]] = defaultdict(dict)
    for combo in combo_targets:
        if not all(ne in plan[combo] for ne in node_edge_targets):
            continue

        db_common = set.intersection(
            *(set(plan[combo][ne]) for ne in node_edge_targets)
        )
        for dbid in db_common:
            sampling_limits[combo][dbid] = min(
                plan[combo][ne][dbid] for ne in node_edge_targets
            )

    print("=== Balanced Sampling Plan ===")
    for combo in sorted(sampling_limits):
        db_parts = [
            f"{dbid}({sampling_limits[combo][dbid]})"
            for dbid in sorted(sampling_limits[combo])
        ]
        print(f"Combo {combo}: {', '.join(db_parts)}")

    return sampling_limits


def select_queries(
    detailed_file: Union[str, Path] = "augmented_join_details.csv",
    sampling_limits: Optional[Dict[Tuple[int, int], Dict[str, int]]] = None,
    node_edge_targets: Set[Tuple[int, int]] = {(2, 1), (3, 2), (4, 3), (5, 4)},
    combo_targets: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], Dict[Tuple[int, int], Dict[str, List[int]]]]:
    """
    Sample query IDs per (cond, proj), node-edge, dbid combo from detailed file.
    """
    if sampling_limits is None:
        raise ValueError("sampling_limits must be provided")
    if combo_targets is None:
        combo_targets = set(sampling_limits.keys())

    queries: DefaultDict[
        Tuple[int, int], DefaultDict[Tuple[int, int], DefaultDict[str, List[int]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(detailed_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["id"])
            dbid = row["db_id"]
            nodes = int(row["num_nodes"])
            edges = int(row["num_edges"])
            cond = int(row["num_conditions"])
            proj = int(row["num_projections"])

            if (nodes, edges) in node_edge_targets and (cond, proj) in combo_targets:
                queries[(cond, proj)][(nodes, edges)][dbid].append(qid)

    sampled: DefaultDict[
        Tuple[int, int], DefaultDict[Tuple[int, int], DefaultDict[str, List[int]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for combo in sampling_limits:
        for dbid, limit in sampling_limits[combo].items():
            for group in node_edge_targets:
                group_queries = queries.get(combo, {}).get(group, {}).get(dbid, [])
                selected = (
                    group_queries
                    if len(group_queries) <= limit
                    else sample(group_queries, limit)
                )
                sampled[combo][group][dbid].extend(selected)

    return sampled


def export_sampled_queries(
    sampled: Dict[Tuple[int, int], Dict[Tuple[int, int], Dict[str, List[int]]]],
) -> None:
    """
    Export sampled queries as JSON and SQL files.
    """
    base = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
    )
    base.mkdir(parents=True, exist_ok=True)
    json_out = base / "sampled_queries.json"
    sql_out = base / "sampled_queries.sql"

    pruned_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_filtered.json"
    )
    gen_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_discarded.json"
    )

    with open(pruned_path, "r", encoding="utf-8") as f:
        pruned = json.load(f)
    with open(gen_path, "r", encoding="utf-8") as f:
        generated = json.load(f)
    all_queries = pruned + generated

    final: List[dict] = []
    for combo in sampled:
        for group in sampled[combo]:
            for dbid in sampled[combo][group]:
                for qid in sampled[combo][group][dbid]:
                    query = next(
                        (q for q in all_queries if q["question_id"] == qid), None
                    )
                    if query:
                        final.append(query)

    final.sort(key=lambda x: x["question_id"])

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    with open(sql_out, "w", encoding="utf-8") as f:
        for q in final:
            f.write(f"{q['SQL']}\t{q['db_id']}\n")


def main() -> None:
    summary_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "aug_queries_details.csv"
    )
    write_query_summary(summary_path)

    limits = generate_balanced_plan(summary_path)
    total = sum(count for db_counts in limits.values() for count in db_counts.values())
    print(f"Total queries to sample: {total}")

    # detailed_file = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "experiment_inputs" / "augmented_join_details.csv"
    # sampled = select_queries(detailed_file, sampling_limits=limits)

    # print("Sampled queries by group:")
    # for combo, groups in sampled.items():
    #     for group, dbs in groups.items():
    #         for dbid, qids in dbs.items():
    #             print(f"Combo {combo}, Group {group}, DB {dbid}: {len(qids)} queries")

    # total_sampled = sum(len(qids) for g in sampled.values() for d in g.values() for qids in d.values())
    # print(f"Total sampled queries: {total_sampled}")

    # export_sampled_queries(sampled)


if __name__ == "__main__":
    main()

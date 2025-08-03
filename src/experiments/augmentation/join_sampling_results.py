from pathlib import Path
import os
import csv
import json
import numpy as np
from typing import Dict, List

from src.experiments.augmentation.utils import compute_exec_accuracy
from src.core.logger.logger import Logger

logger = Logger(__name__)


def load_join_buckets(details_path: Path) -> Dict[int, List[int]]:
    """
    Loads query IDs grouped by their number of JOINs.

    Args:
        details_path: Path to the CSV file containing join details.

    Returns:
        Dictionary mapping join count to a list of question IDs.
    """
    join_buckets = {}
    with open(details_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            join_buckets.setdefault(int(row["num_edges"]), []).append(int(row["id"]))
    return join_buckets


def din_accuracy_by_joins() -> Dict[int, float]:
    """
    Computes DIN-SQL execution accuracy grouped by number of JOINs.

    Returns:
        Dictionary mapping join count to average execution accuracy.
    """
    base = Path(os.getenv("DATA_FOLDER"))
    details_path = (
        base / "experiments/augmentation/experiment_inputs/augmented_join_details.csv"
    )
    pruned = json.load(
        open(
            base / "experiments/augmentation/query_first/bird_qf_filtered.json",
            encoding="utf-8",
        )
    )
    discarded = json.load(
        open(
            base / "experiments/augmentation/query_first/bird_qf_discarded.json",
            encoding="utf-8",
        )
    )
    aug_data = pruned + discarded
    buckets = load_join_buckets(details_path)

    logs = [
        base / "experiments/DIN-SQL/BIRD_sampled_aug/logs.csv",
    ]

    results = []
    for file in logs:
        with open(file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = next(
                    (
                        q["question_id"]
                        for q in aug_data
                        if q["SQL"].strip().lower() == row["gold_query"].strip().lower()
                    ),
                    None,
                )
                if qid is None:
                    continue
                db_path = (
                    base
                    / f"benchmarks/Bird/bird_databases/{row['db_id']}/{row['db_id']}.sqlite"
                )
                try:
                    acc = int(
                        compute_exec_accuracy(
                            row["final_query"], row["gold_query"], db_path
                        )
                    )
                    results.append({"id": int(qid), "ex": acc})
                except Exception:
                    continue

    grouped = {}
    for r in results:
        join_count = next((jc for jc, ids in buckets.items() if r["id"] in ids), None)
        if join_count is not None:
            grouped.setdefault(join_count, []).append(r["ex"])

    return {jc: np.mean(accs) for jc, accs in grouped.items()}


def chess_accuracy_by_joins() -> Dict[int, float]:
    """
    Computes CHESS execution accuracy grouped by number of JOINs.

    Returns:
        Dictionary mapping join count to average execution accuracy.
    """
    base = Path(os.getenv("DATA_FOLDER"))
    details_path = (
        base / "experiments/augmentation/experiment_inputs/augmented_join_details.csv"
    )
    results_dir = base / "experiments/CHESS/BIRD_sampled_aug"
    buckets = load_join_buckets(details_path)

    def parse_filename(name: str):
        parts = name.replace(".json", "").split("_")
        return int(parts[0]), parts[1]

    results = []
    for file in results_dir.glob("*.json"):
        qid, _ = parse_filename(file.name)
        with open(file, "r", encoding="utf-8") as f:
            for d in json.load(f):
                if d.get("tool_name") == "execution_accuracy":
                    acc = d.get("final_SQL", {}).get("exec_res", "")
                    if acc != "error":
                        results.append({"id": qid, "ex": int(acc)})

    grouped = {}
    for r in results:
        join_count = next((jc for jc, ids in buckets.items() if r["id"] in ids), None)
        if join_count is not None:
            grouped.setdefault(join_count, []).append(r["ex"])

    return {jc: np.mean(accs) for jc, accs in grouped.items()}


def mac_accuracy_by_joins() -> Dict[int, float]:
    """
    Computes MAC-SQL execution accuracy grouped by number of JOINs.

    Returns:
        Dictionary mapping join count to average execution accuracy.
    """
    base = Path(os.getenv("DATA_FOLDER"))
    details_path = (
        base / "experiments/augmentation/experiment_inputs/augmented_join_details.csv"
    )
    mac_results = json.load(
        open(
            base / "experiments/MAC-SQL/BIRD_sampled_aug/output_bird.json",
            encoding="utf-8",
        )
    )
    pruned = json.load(
        open(
            base / "experiments/augmentation/query_first/bird_qf_filtered.json",
            encoding="utf-8",
        )
    )
    discarded = json.load(
        open(
            base / "experiments/augmentation/query_first/bird_qf_discarded.json",
            encoding="utf-8",
        )
    )
    buckets = load_join_buckets(details_path)

    aug_lookup = {
        q["SQL"].strip().lower(): q["question_id"] for q in pruned + discarded
    }

    results = []
    for r in mac_results:
        qid = aug_lookup.get(r["ground_truth"].strip().lower())
        if qid is None:
            continue
        db_path = (
            base / f"benchmarks/Bird/bird_databases/{r['db_id']}/{r['db_id']}.sqlite"
        )
        try:
            acc = int(compute_exec_accuracy(r["final_sql"], r["ground_truth"], db_path))
            results.append({"id": qid, "ex": acc})
        except Exception:
            continue

    grouped = {}
    for r in results:
        join_count = next((jc for jc, ids in buckets.items() if r["id"] in ids), None)
        if join_count is not None:
            grouped.setdefault(join_count, []).append(r["ex"])

    return {jc: np.mean(accs) for jc, accs in grouped.items()}

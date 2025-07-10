from pathlib import Path
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from src.core.logger.logger import Logger
from src.core.database.database_handler import DBMS
from src.evaluation import Evaluation, EvaluationTechnique
from src.core.model_manager import OpenAIModel

logger = Logger(name=__name__)


def calculate_execution_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> float:
    config = {
        "evaluation_technique": EvaluationTechnique.EXECUTION_ACCURACY,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(db_path),
        },
        "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
        "logs_dir_path": "data/evaluation_outputs/",
    }
    exact_evaluator = Evaluation(config)
    res = exact_evaluator.run_evaluation(
        predicted_sql=pred_sql,
        ground_truth_sql=gold_sql,
        log=False,
    )
    return res["metrics"]["EX"]


def get_join_buckets(join_details_path: Path) -> Dict[int, List[int]]:
    join_buckets = {}
    with open(join_details_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            join_buckets.setdefault(int(row["num_edges"]), []).append(int(row["id"]))
    return join_buckets


def get_din_results_by_joins() -> Dict[int, float]:
    base_path = Path(os.getenv("DATA_FOLDER"))
    join_details_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "augmented_join_details.csv"
    )
    pruned_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_filtered.json"
    )
    discarded_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_discarded.json"
    )
    result_files = [
        base_path / "experiments" / "DIN-SQL" / "BIRD_50_aug" / "logs-2.csv",
        base_path / "experiments" / "DIN-SQL" / "BIRD_50_aug" / "logs-3.csv",
    ]

    with open(pruned_path, "r", encoding="utf-8") as f:
        pruned = json.load(f)
    with open(discarded_path, "r", encoding="utf-8") as f:
        discarded = json.load(f)
    aug_data = pruned + discarded
    join_buckets = get_join_buckets(join_details_path)

    results = []
    for file in result_files:
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
                    base_path
                    / "benchmarks"
                    / "Bird"
                    / "bird_databases"
                    / f"{row['db_id']}/{row['db_id']}.sqlite"
                )
                try:
                    acc = int(
                        calculate_execution_accuracy(
                            row["final_query"], row["gold_query"], db_path
                        )
                    )
                    results.append({"id": int(qid), "ex": acc})
                except Exception:
                    continue

    results_by_joins = {}
    for res in results:
        join_count = next(
            (jc for jc, ids in join_buckets.items() if res["id"] in ids), None
        )
        if join_count is not None:
            results_by_joins.setdefault(join_count, []).append(res["ex"])

    return {jc: np.mean(accs) for jc, accs in results_by_joins.items()}


def get_chess_results_by_joins() -> Dict[int, float]:
    base_path = Path(os.getenv("DATA_FOLDER"))
    results_path = base_path / "experiments" / "CHESS" / "BIRD_50_aug"
    join_details_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "augmented_join_details.csv"
    )
    join_buckets = get_join_buckets(join_details_path)

    def parse_id_and_db_id(name: str):
        parts = name.replace(".json", "").split("_")
        return int(parts[0]), parts[1]

    results = []
    for file in results_path.glob("*.json"):
        id_, db_id = parse_id_and_db_id(file.name)
        with open(file, "r", encoding="utf-8") as f:
            for d in json.load(f):
                if d.get("tool_name") == "execution_accuracy":
                    acc = d.get("final_SQL", {}).get("exec_res", "")
                    if acc != "error":
                        results.append({"id": id_, "ex": int(acc)})

    results_by_joins = {}
    for res in results:
        join_count = next(
            (jc for jc, ids in join_buckets.items() if res["id"] in ids), None
        )
        if join_count is not None:
            results_by_joins.setdefault(join_count, []).append(res["ex"])

    return {jc: np.mean(accs) for jc, accs in results_by_joins.items()}


def get_mac_results_by_joins() -> Dict[int, float]:
    base_path = Path(os.getenv("DATA_FOLDER"))
    joins_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "augmented_join_details.csv"
    )
    mac_path = (
        base_path / "experiments" / "MAC-SQL" / "BIRD_50_aug" / "output_bird.json"
    )
    pruned_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_filtered.json"
    )
    discarded_path = (
        base_path
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_discarded.json"
    )

    with open(mac_path, "r", encoding="utf-8") as f:
        mac_results = json.load(f)
    with open(pruned_path, "r", encoding="utf-8") as f:
        pruned = json.load(f)
    with open(discarded_path, "r", encoding="utf-8") as f:
        discarded = json.load(f)
    aug_lookup = {
        q["SQL"].strip().lower(): q["question_id"] for q in pruned + discarded
    }
    join_buckets = get_join_buckets(joins_path)

    results = []
    for r in mac_results:
        qid = aug_lookup.get(r["ground_truth"].strip().lower())
        if qid is None:
            continue
        db_path = (
            base_path
            / "benchmarks"
            / "Bird"
            / "bird_databases"
            / f"{r['db_id']}/{r['db_id']}.sqlite"
        )
        try:
            acc = int(
                calculate_execution_accuracy(r["final_sql"], r["ground_truth"], db_path)
            )
            results.append({"id": qid, "ex": acc})
        except Exception:
            continue

    results_by_joins = {}
    for res in results:
        join_count = next(
            (jc for jc, ids in join_buckets.items() if res["id"] in ids), None
        )
        if join_count is not None:
            results_by_joins.setdefault(join_count, []).append(res["ex"])

    return {jc: np.mean(accs) for jc, accs in results_by_joins.items()}


def plot_execution_accuracy_by_joins():
    din = get_din_results_by_joins()
    chess = get_chess_results_by_joins()
    mac = get_mac_results_by_joins()

    print("====== EXECUTION ACCURACY BY JOINS ======")
    # I want the results in % and two numbers after the decimal point
    din = {k: round(v * 100, 2) for k, v in din.items()}
    chess = {k: round(v * 100, 2) for k, v in chess.items()}
    mac = {k: round(v * 100, 2) for k, v in mac.items()}
    logger.log("info", "DIN-SQL execution accuracy by joins: %s", din)
    logger.log("info", "CHESS execution accuracy by joins: %s", chess)
    logger.log("info", "MAC-SQL execution accuracy by joins: %s", mac)

    plt.figure(figsize=(10, 6))
    for label, data in [("DIN-SQL", din), ("CHESS", chess), ("MAC-SQL", mac)]:
        x = sorted(data)
        y = [data[j] for j in x]
        plt.plot(x, y, marker="o", label=label)

    plt.title("Execution Accuracy by Join Count")
    plt.xlabel("Number of JOINs")
    plt.ylabel("Execution Accuracy (%)")
    plt.xticks(sorted(set(din) | set(chess) | set(mac)))
    plt.grid(True)
    plt.legend()

    output_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "execution_accuracy_by_joins.png"
    )
    plt.savefig(output_path)
    logger.log("info", f"Combined execution accuracy plot saved to {output_path}")


def main():
    plot_execution_accuracy_by_joins()
    logger.log("info", "Execution accuracy by joins plot generated successfully.")


if __name__ == "__main__":
    main()

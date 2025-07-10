import os
import json
import csv
import re
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path
from typing import List, Dict
from collections import Counter
from src.core.logger.logger import Logger
from src.core.database.database_handler import DBMS
from src.evaluation import Evaluation, EvaluationTechnique
from src.core.model_manager import OpenAIModel

logger = Logger(name=__name__)


def plot_din_execution_by_joins():
    results_path_1 = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "DIN-SQL"
        / "BIRD_50_aug"
        / "logs-2.csv"
    )
    results_path_2 = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "DIN-SQL"
        / "BIRD_50_aug"
        / "logs-3.csv"
    )
    join_details_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "augmented_join_details.csv"
    )
    pruned_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_filtered.json"
    )
    discarded_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_discarded.json"
    )
    # Load pruned and discarded queries
    with open(pruned_path, "r", encoding="utf-8") as f:
        pruned_data = json.load(f)
    with open(discarded_path, "r", encoding="utf-8") as f:
        discarded_data = json.load(f)
    aug_data = pruned_data + discarded_data

    output_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "din_results_by_joins.png"
    )

    # Read results from both files
    results = []
    for file in [results_path_1, results_path_2]:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = {
                    "id": int(
                        next(
                            (
                                q["question_id"]
                                for q in aug_data
                                if q["SQL"].strip().lower()
                                == row["gold_query"].strip().lower()
                            ),
                            None,
                        )
                    ),
                    "db_id": row["db_id"],
                    "ex": int(
                        calculate_execution_accuracy(
                            row["final_query"].strip(),
                            row["gold_query"].strip(),
                            Path(os.getenv("DATA_FOLDER"))
                            / "benchmarks"
                            / "Bird"
                            / "bird_databases"
                            / f"{row['db_id']}/{row['db_id']}.sqlite",
                        )
                    ),
                }
                results.append(result)

    # Group results by join count
    join_buckets: Dict[int, List[int]] = {}
    with open(join_details_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            join_buckets.setdefault(int(row["num_edges"]), []).append(int(row["id"]))

    results_by_joins: Dict[int, List[int]] = {}
    for res in results:
        join_count = next(
            (jc for jc, ids in join_buckets.items() if res["id"] in ids), None
        )
        if join_count is not None:
            results_by_joins.setdefault(join_count, []).append(res["ex"])
    # Compute mean execution accuracy
    mean_results = {
        jc: np.mean([ex for ex in accs if ex != "error"])
        for jc, accs in results_by_joins.items()
        if accs
    }

    logger.log("info", f"Mean execution accuracy by join count: {mean_results}")
    # Plot
    plt.figure(figsize=(10, 6))
    x, y = sorted(mean_results), [mean_results[j] for j in sorted(mean_results)]
    plt.plot(x, y, marker="o", label="DIN-SQL Execution Accuracy")
    plt.title("DIN-SQL Execution Accuracy by Join Count")
    plt.xlabel("Number of JOINs")
    plt.ylabel("Execution Accuracy (%)")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    logger.log("info", f"DIN-SQL results by joins saved to {output_path}")


def plot_chess_execution_by_joins():
    results_path = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "CHESS" / "BIRD_50_aug"
    )
    join_details = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "augmented_join_details.csv"
    )
    output_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
        / "execution_accuracy_by_joins.png"
    )
    # I wanna read each file under results_path:
    results = []
    for file in results_path.glob("*.json"):
        id, db_id = parse_id_and_db_id(file.name)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                if d.get("tool_name") == "execution_accuracy":
                    final_sql = d.get("final_SQL", {})
                    result = {
                        "id": id,
                        "db_id": db_id,
                        "ex": final_sql.get("exec_res", ""),
                    }
                    results.append(result)
    # Group results by join count
    join_buckets: Dict[int, List[int]] = {}
    with open(join_details, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            join_buckets.setdefault(int(row["num_edges"]), []).append(int(row["id"]))
    results_by_joins: Dict[int, List[int]] = {}
    for res in results:
        join_count = next(
            (jc for jc, ids in join_buckets.items() if res["id"] in ids), None
        )
        if join_count is not None:
            results_by_joins.setdefault(join_count, []).append(res["ex"])
    # Compute mean execution accuracy
    mean_results = {
        jc: np.mean([int(ex) for ex in accs if ex != "error"])
        for jc, accs in results_by_joins.items()
        if accs
    }
    logger.log("info", f"Mean execution accuracy by join count: {mean_results}")
    # Plot
    plt.figure(figsize=(10, 6))
    x, y = sorted(mean_results), [mean_results[j] for j in sorted(mean_results)]
    plt.plot(x, y, marker="o", label="CHESS Execution Accuracy")
    plt.title("CHESS Execution Accuracy by Join Count")
    plt.xlabel("Number of JOINs")
    plt.ylabel("Execution Accuracy (%)")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    logger.log("info", f"CHESS results by joins saved to {output_path}")


def plot_mac_execution_by_joins():
    base_path = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation"
    joins_path = base_path / "experiment_inputs" / "augmented_join_details.csv"
    mac_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "MAC-SQL"
        / "BIRD_50_aug"
        / "output_bird.json"
    )
    pruned_path = base_path / "query_first" / "bird_qf_filtered.json"
    discarded_path = base_path / "query_first" / "bird_qf_discarded.json"

    # Load data
    with open(mac_path, "r", encoding="utf-8") as f:
        mac_results = json.load(f)
    with open(pruned_path, "r", encoding="utf-8") as f:
        pruned = json.load(f)
    with open(discarded_path, "r", encoding="utf-8") as f:
        discarded = json.load(f)
    aug_data = pruned + discarded
    aug_lookup = {q["SQL"].strip().lower(): q["question_id"] for q in aug_data}

    # Load join groupings
    join_buckets: Dict[int, List[int]] = {}
    with open(joins_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            join_buckets.setdefault(int(row["num_edges"]), []).append(int(row["id"]))

    # Build result list with execution accuracy
    results = []
    for r in mac_results:
        gold_sql = r["ground_truth"].strip().lower()
        qid = aug_lookup.get(gold_sql)
        if qid is None:
            continue
        db_path = (
            Path(os.getenv("DATA_FOLDER"))
            / "benchmarks"
            / "Bird"
            / "bird_databases"
            / f"{r['db_id']}/{r['db_id']}.sqlite"
        )
        try:
            acc = int(
                calculate_execution_accuracy(
                    r["final_sql"].strip().replace("\n", " "),
                    r["ground_truth"].strip(),
                    db_path,
                )
            )
            results.append({"id": qid, "ex": acc})
        except Exception:
            continue

    # Group results by join count
    results_by_joins: Dict[int, List[int]] = {}
    for res in results:
        join_count = next(
            (jc for jc, ids in join_buckets.items() if res["id"] in ids), None
        )
        if join_count is not None:
            results_by_joins.setdefault(join_count, []).append(res["ex"])

    # Compute mean execution accuracy
    mean_results = {jc: np.mean(accs) for jc, accs in results_by_joins.items() if accs}
    logger.log("info", f"Mean execution accuracy by join count: {mean_results}")

    # Plot
    plt.figure(figsize=(10, 6))
    x, y = sorted(mean_results), [mean_results[j] for j in sorted(mean_results)]
    plt.plot(x, y, marker="o", label="MAC-SQL Execution Accuracy")
    plt.title("MAC-SQL Execution Accuracy by Join Count")
    plt.xlabel("Number of JOINs")
    plt.ylabel("Execution Accuracy (%)")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()

    output_path = base_path / "experiment_inputs" / "mac_results_by_joins.png"
    plt.savefig(output_path)
    logger.log("info", f"MAC-SQL results by joins saved to {output_path}")


def sample_aug_queries_by_joins():
    base_path = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation"
    join_details_path = base_path / "experiment_inputs" / "augmented_join_details.csv"
    pruned_path = base_path / "query_first" / "bird_qf_filtered.json"
    discarded_path = base_path / "query_first" / "bird_qf_discarded.json"

    # Load augmented data
    with open(pruned_path, "r", encoding="utf-8") as f:
        pruned_data = json.load(f)
    with open(discarded_path, "r", encoding="utf-8") as f:
        discarded_data = json.load(f)
    aug_data = pruned_data + discarded_data

    # Map question_id to full query for fast lookup
    aug_data_lookup = {q["question_id"]: q for q in aug_data}

    # Group query IDs by join count
    join_buckets: Dict[int, List[int]] = {}
    with open(join_details_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            join_count = int(row["num_edges"])
            qid = int(row["id"])
            join_buckets.setdefault(join_count, []).append(qid)

    selected_queries = []

    for join_count in range(1, 6):
        ids = join_buckets.get(join_count, [])
        if len(ids) > 50:
            sampled_ids = np.random.choice(ids, size=50, replace=False)
            for qid in sampled_ids:
                query = aug_data_lookup.get(qid)
                if query:
                    selected_queries.append(query)

    # Save JSON
    json_output_path = base_path / "experiment_inputs" / "aug_queries_per_joins.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(selected_queries, f, indent=4, ensure_ascii=False)

    # Save SQL file with db_id tab-separated
    sql_output_path = base_path / "experiment_inputs" / "aug_queries_per_joins.sql"
    with open(sql_output_path, "w", encoding="utf-8") as f:
        for q in selected_queries:
            if q.get("SQL") and q.get("db_id"):
                f.write(f"{q['SQL']}\t{q['db_id']}\n")


def ex_per_joins(system_names):
    input_folder = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
    )

    def get_file(system, name):
        return input_folder / f"{system}_{name}_results.csv"

    def load_edges(system, which):
        file_path = input_folder / f"{which}_aug_characteristics.csv"
        with open(file_path, "r", encoding="utf-8") as f:
            return {row["id"]: row["num_edges"] for row in csv.DictReader(f)}

    def group_by_joins(result_file, id_to_edges):
        by_joins = {}
        with open(result_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                join_count = id_to_edges.get(row["id"])
                if join_count is None:
                    continue
                by_joins.setdefault(join_count, []).append(
                    {"id": row["id"], "ex": row["ex"]}
                )
        return by_joins

    def safe_mean(data):
        values = [float(d["ex"]) for d in data if d["ex"] != "error"]
        return np.mean(values) if values else 0.0

    def plot_comparison(data_dict, title, output_file):
        plt.figure(figsize=(10, 6))
        for system, data in data_dict.items():
            keys = sorted(data.keys(), key=int)
            acc = [safe_mean(data[k]) for k in keys]
            print(f"[{system}] Keys: {keys}")
            print(f"[{system}] Accuracies: {acc}")
            plt.plot(keys, acc, marker="o", label=system)
        plt.title(title)
        plt.xlabel("Number of JOINs")
        plt.ylabel("Execution Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.savefig(output_file)
        plt.close()

    for mode in ["dev", "aug"]:
        results_by_system = {}
        for system in system_names:
            edge_file_type = "original" if mode == "dev" else "filtered"
            edges = load_edges(system, edge_file_type)
            grouped = group_by_joins(get_file(system, mode), edges)
            results_by_system[system] = grouped

        output_file = input_folder / f"{'_'.join(system_names)}_{mode}_ex_per_joins.png"
        title = f"Execution Accuracy per Number of JOINs ({mode.capitalize()})"
        plot_comparison(results_by_system, title, output_file)
        logger.log("info", f"{title} plot saved to {output_file}")


def join_details(pruned_aug: List[Dict], generated_aug: List[Dict]) -> None:
    base = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "query_first"
    )
    pruned_aug_data = json.load(open(base / "bird_qf_filtered.json", encoding="utf-8"))
    generated_aug_data = json.load(
        open(base / "bird_qf_discarded.json", encoding="utf-8")
    )
    rule_input_base = Path(os.getenv("RULE_INPUTS_BASE"))

    def stats(g, qid, dbid, query):
        n = g.number_of_nodes()
        d = list(dict(g.degree()).values())
        c = nx.cycle_basis(g)
        join_count = len(re.findall(r"\bJOIN\b", query, flags=re.IGNORECASE))

        return {
            "id": qid,
            "db_id": dbid,
            "num_nodes": n,
            "num_edges": g.number_of_edges(),
            "num_joins": join_count,
            "avg_degree": sum(d) / n if n else 0,
            "is_cyclic": bool(c),
            "num_cycles": len(c),
        }

    def get_id(q, data):
        return next(i["question_id"] for i in data if i["SQL"] == q["query_first"])

    pruned_details = [
        stats(
            q["ext_jq_graph"], get_id(q, pruned_aug_data), q["db_id"], q["query_first"]
        )
        for q in pruned_aug
    ]
    generated_details = [
        stats(
            q["ext_jq_graph"],
            get_id(q, generated_aug_data),
            q["db_id"],
            q["query_first"],
        )
        for q in generated_aug
    ]
    # Combine pruned and generated into one list
    augmented_details = pruned_details + generated_details

    original_details = []
    for folder in rule_input_base.iterdir():
        for file in folder.glob("*.pkl") if folder.is_dir() else []:
            for d in pickle.load(open(file, "rb")):
                query = d.get("flattened_query") or d.get("SQL")
                original_details.append(
                    stats(d["jq_graph"], d["question_id"], d["db_id"], query)
                )

    output_dir = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
    )
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
    ]

    # Function to log join count distribution
    def log_distribution(name, data):
        join_dist = {}
        for d in data:
            j = d["num_joins"]
            join_dist[j] = join_dist.get(j, 0) + 1
        logger.log(
            "info", f"Join distribution for {name}: {dict(sorted(join_dist.items()))}"
        )

    def log_distribution_for_number_edges(name, data):
        edge_dist = {}
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
        # log_distribution(name, data)
        log_distribution_for_number_edges(name, data)


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


def turn_results_into_csv(results: List[Dict], system_name: str, mode: str) -> None:
    output_dir = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / f"{system_name}_{mode}_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "db_id",
                "question",
                "evidence",
                "gold_SQL",
                "pred_SQL",
                "ex",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["id"]))  # sort by id ascending

    print(f"Results saved to {csv_file}")
    logger.log("info", f"Results for {system_name} in {mode} mode saved to {csv_file}")


def parse_id_and_db_id(filename: str) -> (int, str):
    """
    Extract leading integer ID and rest as db_id.
    Example: '123_sports_complex.json' -> (123, 'sports_complex')
    """
    match = re.match(r"^(\d+)_([^.]+)", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected format.")
    return int(match.group(1)), match.group(2)


def read_chess_results(system_name: str, mode: str) -> None:
    base_path = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / system_name / f"BIRD_{mode}"
    )
    results = []

    for file in sorted(base_path.glob("*.json")):
        try:
            file_id, db_id = parse_id_and_db_id(file.name)
            result = {
                "id": file_id,
                "db_id": db_id,
                "question": "",
                "evidence": "",
                "gold_SQL": "",
                "pred_SQL": "",
                "ex": "",
            }

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data:
                    if d.get("tool_name") == "execution_accuracy":
                        final_sql = d.get("final_SQL", {})
                        result["question"] = final_sql.get("Question", "")
                        result["evidence"] = final_sql.get("Evidence", "")
                        result["gold_SQL"] = final_sql.get("GOLD_SQL", "").strip()
                        result["pred_SQL"] = final_sql.get("PREDICTED_SQL", "").strip()
                        result["ex"] = final_sql.get("exec_res", "")
                        break

            results.append(result)
        except Exception as e:
            logger.log("error", f"Error reading {file.name}: {e}")

    turn_results_into_csv(results, system_name, mode)


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


def read_din_results(system_name: str, mode: str) -> None:
    data_folder = Path(os.getenv("DATA_FOLDER"))
    results_path = (
        data_folder / "experiments" / system_name / f"BIRD_{mode}" / "logs.csv"
    )
    db_path = data_folder / "benchmarks" / "Bird" / "bird_databases"

    data_file = {
        "dev": data_folder / "benchmarks" / "Bird" / "bird_dev.json",
        "aug": data_folder
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_filtered.json",
    }.get(mode)

    if not data_file:
        logger.log("error", f"Invalid mode: {mode}")
        return

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(results_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results = []

            for row in reader:
                gold_sql = row["gold_query"].strip().lower()
                match = next(
                    (d for d in data if d["SQL"].strip().lower() == gold_sql), None
                )
                if not match:
                    continue

                result = {
                    "id": match["question_id"],
                    "db_id": row["db_id"],
                    "question": match["question"],
                    "evidence": match["evidence"],
                    "gold_SQL": row["gold_query"].strip(),
                    "pred_SQL": row["final_query"].strip(),
                }

                result["ex"] = str(
                    calculate_execution_accuracy(
                        result["pred_SQL"],
                        result["gold_SQL"],
                        db_path / f"{result['db_id']}/{result['db_id']}.sqlite",
                    )
                )

                results.append(result)

    except Exception as e:
        logger.log(
            "error",
            f"Error reading files for system '{system_name}' in mode '{mode}': {e}",
        )
        return

    turn_results_into_csv(results, system_name, mode)


def read_mac_results(system_name: str, mode: str) -> None:
    data_folder = Path(os.getenv("DATA_FOLDER"))
    base_path = (
        data_folder / "experiments" / "augmentation" / system_name / f"BIRD_{mode}"
    )
    results_path = base_path / (
        "output_bird.json" if mode == "dev" else "eval_result_dev.json"
    )

    results = []

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if mode == "aug":
            aug_set_path = (
                data_folder
                / "experiments"
                / "augmentation"
                / "query_first"
                / "bird_qf_filtered.json"
            )
            with open(aug_set_path, "r", encoding="utf-8") as f:
                aug_data = json.load(f)

            for row in data:
                gold = row["gold"].strip().lower()
                qid = next(
                    (
                        d["question_id"]
                        for d in aug_data
                        if d["SQL"].strip().lower() == gold
                    ),
                    None,
                )
                results.append(
                    {
                        "id": qid,
                        "db_id": row["db_id"],
                        "question": row["question"],
                        "evidence": row["evidence"],
                        "gold_SQL": row["gold"].strip(),
                        "pred_SQL": row["pred"].strip(),
                        "ex": row["res"],
                    }
                )

        else:  # dev mode
            for row in data:
                gold_sql = row["ground_truth"].strip()
                pred_sql = row["final_sql"].strip().replace("\n", " ")
                db_path = (
                    data_folder
                    / "benchmarks"
                    / "Bird"
                    / "bird_databases"
                    / f"{row['db_id']}/{row['db_id']}.sqlite"
                )
                ex_score = str(
                    calculate_execution_accuracy(pred_sql, gold_sql, db_path)
                )

                results.append(
                    {
                        "id": row["idx"],
                        "db_id": row["db_id"],
                        "question": row["query"],
                        "evidence": row["evidence"],
                        "gold_SQL": gold_sql,
                        "pred_SQL": pred_sql,
                        "ex": ex_score,
                    }
                )

    except Exception as e:
        logger.log(
            "error",
            f"Error reading files for system '{system_name}' in mode '{mode}': {e}",
        )
        return

    turn_results_into_csv(results, system_name, mode)


def calculate_delta_ex(system_name: str) -> None:
    results_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "experiment_inputs"
    )

    # File paths
    aug_file = results_path / f"{system_name}_aug_results.csv"
    dev_file = results_path / f"{system_name}_dev_results.csv"
    output_file = results_path / f"{system_name}_delta_ex_results.csv"

    # Read dev results into a dictionary indexed by extracted ext_id
    with open(dev_file, "r", encoding="utf-8") as f_dev:
        dev_reader = csv.DictReader(f_dev)
        dev_lookup: Dict[str, Dict] = {int(row["id"]): row for row in dev_reader}

    delta_ex_results = []

    with open(aug_file, "r", encoding="utf-8") as f_aug:
        aug_reader = csv.DictReader(f_aug)

        for row in aug_reader:
            ext_id = int(str(row["id"])[2:6])
            dev_row = dev_lookup.get(ext_id)

            if dev_row:
                try:
                    delta_ex = float(row["ex"]) - float(dev_row["ex"])
                except ValueError:
                    delta_ex = None  # fallback if values are not convertible

                delta_ex_results.append(
                    {
                        "id": row["id"],
                        "ext_id": ext_id,
                        "db_id": row["db_id"],
                        "aug_gold": row["gold_SQL"],
                        "aug_pred": row["pred_SQL"],
                        "aug_ex": row["ex"],
                        "dev_gold": dev_row["gold_SQL"],
                        "dev_pred": dev_row["pred_SQL"],
                        "dev_ex": dev_row["ex"],
                        "delta_ex": delta_ex,
                    }
                )

    # Write output CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "id",
                "ext_id",
                "db_id",
                "aug_gold",
                "aug_pred",
                "aug_ex",
                "dev_gold",
                "dev_pred",
                "dev_ex",
                "delta_ex",
            ],
        )
        writer.writeheader()
        writer.writerows(delta_ex_results)

    delta_counts = Counter(
        int(res["delta_ex"]) for res in delta_ex_results if res["delta_ex"] is not None
    )

    logger.log("info", f"Delta Execution Accuracy Summary for {system_name}:")

    logger.log(
        "info",
        f"delta_ex =  1 --> {int(delta_counts.get(1, 0)) / len(delta_ex_results) * 100:.2f}%",
    )
    logger.log(
        "info",
        f"delta_ex = -1 --> {int(delta_counts.get(-1, 0)) / len(delta_ex_results) * 100:.2f}%",
    )
    logger.log(
        "info",
        f"delta_ex =  0 --> {int(delta_counts.get(0, 0)) / len(delta_ex_results) * 100:.2f}%",
    )

    # Percentage of aug_ex == 1 (still on subset)
    aug_ex_count = sum(1 for res in delta_ex_results if res["aug_ex"] == "1")

    # Percentage of dev_ex == 1 (on full dev set)
    total_dev_ex_count = sum(1 for row in dev_lookup.values() if row["ex"] == "1")

    logger.log(
        "info",
        f"Percentage of aug_ex --> 1: {aug_ex_count / len(delta_ex_results) * 100:.2f}%",
    )

    logger.log(
        "info",
        f"Percentage of dev_ex --> 1 (full dev set): {total_dev_ex_count / len(dev_lookup) * 100:.2f}%",
    )

    # Extract unique dev queries by ext_id
    unique_dev_ext_ids = {res["ext_id"] for res in delta_ex_results}

    # Filter dev_lookup to only include those unique queries
    unique_dev_rows = [dev_lookup[ext_id] for ext_id in unique_dev_ext_ids]

    # Count how many of those had ex == 1
    unique_dev_ex_1_count = sum(1 for row in unique_dev_rows if row["ex"] == "1")

    # Log the result
    logger.log("info", f"Unique dev queries involved: {len(unique_dev_rows)}")
    logger.log(
        "info",
        f"Percentage of dev_ex --> 1 (on unique {len(unique_dev_rows)} dev queries): {unique_dev_ex_1_count / len(unique_dev_rows) * 100:.2f}%",
    )

    # Let's count delta_ex values = 0 knowing that dev_ex and aug_ex are both 0

    zero_count = sum(
        1
        for res in delta_ex_results
        if res["delta_ex"] == 0 and res["dev_ex"] == "0" and res["aug_ex"] == "0"
    )
    if zero_count > 0:
        logger.log(
            "info",
            f"delta_ex = 0 (both dev_ex and aug_ex are 0) --> {zero_count / len(delta_ex_results) * 100:.2f}%",
        )
    # Now for delta_ex = 0 but both dev_ex and aug_ex are 1
    one_count = sum(
        1
        for res in delta_ex_results
        if res["delta_ex"] == 0 and res["dev_ex"] == "1" and res["aug_ex"] == "1"
    )
    if one_count > 0:
        logger.log(
            "info",
            f"delta_ex = 0 (both dev_ex and aug_ex are 1) --> {one_count / len(delta_ex_results) * 100:.2f}%",
        )
        print()
        print()


def main() -> None:
    # for mode in ["aug"]:
    #     for system in ["MAC-SQL"]:
    #         read_mac_results(system, mode)

    #     for system in ["CHESS", "DIN-SQL"]:
    #         if system == "CHESS":
    #             read_chess_results(system, mode)
    #         elif system == "DIN-SQL":
    #             read_din_results(system, mode)

    # for system in ["CHESS", "DIN-SQL", "MAC-SQL"]:
    #     calculate_delta_ex(system)

    # filtered_aug, discarded_aug = read_queries_w_jqg()
    # for fa in filtered_aug:
    #     is_connected = nx.is_connected(fa["ext_jq_graph"])
    #     if is_connected == False:
    #         print("Filtered graph is not connected for question_id:", fa["question_id"])

    # for da in discarded_aug:
    #     is_connected = nx.is_connected(da["ext_jq_graph"])
    #     if is_connected == False:
    #         print("Discarded graph is not connected for question_id:", da["question_id"])
    # jq_graph_characteristics(filtered_aug, discarded_aug)
    # ex_per_joins(["CHESS", "DIN-SQL"])
    # pruned_aug, generated_aug = read_queries_w_jqg()
    # join_details(pruned_aug, generated_aug)
    # sample_aug_queries_by_joins()
    # plot_mac_execution_by_joins()
    # plot_chess_execution_by_joins()
    plot_din_execution_by_joins()


if __name__ == "__main__":
    main()

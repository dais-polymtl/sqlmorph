import os
import re
import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from src.core.logger.logger import Logger
from src.experiments.augmentation.utils import compute_exec_accuracy

logger = Logger(name=__name__)


def calculate_delta_ex(system_name, technique) -> None:
    # We will read results from the csv files, compute the delta execution accuracy, and add it to the file with the technique name, ok
    technique_file = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "lt_outputs"
        / f"{system_name}_test_{technique}_results.csv"
    )
    original_file = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / "augmentation"
        / "lt_outputs"
        / f"{system_name}_test_ori_results.csv"
    )

    # Read the original results
    original_results = []
    with original_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_results.append(
                {
                    "id": int(row["id"]),
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "evidence": row["evidence"],
                    "gold_SQL": row["gold_SQL"],
                    "pred_SQL": row["pred_SQL"],
                    "ex": int(row["ex"]),
                }
            )

    # Read the technique results
    technique_results = []
    with technique_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            technique_results.append(
                {
                    "id": int(row["id"]),
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "evidence": row["evidence"],
                    "gold_SQL": row["gold_SQL"],
                    "pred_SQL": row["pred_SQL"],
                    "ex": int(row["ex"]),
                }
            )

    # We will compute the delta execution accuracy, print it and add it in the technique file
    original_index = {(item["id"], item["db_id"]): item for item in original_results}
    delta_results = []

    for tech in technique_results:
        key = (tech["id"], tech["db_id"])
        if key in original_index:
            original = original_index[key]
            delta_ex = tech["ex"] - original["ex"]
            delta_results.append(
                {
                    "id": tech["id"],
                    "db_id": tech["db_id"],
                    "question": tech["question"],
                    "evidence": tech["evidence"],
                    "gold_SQL": tech["gold_SQL"],
                    "pred_SQL": tech["pred_SQL"],
                    "ex": tech["ex"],
                    "delta_ex": delta_ex,
                }
            )
        else:
            logger.log(
                "warning",
                f"No match found in original results for id {tech['id']} and db_id {tech['db_id']}",
            )

    # Save the delta results to the technique file
    with technique_file.open("w", newline="", encoding="utf-8") as f:
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
                "delta_ex",
            ],
        )
        writer.writeheader()
        writer.writerows(delta_results)

        # We will print also the percentage of each delta execution accuracy values. (1, 0, -1)
    delta_ex_values = [r["delta_ex"] for r in delta_results]
    total = len(delta_ex_values)
    if total > 0:
        counts = Counter(delta_ex_values)
        percentages = {k: (v / total) * 100 for k, v in counts.items()}
        percentages = {k: round(v, 2) for k, v in percentages.items()}
        logger.log(
            "info",
            f"Delta execution accuracy percentages for {system_name}_{technique}: {percentages}",
        )

    # For delta ex = 0, we need to separate between ex = 1 and ex = 0
    delta_ex_0 = [r for r in delta_results if r["delta_ex"] == 0]
    ex_1_count = sum(1 for r in delta_ex_0 if r["ex"] == 1)
    ex_0_count = sum(1 for r in delta_ex_0 if r["ex"] == 0)
    # percentage of delta ex = 0 with ex = 1 and ex = 0
    if len(delta_ex_0) > 0:
        percentage_ex_1 = (ex_1_count / len(delta_ex_values)) * 100
        percentage_ex_0 = (ex_0_count / len(delta_ex_values)) * 100
        logger.log(
            "info",
            f"Delta execution accuracy = 0: {len(delta_ex_0)} queries, with {percentage_ex_1:.2f}% having ex = 1 and {percentage_ex_0:.2f}% having ex = 0",
        )


def calculate_avg_ex(results: List[Dict]) -> float:
    """
    Calculate the average execution accuracy from a list of results.
    """
    ex_values = [int(r["ex"]) for r in results if "ex" in r]
    if not ex_values:
        return 0.0
    return np.mean(ex_values) * 100


def parse_id_and_db_id(filename: str) -> Tuple[int, str]:
    """
    Extract the integer ID and DB ID from filename.
    Example: '123_sports_complex.json' → (123, 'sports_complex')
    """
    match = re.match(r"^(\d+)_([^.]+)", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected format.")
    return int(match.group(1)), match.group(2)


def turn_results_into_csv(results: List[Dict], system_name: str, tech: str) -> None:
    """
    Save a list of result dictionaries into a CSV file for a given system and mode.
    """
    output_dir = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "lt_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{system_name}_test_{tech}_results.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
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
        writer.writerows(sorted(results, key=lambda r: r["id"]))

    logger.log("info", f"Results for {system_name} in {tech} mode saved to {path}")


def read_chess_results(system_name: str, tech: str) -> None:
    """
    Read CHESS system JSON output and convert into structured CSV.
    """
    results = []
    base_path = (
        Path(os.getenv("DATA_FOLDER"))
        / "experiments"
        / system_name
        / "BIRD_lt"
        / f"test_{tech}"
    )

    for file in sorted(base_path.glob("*.json")):
        try:
            file_id, db_id = parse_id_and_db_id(file.name)
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                if entry.get("tool_name") == "execution_accuracy":
                    final_sql = entry.get("final_SQL", {})
                    results.append(
                        {
                            "id": file_id,
                            "db_id": db_id,
                            "question": final_sql.get("Question", ""),
                            "evidence": final_sql.get("Evidence", ""),
                            "gold_SQL": final_sql.get("GOLD_SQL", "").strip(),
                            "pred_SQL": final_sql.get("PREDICTED_SQL", "").strip(),
                            "ex": (
                                final_sql.get("exec_res", 0)
                                if final_sql.get("exec_res") != "error"
                                else 0
                            ),
                        }
                    )
                    break
        except Exception as e:
            logger.log("error", f"Error reading {file.name}: {e}")

    # We will print the average execution accuracy per system and technique in %

    avg_exec_accuracy = calculate_avg_ex(results)
    logger.log(
        "info",
        f"Average execution accuracy for {system_name} in {tech} mode: {avg_exec_accuracy:.2f}%",
    )

    turn_results_into_csv(results, system_name, tech)


def read_din_results(system_name: str, tech: str) -> None:
    """
    Read DIN-SQL logs and evaluate execution accuracy against ground truth.
    """
    data_folder = Path(os.getenv("DATA_FOLDER"))
    results_path = (
        data_folder
        / "experiments"
        / system_name
        / "BIRD_lt"
        / f"test_{tech}"
        / "logs.csv"
    )
    db_path = data_folder / "benchmarks" / "Bird" / "lt_dbs"

    mode_data_map = {
        "ori": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_syn_rep_original_questions.json",
        "sr": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_syn_rep_queries.json",
        "bt": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_backtrans_queries.json",
        "ca": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_context_aug_queries.json",
    }

    data_file = mode_data_map.get(tech)
    if not data_file:
        logger.log("error", f"Invalid mode: {tech}")
        return

    try:
        with data_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        with results_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results = []

            for row in reader:
                gold_sql = row["gold_query"].strip().lower()
                match = next(
                    (d for d in data if d["SQL"].strip().lower() == gold_sql), None
                )
                if not match:
                    continue

                pred_sql = row["final_query"].strip()
                gold_sql = row["gold_query"].strip()
                db_file = db_path / f"{row['db_id']}/{row['db_id']}.sqlite"

                result = {
                    "id": match["question_id"],
                    "db_id": row["db_id"],
                    "question": match["question"],
                    "evidence": match["evidence"],
                    "gold_SQL": gold_sql,
                    "pred_SQL": pred_sql,
                    "ex": str(compute_exec_accuracy(pred_sql, gold_sql, db_file)),
                }

                results.append(result)
    except Exception as e:
        logger.log("error", f"Failed reading DIN-SQL results: {e}")
        return

    avg_ex_value = calculate_avg_ex(results)
    logger.log(
        "info",
        f"Average execution accuracy for {system_name} in {tech} mode: {avg_ex_value:.2f}%",
    )

    turn_results_into_csv(results, system_name, tech)


def read_mac_results(system_name: str, tech: str) -> None:
    """
    Parse MAC-SQL JSON output (dev or aug) and compute execution accuracy if needed.
    """
    data_folder = Path(os.getenv("DATA_FOLDER"))
    results_file = (
        data_folder
        / "experiments"
        / system_name
        / "BIRD_lt"
        / f"test_{tech}"
        / "output_bird.json"
    )

    results = []
    mode_data_map = {
        "ori": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_syn_rep_original_questions.json",
        "sr": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_syn_rep_queries.json",
        "bt": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_backtrans_queries.json",
        "ca": data_folder
        / "rule_outputs"
        / "lt_elimination"
        / "test"
        / "test_context_aug_queries.json",
    }
    try:

        with results_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        with (mode_data_map.get(tech)).open("r") as f:
            aug_data = json.load(f)

        for row in data:
            qid = next(
                (
                    d["question_id"]
                    for d in aug_data
                    if d["SQL"].strip().lower() == row["ground_truth"].strip().lower()
                ),
                None,
            )
            pred_sql = row["final_sql"].strip().replace("\n", " ")
            gold_sql = row["ground_truth"].strip()
            db_file = (
                data_folder
                / "benchmarks"
                / "Bird"
                / "lt_dbs"
                / f"{row['db_id']}/{row['db_id']}.sqlite"
            )
            results.append(
                {
                    "id": qid,
                    "db_id": row["db_id"],
                    "question": row["query"],
                    "evidence": row["evidence"],
                    "gold_SQL": gold_sql,
                    "pred_SQL": pred_sql,
                    "ex": str(compute_exec_accuracy(pred_sql, gold_sql, db_file)),
                }
            )

    except Exception as e:
        logger.log("error", f"Failed reading MAC-SQL {tech} results: {e}")
        return

    avg_ex_value = calculate_avg_ex(results)
    logger.log(
        "info",
        f"Average execution accuracy for {system_name} in {tech} mode: {avg_ex_value:.2f}%",
    )

    turn_results_into_csv(results, system_name, tech)


def main():
    """
    Main function to run the CHESS results processing.
    """
    # sys_names = ["CHESS"]
    # techniques = ["ori", "sr", "bt", "ca"]
    # for sys_name in sys_names:
    #     for tech in techniques:
    #         logger.log("info", f"Processing {sys_name} results for {tech} technique")
    #         read_chess_results(sys_name, tech)

    # read_mac_results("MAC-SQL", "ori")

    # sys_names = ["MAC-SQL"]
    # techniques = ["sr", "bt", "ca"]
    # for sys_name in sys_names:
    #     for tech in techniques:
    calculate_delta_ex("CHESS", "sr")


if __name__ == "__main__":
    main()

import os
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from src.core.logger.logger import Logger
from src.experiments.augmentation.utils import compute_exec_accuracy

logger = Logger(name=__name__)


def parse_id_and_db_id(filename: str) -> Tuple[int, str]:
    """
    Extract the integer ID and DB ID from filename.
    Example: '123_sports_complex.json' → (123, 'sports_complex')
    """
    match = re.match(r"^(\d+)_([^.]+)", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected format.")
    return int(match.group(1)), match.group(2)


def turn_results_into_csv(results: List[Dict], system_name: str, mode: str) -> None:
    """
    Save a list of result dictionaries into a CSV file for a given system and mode.
    """
    output_dir = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{system_name}_{mode}_results.csv"

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

    logger.log("info", f"Results for {system_name} in {mode} mode saved to {path}")


def read_chess_results(system_name: str, mode: str) -> None:
    """
    Read CHESS system JSON output and convert into structured CSV.
    """
    results = []
    base_path = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / system_name / f"BIRD_{mode}"
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

    turn_results_into_csv(results, system_name, mode)


def read_din_results(system_name: str, mode: str) -> None:
    """
    Read DIN-SQL logs and evaluate execution accuracy against ground truth.
    """
    data_folder = Path(os.getenv("DATA_FOLDER"))
    results_path = (
        data_folder / "experiments" / system_name / f"BIRD_{mode}" / "logs.csv"
    )
    db_path = data_folder / "benchmarks" / "Bird" / "bird_databases"

    mode_data_map = {
        "dev": data_folder / "benchmarks" / "Bird" / "bird_dev.json",
        "aug": data_folder
        / "experiments"
        / "augmentation"
        / "query_first"
        / "bird_qf_filtered.json",
    }

    data_file = mode_data_map.get(mode)
    if not data_file:
        logger.log("error", f"Invalid mode: {mode}")
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

    turn_results_into_csv(results, system_name, mode)


def read_mac_results(system_name: str, mode: str) -> None:
    """
    Parse MAC-SQL JSON output (dev or aug) and compute execution accuracy if needed.
    """
    data_folder = Path(os.getenv("DATA_FOLDER"))
    base_path = (
        data_folder / "experiments" / "augmentation" / system_name / f"BIRD_{mode}"
    )
    result_file = base_path / (
        "output_bird.json" if mode == "dev" else "eval_result_dev.json"
    )
    results = []

    try:
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if mode == "aug":
            with (
                data_folder
                / "experiments"
                / "augmentation"
                / "query_first"
                / "bird_qf_filtered.json"
            ).open("r") as f:
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

        else:  # dev
            for row in data:
                pred_sql = row["final_sql"].strip().replace("\n", " ")
                gold_sql = row["ground_truth"].strip()
                db_file = (
                    data_folder
                    / "benchmarks"
                    / "Bird"
                    / "bird_databases"
                    / f"{row['db_id']}/{row['db_id']}.sqlite"
                )

                results.append(
                    {
                        "id": row["idx"],
                        "db_id": row["db_id"],
                        "question": row["query"],
                        "evidence": row["evidence"],
                        "gold_SQL": gold_sql,
                        "pred_SQL": pred_sql,
                        "ex": str(compute_exec_accuracy(pred_sql, gold_sql, db_file)),
                    }
                )

    except Exception as e:
        logger.log("error", f"Failed reading MAC-SQL {mode} results: {e}")
        return

    turn_results_into_csv(results, system_name, mode)


def calculate_delta_ex(system_name: str) -> None:
    """
    Compute the delta in execution accuracy (aug - dev) per example.
    """
    base_dir = Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation"
    aug_path = base_dir / f"{system_name}_aug_results.csv"
    dev_path = base_dir / f"{system_name}_dev_results.csv"
    out_path = base_dir / f"{system_name}_delta_ex_results.csv"

    with dev_path.open("r", encoding="utf-8") as f:
        dev_lookup = {int(row["id"]): row for row in csv.DictReader(f)}

    delta_results = []
    unique_dev_rows = []
    with aug_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ext_id = int(str(row["id"])[2:6])
            dev_row = dev_lookup.get(ext_id)
            if dev_row not in unique_dev_rows:
                # Ensure we only add unique dev rows
                unique_dev_rows.append(dev_row)
            if not dev_row:
                continue

            try:
                delta = float(row["ex"]) - float(dev_row["ex"])
            except ValueError:
                delta = None

            delta_results.append(
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
                    "delta_ex": delta,
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
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
        writer.writerows(delta_results)

    # Logging breakdown
    counter = Counter(
        int(res["delta_ex"]) for res in delta_results if res["delta_ex"] is not None
    )
    total = len(delta_results)

    logger.log("info", f"Delta Execution Accuracy Summary for {system_name}:")
    for delta in [-1, 0, 1]:
        logger.log(
            "info", f"delta_ex = {delta} --> {counter[delta] / total * 100:.2f}%"
        )
        # We need to differentiate between delta ex == 0 when EX ori == 1 and EX ori == 0
        if delta == 0:
            ex_ori_1 = sum(
                1 for r in delta_results if r["aug_ex"] == "1" and r["dev_ex"] == "1"
            )
            ex_ori_0 = sum(
                1 for r in delta_results if r["aug_ex"] == "0" and r["dev_ex"] == "0"
            )
            logger.log("info", f"EX ori 1: {ex_ori_1 / total * 100:.2f}%")
            logger.log("info", f"EX ori 0: {ex_ori_0 / total * 100:.2f}%")

    aug_ex_1 = sum(1 for r in delta_results if r["aug_ex"] == "1")
    dev_ex_1 = sum(1 for r in dev_lookup.values() if r["ex"] == "1")

    logger.log("info", f"aug_ex == 1 → {aug_ex_1 / total * 100:.2f}%")
    logger.log("info", f"dev_ex == 1 → {dev_ex_1 / len(dev_lookup) * 100:.2f}%")

    # We will print the EX aug for the unique queries that match the aug set
    logger.log("info", f"Unique dev rows: {len(unique_dev_rows)}")
    # we need
    logger.log(
        "info",
        f"Unique dev rows with EX == 1: {sum(1 for r in unique_dev_rows if r['ex'] == '1')}",
    )
    # we need percentage instead of count
    logger.log(
        "info",
        f"Unique dev rows with EX == 1: {sum(1 for r in unique_dev_rows if r['ex'] == '1') / len(unique_dev_rows) * 100:.2f}%",
    )


def main() -> None:
    """
    Execute the results parsing and delta computation pipeline.
    """
    for mode in ["dev", "aug"]:
        read_mac_results("MAC-SQL", mode)
        read_chess_results("CHESS", mode)
        read_din_results("DIN-SQL", mode)

    for system in ["CHESS", "DIN-SQL", "MAC-SQL"]:
        calculate_delta_ex(system)


if __name__ == "__main__":
    main()

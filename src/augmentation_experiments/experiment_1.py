import os
import json
from pathlib import Path
from collections import defaultdict


def write_json(output_dir, filename, data):
    with open(output_dir / filename, "w") as f:
        json.dump(data, f, indent=2)


def write_sql(output_dir, filename, queries):
    with open(output_dir / filename, "w") as f:
        for query in queries:
            sql = query.get("SQL")
            db_id = query.get("db_id")
            if sql and db_id:
                f.write(f"{sql}\t{db_id}\n")


def main():
    base_dir = os.getenv("RULE_OUTPUTS_BASE")
    if base_dir is None:
        raise EnvironmentError("RULE_OUTPUTS_BASE environment variable is not set.")

    output_dir = Path(base_dir)
    all_extended_discarded = []
    all_extended_pruned = []
    all_original_discarded = []
    all_original_pruned = []

    for file in sorted(output_dir.glob("*_query_first_discarded_extended.json")):
        with file.open("r") as f:
            try:
                augmented_data = json.load(f)
                all_extended_discarded.extend(augmented_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")

    for file in sorted(output_dir.glob("*_query_first_filtered_extended.json")):
        with file.open("r") as f:
            try:
                original_data = json.load(f)
                all_extended_pruned.extend(original_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")

    for file in sorted(output_dir.glob("*_query_first_discarded_original.json")):
        with file.open("r") as f:
            try:
                original_data = json.load(f)
                all_original_discarded.extend(original_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")
    for file in sorted(output_dir.glob("*_query_first_filtered_original.json")):
        with file.open("r") as f:
            try:
                original_data = json.load(f)
                all_original_pruned.extend(original_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")

    # Create subfolder: output_dir/experiment_1
    experiment_dir = output_dir / "experiment_1"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    id_counter = defaultdict(int)
    [
        ext_dis.update(
            {
                "question_id": int(orig_dis["question_id"]) * 10000
                + id_counter[int(orig_dis["question_id"])]
            }
        )
        or id_counter.__setitem__(
            int(orig_dis["question_id"]), id_counter[int(orig_dis["question_id"])] + 1
        )
        for ext_dis, orig_dis in zip(all_extended_discarded, all_original_discarded)
    ]

    [
        ext_pru.update(
            {
                "question_id": int(orig_pru["question_id"]) * 10000
                + id_counter[int(orig_pru["question_id"])]
            }
        )
        or id_counter.__setitem__(
            int(orig_pru["question_id"]), id_counter[int(orig_pru["question_id"])] + 1
        )
        for ext_pru, orig_pru in zip(all_extended_pruned, all_original_pruned)
    ]

    # Write JSON files
    write_json(experiment_dir, "extended_discarded.json", all_extended_discarded)
    write_json(experiment_dir, "extended_pruned.json", all_extended_pruned)
    write_json(experiment_dir, "original_discarded.json", all_original_discarded)
    write_json(experiment_dir, "original_pruned.json", all_original_pruned)

    # Write SQL files
    write_sql(experiment_dir, "extended_discarded.sql", all_extended_discarded)
    write_sql(experiment_dir, "extended_pruned.sql", all_extended_pruned)
    write_sql(experiment_dir, "original_discarded.sql", all_original_discarded)
    write_sql(experiment_dir, "original_pruned.sql", all_original_pruned)


if __name__ == "__main__":
    main()

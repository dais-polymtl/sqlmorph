import os
import json
from pathlib import Path
from collections import defaultdict
from tabulate import tabulate


def clean_rewritten_fields_in_files():
    input_folder = Path(os.getenv("DATA_FOLDER")) / "rule_outputs" / "lt_elimination"
    techniques = ["syn_rep", "backtrans", "context_aug"]
    splits = ["train", "dev", "test"]

    for split in splits:
        for tech in techniques:
            file_path = input_folder / split / f"{split}_{tech}_queries.json"
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                # Clean rewritten_question
                entry["new_question"] = (
                    entry["new_question"].replace("Rewritten question:", "").strip()
                )

                entry["new_evidence"] = (
                    entry["new_evidence"].replace("Rewritten evidence:", "").strip()
                )

            # Save the cleaned data back to the same file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print("All rewritten questions and evidences cleaned and saved.")


def hiding_score_side_by_side():
    input_folder = Path(os.getenv("DATA_FOLDER")) / "rule_outputs" / "lt_elimination"
    techniques = ["syn_rep", "backtrans", "context_aug"]
    splits = ["train", "dev", "test"]

    data_by_split = {split: {tech: [] for tech in techniques} for split in splits}
    for split in splits:
        for tech in techniques:
            file = input_folder / split / f"{split}_{tech}_stats.json"
            with open(file) as f:
                data_by_split[split][tech] = json.load(f)

    for split in splits:
        # Collect all pairs across all techniques and types for this split
        all_pairs = set()

        # Store counts: dict[tech][type]['(relaxed, strict)'] = count
        counts = {
            tech: {"Q": defaultdict(int), "E": defaultdict(int)} for tech in techniques
        }

        # Fill counts and track all pairs
        for tech in techniques:
            queries = data_by_split[split][tech]
            for q in queries:
                qr = round(q["question_relaxed_score"], 3)
                qs = round(q["question_strict_score"], 3)
                er = round(q["evidence_relaxed_score"], 3)
                es = round(q["evidence_strict_score"], 3)

                q_pair = (qr, qs)
                e_pair = (er, es)

                counts[tech]["Q"][q_pair] += 1
                counts[tech]["E"][e_pair] += 1

                all_pairs.add(q_pair)
                all_pairs.add(e_pair)

        # Sort all pairs by relaxed then strict
        sorted_pairs = sorted(all_pairs)

        # Build table rows
        headers = ["Relaxed", "Strict"]
        for tech in techniques:
            headers.append(f"{tech} Q")
            headers.append(f"{tech} E")

        table = []
        for pair in sorted_pairs:
            row = [pair[0], pair[1]]
            for tech in techniques:
                q_count = counts[tech]["Q"].get(pair, 0)
                e_count = counts[tech]["E"].get(pair, 0)
                row.extend([q_count, e_count])
            table.append(row)

        print(f"\n=== {split.upper()} ===")
        print(tabulate(table, headers=headers, tablefmt="grid"))


def linker_table_stats():
    input_folder = Path(os.getenv("DATA_FOLDER")) / "rule_outputs" / "lt_elimination"
    techniques = ["syn_rep", "backtrans", "context_aug"]
    splits = ["train", "dev", "test"]

    # Load all data
    data_by_split = {split: {tech: [] for tech in techniques} for split in splits}
    for split in splits:
        for tech in techniques:
            file = input_folder / split / f"{split}_{tech}_stats.json"
            with open(file) as f:
                data_by_split[split][tech] = json.load(f)

    # Compute stats
    stats = {split: {} for split in splits}
    for split in splits:
        for tech in techniques:
            queries = data_by_split[split][tech]

            n_queries = len(queries)
            n_ques = n_ev = n_qcomps = n_ecomps = n_qlemmas = n_elemmas = 0
            fail_q_relaxed = fail_q_strict = fail_e_relaxed = fail_e_strict = 0

            sum_q_relaxed = sum_q_strict = sum_e_relaxed = sum_e_strict = 0.0

            for q in queries:
                if q.get("old_question") != "":
                    n_ques += 1
                if q.get("old_evidence") != "":
                    n_ev += 1
                q_comps = q["question_components"]
                e_comps = q["evidence_components"]

                n_qcomps += len(q_comps)
                n_ecomps += len(e_comps)
                n_qlemmas += sum(len(c["matched_lemmas"]) for c in q_comps)
                n_elemmas += sum(len(c["matched_lemmas"]) for c in e_comps)

                q_relaxed = q["question_relaxed_score"]
                q_strict = q["question_strict_score"]
                e_relaxed = q["evidence_relaxed_score"]
                e_strict = q["evidence_strict_score"]

                sum_q_relaxed += q_relaxed
                sum_q_strict += q_strict
                sum_e_relaxed += e_relaxed
                sum_e_strict += e_strict

                if q_relaxed < 1.0:
                    fail_q_relaxed += 1
                if q_strict < 1.0:
                    fail_q_strict += 1
                if e_relaxed < 1.0:
                    fail_e_relaxed += 1
                if e_strict < 1.0:
                    fail_e_strict += 1

            avg_q_relaxed = (
                round(sum_q_relaxed / n_queries, 3) if n_queries > 0 else 0.0
            )
            avg_q_strict = round(sum_q_strict / n_queries, 3) if n_queries > 0 else 0.0
            avg_e_relaxed = (
                round(sum_e_relaxed / n_queries, 3) if n_queries > 0 else 0.0
            )
            avg_e_strict = round(sum_e_strict / n_queries, 3) if n_queries > 0 else 0.0

            stats[split][tech] = {
                "n_queries": n_queries,
                "n_ques": n_ques,
                "n_ev": n_ev,
                "n_qcomps": n_qcomps,
                "n_ecomps": n_ecomps,
                "n_qlemmas": n_qlemmas,
                "n_elemmas": n_elemmas,
                "fail_q_relaxed": fail_q_relaxed,
                "fail_q_strict": fail_q_strict,
                "fail_e_relaxed": fail_e_relaxed,
                "fail_e_strict": fail_e_strict,
                "avg_q_relaxed": avg_q_relaxed,
                "avg_q_strict": avg_q_strict,
                "avg_e_relaxed": avg_e_relaxed,
                "avg_e_strict": avg_e_strict,
            }

    # Display results
    headers = [
        "tech",
        "n_queries",
        "n_ques",
        "n_ev",
        "n_qcomps",
        "n_ecomps",
        "n_qlemmas",
        "n_elemmas",
        "fail_q_relaxed",
        "fail_q_strict",
        "fail_e_relaxed",
        "fail_e_strict",
        "avg_q_relaxed",
        "avg_q_strict",
        "avg_e_relaxed",
        "avg_e_strict",
    ]

    for split in splits:
        print(f"\n=== {split.upper()} ===")
        table = [[tech] + list(stats[split][tech].values()) for tech in techniques]
        print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":

    linker_table_stats()
    hiding_score_side_by_side()

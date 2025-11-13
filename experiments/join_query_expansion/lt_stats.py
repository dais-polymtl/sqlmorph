from pathlib import Path
from typing import List, Dict
import json
import pickle
import os
import sqlglot
from sqlglot.expressions import Column, Table, Select, Join
from tabulate import tabulate
import re
import pandas as pd
from collections import Counter


def show_result_tuple_distribution(results, group_name, flag_key, model_key):
    """
    Show how many examples with `flag_key=True` have each (delta_ex, ex) tuple from model_key (e.g., "CHESS").
    """
    subset = [r for r in results if r.get(flag_key) and r.get(model_key)]
    total = len(subset)
    counter = Counter(
        tuple(r[model_key]) for r in subset if isinstance(r[model_key], (list, tuple))
    )

    print(f"\n🔎 {group_name} — {model_key} Results ({total} examples):")
    rows = []
    for tup, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        rows.append([tup, count, f"{count / total * 100:.1f}%"])
    print(
        tabulate(
            rows, headers=["(delta_ex, ex)", "Count", "Percent"], tablefmt="fancy_grid"
        )
    )


def lt_analysis_after_hiding(set_name: str, technique: str) -> None:
    set_data = read_jqgs_per_tech(set_name, technique)
    output_dir = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "lt_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{set_name}_{technique}_lt_analysis.json"

    # Read results for that technique from the three systems
    if set_name == "test":
        chess_results_file = (
            Path(os.getenv("DATA_FOLDER"))
            / "experiments"
            / "augmentation"
            / "lt_outputs"
            / f"CHESS_test_{technique}_results.csv"
        )
        mac_results_file = (
            Path(os.getenv("DATA_FOLDER"))
            / "experiments"
            / "augmentation"
            / "lt_outputs"
            / f"MAC-SQL_test_{technique}_results.csv"
        )
        din_results_file = (
            Path(os.getenv("DATA_FOLDER"))
            / "experiments"
            / "augmentation"
            / "lt_outputs"
            / f"DIN-SQL_test_{technique}_results.csv"
        )

        # we will read the results from the three systems
        chess_results = pd.read_csv(chess_results_file)
        mac_results = pd.read_csv(mac_results_file)
        din_results = pd.read_csv(din_results_file)

    all_rows = []
    for query in set_data:
        sql_query = query.get("SQL", "")
        linker_table = query.get("central_table", "")
        if not linker_table:
            continue

        attributes = extract_lt_attribs(sql_query, linker_table)
        question = query.get("new_question", "")
        evidence = query.get("new_evidence", "")

        obj = {
            "id": query["question_id"],
            "sql": sql_query,
            "q": question,
            "ev": evidence,
            "lt": linker_table,
            "lt_atts": attributes["all"],
            "lt_joins": attributes["joins"],
            "lt_proj_filt": attributes["proj_filt"],
            "q_has_joins": check_attributes_in_text(question, attributes["joins"]),
            "q_has_proj_filt": check_attributes_in_text(
                question, attributes["proj_filt"]
            ),
            "ev_has_joins": check_attributes_in_text(evidence, attributes["joins"]),
            "ev_has_proj_filt": check_attributes_in_text(
                evidence, attributes["proj_filt"]
            ),
        }
        if set_name == "test":
            chess_query_result = next(
                (
                    row
                    for _, row in chess_results.iterrows()
                    if row["id"] == query["question_id"]
                    and row["db_id"] == query["db_id"]
                ),
                None,
            )
            mac_query_result = next(
                (
                    row
                    for _, row in mac_results.iterrows()
                    if row["id"] == query["question_id"]
                    and row["db_id"] == query["db_id"]
                ),
                None,
            )
            din_query_result = next(
                (
                    row
                    for _, row in din_results.iterrows()
                    if row["id"] == query["question_id"]
                    and row["db_id"] == query["db_id"]
                ),
                None,
            )

            print(chess_query_result)

        # Only add results if found
        if chess_query_result is not None:
            obj["CHESS"] = (chess_query_result["delta_ex"], chess_query_result["ex"])
        if mac_query_result is not None:
            obj["MAC-SQL"] = (mac_query_result["delta_ex"], mac_query_result["ex"])
        if din_query_result is not None:
            obj["DIN-SQL"] = (din_query_result["delta_ex"], din_query_result["ex"])

        all_rows.append(obj)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)


# We will do a function that does a csv file with the results of this analysis. So for each set, we create a csv file with the following columns: question_id. SQL, question, evidence, central_table, central_table_attributes, central_table_joins, central_table_projections_and_filters, question_includes_joins, question_includes_pandf, evidence_includes_joins, evidence_includes_pandf
def lt_analysis_per_set(set_name: str) -> None:
    set_data = read_jqgs_per_set(set_name)
    output_dir = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "lt_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{set_name}_lt_analysis.json"

    all_rows = []

    for query in set_data:
        sql_query = query.get("SQL", "")
        linker_table = query.get("central_table", "")
        if not linker_table:
            continue

        attributes = extract_lt_attribs(sql_query, linker_table)
        question = query.get("question", "")
        evidence = query.get("evidence", "")

        all_rows.append(
            {
                "id": query["question_id"],
                "sql": sql_query,
                "q": question,
                "ev": evidence,
                "lt": linker_table,
                "lt_atts": attributes["all"],
                "lt_joins": attributes["joins"],
                "lt_proj_filt": attributes["proj_filt"],
                "q_has_joins": check_attributes_in_text(question, attributes["joins"]),
                "q_has_proj_filt": check_attributes_in_text(
                    question, attributes["proj_filt"]
                ),
                "ev_has_joins": check_attributes_in_text(evidence, attributes["joins"]),
                "ev_has_proj_filt": check_attributes_in_text(
                    evidence, attributes["proj_filt"]
                ),
            }
        )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)


def read_json_analysis(set_name: str) -> List[Dict]:
    output_dir = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "lt_outputs"
    )
    output_file = output_dir / f"{set_name}_lt_analysis.json"

    if not output_file.exists():
        raise FileNotFoundError(f"File {output_file} does not exist.")

    with output_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    # Same summary analysis as before
    q_non_empty = [r for r in results if r["q"].strip()]
    ev_non_empty = [r for r in results if r["ev"].strip()]
    q_total = len(q_non_empty)
    ev_total = len(ev_non_empty)

    q_has_joins = sum(r["q_has_joins"] for r in q_non_empty)
    q_has_proj_filt = sum(r["q_has_proj_filt"] for r in q_non_empty)
    ev_has_joins = sum(r["ev_has_joins"] for r in ev_non_empty)
    ev_has_proj_filt = sum(r["ev_has_proj_filt"] for r in ev_non_empty)

    summary_table = [
        ["Set", set_name],
        ["Total Examples", len(results)],
        ["Non-empty Questions", q_total],
        [
            "Q includes Joins",
            (
                f"{q_has_joins} ({(q_has_joins / q_total * 100):.1f}%)"
                if q_total
                else "0 (0.0%)"
            ),
        ],
        [
            "Q includes Proj/Filter",
            (
                f"{q_has_proj_filt} ({(q_has_proj_filt / q_total * 100):.1f}%)"
                if q_total
                else "0 (0.0%)"
            ),
        ],
        ["Non-empty Evidence", ev_total],
        [
            "Ev includes Joins",
            (
                f"{ev_has_joins} ({(ev_has_joins / ev_total * 100):.1f}%)"
                if ev_total
                else "0 (0.0%)"
            ),
        ],
        [
            "Ev includes Proj/Filter",
            (
                f"{ev_has_proj_filt} ({(ev_has_proj_filt / ev_total * 100):.1f}%)"
                if ev_total
                else "0 (0.0%)"
            ),
        ],
    ]

    print(tabulate(summary_table, tablefmt="grid"))

    return results


def read_json_per_tech_analysis(set_name: str, technique: str) -> List[Dict]:
    output_dir = (
        Path(os.getenv("DATA_FOLDER")) / "experiments" / "augmentation" / "lt_outputs"
    )
    output_file = output_dir / f"{set_name}_{technique}_lt_analysis.json"

    if not output_file.exists():
        raise FileNotFoundError(f"File {output_file} does not exist.")

    with output_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    # Same summary analysis as before
    q_non_empty = [r for r in results if r["q"].strip()]
    ev_non_empty = [r for r in results if r["ev"].strip()]
    q_total = len(q_non_empty)
    ev_total = len(ev_non_empty)

    q_has_joins = sum(r["q_has_joins"] for r in q_non_empty)
    q_has_proj_filt = sum(r["q_has_proj_filt"] for r in q_non_empty)
    ev_has_joins = sum(r["ev_has_joins"] for r in ev_non_empty)
    ev_has_proj_filt = sum(r["ev_has_proj_filt"] for r in ev_non_empty)

    summary_table = [
        ["Set", set_name],
        ["Total Examples", len(results)],
        ["Non-empty Questions", q_total],
        [
            "Q includes Joins",
            (
                f"{q_has_joins} ({(q_has_joins / q_total * 100):.1f}%)"
                if q_total
                else "0 (0.0%)"
            ),
        ],
        [
            "Q includes Proj/Filter",
            (
                f"{q_has_proj_filt} ({(q_has_proj_filt / q_total * 100):.1f}%)"
                if q_total
                else "0 (0.0%)"
            ),
        ],
        ["Non-empty Evidence", ev_total],
        [
            "Ev includes Joins",
            (
                f"{ev_has_joins} ({(ev_has_joins / ev_total * 100):.1f}%)"
                if ev_total
                else "0 (0.0%)"
            ),
        ],
        [
            "Ev includes Proj/Filter",
            (
                f"{ev_has_proj_filt} ({(ev_has_proj_filt / ev_total * 100):.1f}%)"
                if ev_total
                else "0 (0.0%)"
            ),
        ],
    ]

    print(tabulate(summary_table, tablefmt="grid"))

    if set_name == "test":
        breakdowns = [
            ("Q with joins", "q_has_joins"),
            ("Q with proj/filter", "q_has_proj_filt"),
            ("Ev with joins", "ev_has_joins"),
            ("Ev with proj/filter", "ev_has_proj_filt"),
        ]

    for group_name, flag in breakdowns:
        for model_key in ["CHESS", "MAC-SQL", "DIN-SQL"]:
            show_result_tuple_distribution(results, group_name, flag, model_key)

    return results


# def check_attributes_in_text(text: str, attributes: List[str]) -> bool:
#     text = text.lower()  # Normalize the input text

#     qualified_atts = [att for att in attributes if '.' in att]
#     unqualified_set = set(att.split('.')[-1].lower() for att in qualified_atts)

#     # Build regex pattern for all possible word-boundary matches
#     patterns = set()

#     # Qualified (as full strings)
#     for qatt in qualified_atts:
#         patterns.add(rf"\b{re.escape(qatt.lower())}\b")

#     # Unqualified from qualified
#     for uatt in unqualified_set:
#         patterns.add(rf"\b{re.escape(uatt)}\b")

#     # Explicitly unqualified
#     for att in attributes:
#         if '.' not in att:
#             patterns.add(rf"\b{re.escape(att.lower())}\b")

#     for pattern in patterns:
#         if re.search(pattern, text):
#             return True

#     return False


def check_attributes_in_text(text: str, attributes: List[str]) -> bool:
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)  # tokenize the text (words only)

    qualified_atts = [att for att in attributes if "." in att]
    unqualified_set = set(att.split(".")[-1].lower() for att in qualified_atts)

    # Combine all attribute forms to check (qualified + unqualified + explicit unqualified)
    all_atts = set()
    for att in qualified_atts:
        all_atts.add(att.lower())
    for att in unqualified_set:
        all_atts.add(att)
    for att in attributes:
        if "." not in att:
            all_atts.add(att.lower())

    # 1) Try exact word-boundary match first
    for att in all_atts:
        pattern = rf"\b{re.escape(att)}\b"
        if re.search(pattern, text):
            return True

    # 2) If no exact match, fallback: split attribute by "_" or " " and check parts
    for att in all_atts:
        parts = re.split(r"[_\s]+", att)
        for part in parts:
            if not part or len(part) == 1:  # skip empty
                continue
            for token in tokens:
                # relaxed check: token startswith part OR token == part
                if token.startswith(part) or token == part:
                    return True

    return False


def extract_lt_attribs(sql_query: str, central_table: str) -> Dict[str, List[str]]:
    """Extract central table attributes used in joins, projections, and filters (case-insensitive)."""
    try:
        parsed = sqlglot.parse_one(sql_query, dialect="mysql")
    except Exception as e:
        print(f"SQL parsing error: {e}")
        return {"all": [], "joins": [], "proj_filt": []}

    central_table = central_table.lower()
    central_aliases = set()
    all_attrs = set()
    join_attrs = set()
    proj_filt_attrs = set()

    # Step 1: Get aliases for central table
    for node in parsed.find_all(Table):
        if node.name.lower() == central_table:
            alias = node.args.get("alias")
            if alias and alias.name:
                central_aliases.add(alias.name.lower())
            central_aliases.add(central_table)

    def is_central_column(col_node: Column) -> bool:
        table = col_node.table.lower() if col_node.table else None
        return table in central_aliases or (
            table is None and central_table in central_aliases
        )

    def full_col(col_node: Column) -> str:
        table = col_node.table if col_node.table else None
        return f"{table}.{col_node.name}" if table else col_node.name

    # Step 2: Find all central table columns
    for node in parsed.walk():
        if isinstance(node, Column) and is_central_column(node):
            all_attrs.add(full_col(node))

    # Step 3: Joins — look in ON clause of JOINs
    for join in parsed.find_all(Join):
        on_expr = join.args.get("on")
        if on_expr:
            for col in on_expr.find_all(Column):
                if is_central_column(col):
                    join_attrs.add(full_col(col))

    # Step 4: Projections — SELECT columns
    if isinstance(parsed, Select):
        for expr in parsed.expressions:
            for col in expr.find_all(Column):
                if is_central_column(col):
                    proj_filt_attrs.add(full_col(col))

    # Step 5: Filters — WHERE clause
    where_expr = parsed.args.get("where")
    if where_expr:
        for col in where_expr.find_all(Column):
            if is_central_column(col):
                proj_filt_attrs.add(full_col(col))

    return {
        "all": sorted(all_attrs),
        "joins": sorted(join_attrs),
        "proj_filt": sorted(proj_filt_attrs),
    }


def read_jqgs_per_set(set_name: str) -> List[Dict]:
    """
    Reads JSON queries from a specified set (train, dev, test).
    """
    data_folder = (
        Path(os.getenv("DATA_FOLDER")) / "rule_outputs" / "lt_elimination" / set_name
    )
    file_path = data_folder / f"{set_name}_jqgs_with_lt.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def read_jqgs_per_tech(set_name: str, technique: str) -> List[Dict]:
    """
    Reads JSON queries from a specified set (train, dev, test).
    """
    data_folder = (
        Path(os.getenv("DATA_FOLDER")) / "rule_outputs" / "lt_elimination" / set_name
    )

    technique_map = {
        "sr": "syn_rep",
        "bt": "backtrans",
        "ca": "context_aug",
    }
    file_path = (
        data_folder / f"{set_name}_{technique_map.get(technique, "")}_stats.json"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = json.load(f)

    return data


def main():
    set_name = "test"

    technique = "sr"

    lt_analysis_after_hiding(set_name, technique)
    read_json_per_tech_analysis(set_name, technique)


if __name__ == "__main__":
    main()

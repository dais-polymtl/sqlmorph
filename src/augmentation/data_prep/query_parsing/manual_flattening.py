from __future__ import annotations
from typing import List, Dict, Tuple

from llm_flattening import compare_queries
from query_loader import contains_nested_select_or_cte
from sqlglot import parse_one

import os
import json
from pathlib import Path


def prompt_user_for_flattening(
    nested_query: str, index: int, total: int
) -> Tuple[bool, str]:
    print(f"\n🧩 Query {index + 1} / {total}")
    print("🔍 Original nested SQL query:")
    print(nested_query)
    while True:
        ans = input("\n❓ Is this query impossible to flatten? (Y/N): ").strip().lower()
        if ans in {"y", "n"}:
            break
        print("Please enter 'Y' or 'N'.")
    if ans == "y":
        return True, ""
    print("\n✏️  Enter your manually flattened version (single line):")
    user_query = input("> ").strip()
    return False, user_query


def flatten_queries_manually(
    queries: List[Dict[str, str]], dataset: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Interactively prompts the user to flatten nested SQL queries manually.
    Tracks queries impossible to flatten separately.
    """
    manually_flattened_queries = []
    impossible_to_flatten = []
    total = len(queries)

    for i, query in enumerate(queries):
        db_id = query.get("db_id", "")
        sql_query = query.get("SQL") or query.get("sql", "")
        if not db_id or not sql_query:
            print(f"⚠️  Skipping invalid query at index {i}.")
            continue

        data_folder = os.getenv("DATA_FOLDER", "data")
        base_db_path = (
            Path(data_folder) / "benchmarks" / dataset / f"{dataset.lower()}_databases"
        )

        if dataset == "Bird":
            db_path = base_db_path / db_id / f"{db_id}.sqlite"
        elif dataset == "Beaver":
            db_path = base_db_path / db_id / f"{db_id}.sql"
        else:
            print(f"❌ Unsupported dataset '{dataset}'. Skipping query.")
            continue

        while True:
            impossible, user_query = prompt_user_for_flattening(sql_query, i, total)

            if impossible:
                print("⚠️ Marking this query as impossible to flatten.")
                impossible_to_flatten.append(query)
                break

            try:
                if contains_nested_select_or_cte(
                    parse_one(user_query, dialect="mysql")
                ):
                    print("❌ Your query still appears to be nested. Please try again.")
                    continue
            except Exception as e:
                print(f"❌ Syntax error in your query: {e}")
                continue

            try:
                if compare_queries(sql_query, user_query, db_path):
                    print("✅ Success! Your flattened query returns the same results.")
                    query["flattened_query"] = user_query
                    manually_flattened_queries.append(query)
                    break
                else:
                    print("❌ Query results differ. Please review your flattening.")
            except Exception as e:
                print(f"❌ Error comparing queries: {e}")
                print("🔄 Please try again.")

    return manually_flattened_queries, impossible_to_flatten


def enter_tables_joins_manually(dataset: str) -> None:
    """
    Prompts user for table and join information for non-flattened queries and writes parsed versions to file.
    """
    data_folder = os.getenv("DATA_FOLDER", "data")
    non_parsed_path = Path(
        data_folder, "new_parsing", f"{dataset.lower()}_non_parsed_queries"
    )
    parsed_path = Path(data_folder, "new_parsing", f"{dataset.lower()}_parsed_queries")

    with open(non_parsed_path / "non_flattened_queries.json", "r") as f:
        non_flattened_queries = json.load(f)

    data_per_db = {
        db_id: [] for db_id in set(q["db_id"] for q in non_flattened_queries)
    }

    print("📝 The following queries are not flattened. Enter details manually.")
    for query in non_flattened_queries:
        db_id = query.get("db_id", "")
        print("\n🔍 SQL Query:", query.get("SQL", "") or query.get("sql", ""))
        print("📋 Enter the involved tables (comma-separated):")
        tables = [t.strip() for t in input("> ").strip().split(",")]
        print("🔗 Enter join relations like Tab1.id = Tab2.id (comma-separated):")
        join_relations = [jr.strip() for jr in input("> ").strip().split(",")]

        query["tables"] = [{"table_name": t, "table_alias": ""} for t in tables]
        query["jr_w_aliases"] = []
        query["jr_wo_aliases"] = join_relations
        query["flattened_query"] = ""
        data_per_db[db_id].append(query)

    for db_id, queries in data_per_db.items():
        target_file = parsed_path / f"parsed_queries_{db_id}.json"
        if target_file.exists():
            with open(target_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(queries)
        with open(target_file, "w") as f:
            json.dump(existing_data, f, indent=4)

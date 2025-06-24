import os
import json
from pathlib import Path
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sqlglot
from sqlglot.expressions import Table

import sqlparse
from sqlparse.sql import Comparison
import re
from typing import List

from src.augmentation.jq_graph_augmentation.sql_query_gen import (
    aug_n_table_sql_queries,
)
from src.augmentation.jq_graph_augmentation.nl_query_gen import (
    gen_nl,
)
from src.augmentation.jq_graph_augmentation.persistence import save_query_first


def generate_aug_queries_rule_1(db_ids):
    for db_id, max_tables in db_ids:
        filtered_aug, _ = aug_n_table_sql_queries(
            db_id=db_id,
            num_tables=max_tables + 1,
            graph_first=False,
        )
        filtered_aug = gen_nl(
            filtered_aug,
            graph_first=False,
        )
        save_query_first(filtered_aug, f"experiment_1_filtered_{db_id}")


def extract_joins(token_list) -> List[Comparison]:
    join_pattern = re.compile(r"\b(\w+\.\w+)\s*=\s*(\w+\.\w+)\b", re.IGNORECASE)
    join_conditions = []
    for token in token_list:
        if isinstance(token, Comparison):
            parent_token = token.parent
            if any(str(t).upper() == "ON" for t in parent_token.tokens):
                if join_pattern.match(str(token)):
                    join_conditions.append(token)
        elif token.is_group:
            join_conditions.extend(extract_joins(token.tokens))
    return join_conditions


def get_joins_w_aliases(sql: str) -> List[str]:
    parsed = sqlparse.parse(sql)
    join_conditions = []
    for statement in parsed:
        join_conditions.extend([str(t) for t in extract_joins(statement.tokens)])
    return join_conditions


def represent_queries_in_graphs(queries):
    graphs_list = []

    for query in queries:
        sql_query = query.get("SQL")
        if not sql_query:
            continue

        # Use sqlglot to parse tables and aliases
        try:
            parsed_query = sqlglot.parse_one(sql_query, dialect="mysql")
            tables = [
                (table.name, table.alias) for table in parsed_query.find_all(Table)
            ]
        except Exception as e:
            print(f"sqlglot failed to parse: {sql_query[:80]}... Error: {e}")
            tables = []

        # Use sqlparse to extract join conditions
        join_edges = get_joins_w_aliases(sql_query)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(table[0] for table in tables)

        for condition in join_edges:
            left, right = (
                condition.split("=")[0].strip(),
                condition.split("=")[1].strip(),
            )
            left_table = next(
                (table[0] for table in tables if table[1] == left.split(".")[0]), None
            )
            right_table = next(
                (table[0] for table in tables if table[1] == right.split(".")[0]), None
            )
            if left_table and right_table:
                G.add_edge(left_table, right_table, condition=condition)

        graphs_list.append(G)

    return graphs_list


# 🔹 NEW: Extract structural features from query graphs
def extract_graph_features(graphs, label):
    features = []
    for g in graphs:
        if len(g.nodes) == 0:
            continue
        feature = {
            "set": label,
            "num_tables": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "avg_degree": sum(dict(g.degree()).values()) / float(g.number_of_nodes()),
            "has_cycle": int(any(nx.cycle_basis(g))),  # 1 if cyclic, 0 if acyclic
            "avg_centrality": sum(nx.degree_centrality(g).values()) / len(g),
        }
        features.append(feature)
    return features


def plot_box_distributions(features_df, output_dir):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    features = ["num_tables", "num_edges", "avg_degree", "avg_centrality"]
    sets = ["original", "augmented"]

    for feature in features:
        data = [features_df[features_df["set"] == s][feature].values for s in sets]

        plt.figure(figsize=(8, 5))
        plt.boxplot(
            data,
            labels=sets,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="red"),
        )
        plt.title(f"Boxplot of {feature.replace('_', ' ').title()}")
        plt.ylabel(feature.replace("_", " ").title())
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{feature}_boxplot.png")
        plt.close()


def plot_violin_distributions(features_df, output_dir):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    features = ["num_tables", "num_edges", "avg_degree", "avg_centrality"]

    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=features_df, x="set", y=feature, palette="pastel")
        plt.title(f"Violin Plot of {feature.replace('_', ' ').title()}")
        plt.ylabel(feature.replace("_", " ").title())
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{feature}_violinplot.png")
        plt.close()


def plot_cycle_bar(features_df, output_dir):

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    sets = ["original", "augmented"]
    counts = []
    for s in sets:
        group = features_df[features_df["set"] == s]
        cyclic = np.sum(group["has_cycle"] == 1)
        acyclic = np.sum(group["has_cycle"] == 0)
        counts.append((cyclic, acyclic))

    width = 0.35
    x = np.arange(len(sets))

    plt.figure(figsize=(6, 4))
    plt.bar(
        x - width / 2, [c[0] for c in counts], width, label="Cyclic", color="orange"
    )
    plt.bar(
        x + width / 2, [c[1] for c in counts], width, label="Acyclic", color="skyblue"
    )
    plt.xticks(x, sets)
    plt.title("Presence of Cycles in Graphs")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plots_dir / "cycle_presence.png")
    plt.close()


# 🔹 NEW: Summary stats
def save_summary_stats(features_df, output_dir):
    summary_stats = features_df.groupby("set").agg(
        {
            "num_tables": ["mean", "std", "min", "max"],
            "num_edges": ["mean", "std", "min", "max"],
            "avg_degree": ["mean", "std", "min", "max"],
            "avg_centrality": ["mean", "std", "min", "max"],
            "has_cycle": ["sum", "mean"],
        }
    )
    summary_stats.to_csv(output_dir / "summary_statistics.csv")


def main(db_ids):
    base_dir = os.getenv("RULE_OUTPUTS_BASE")
    if base_dir is None:
        raise EnvironmentError("RULE_OUTPUTS_BASE environment variable is not set.")

    output_dir = Path(base_dir) / "rule_2"
    all_augmented_queries = []
    all_original_queries = []

    for file in sorted(output_dir.glob("exp1_filt_*_extended.json")):
        with file.open("r") as f:
            try:
                augmented_data = json.load(f)
                all_augmented_queries.extend(augmented_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")

    for file in sorted(output_dir.glob("exp1_filt_*_original.json")):
        with file.open("r") as f:
            try:
                original_data = json.load(f)
                all_original_queries.extend(original_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {file}: {e}")

    # === Represent queries as graphs ===
    augmented_graphs = represent_queries_in_graphs(all_augmented_queries)
    original_graphs = represent_queries_in_graphs(all_original_queries)

    # 🔹 Output subdir for experiment
    experiment_dir = output_dir / "experiment_2"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 🔹 Compute graph features
    original_features = extract_graph_features(original_graphs, label="original")
    augmented_features = extract_graph_features(augmented_graphs, label="augmented")
    features_df = pd.DataFrame(original_features + augmented_features)
    print("Columns: ", features_df.columns)
    print()
    print()

    # 🔹 Plot feature distributions
    plot_box_distributions(features_df, experiment_dir)  # Box plots
    plot_violin_distributions(features_df, experiment_dir)  # Violin plots
    plot_cycle_bar(features_df, experiment_dir)  # Optional: cycle presence

    # 🔹 Save summary statistics
    save_summary_stats(features_df, experiment_dir)


if __name__ == "__main__":
    db_ids = [
        ("california_schools", 2),
        ("card_games", 3),
        ("codebase_community", 4),
        ("debit_card_specializing", 3),
        ("european_football_2", 4),
        ("financial", 5),
        ("formula_1", 4),
        ("student_club", 4),
        ("superhero", 4),
        ("thrombosis_prediction", 3),
        ("toxicology", 3),
    ]
    main(db_ids)

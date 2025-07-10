import os
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from src.core.logger.logger import Logger

logger = Logger(__name__)


def save_jq_graph_image(
    jq_graph: nx.Graph,
    db_id: str,
    jq_graph_id: int,
    node_labels: Dict,
    edge_labels: Dict,
    output_dir: str,
) -> None:
    # Compute layout
    pos = nx.spring_layout(jq_graph, seed=42)

    plt.figure(figsize=(12, 9))

    # Draw the subgraph only
    nx.draw_networkx_nodes(jq_graph, pos, node_color="lightblue", node_size=2500)
    nx.draw_networkx_edges(
        jq_graph, pos, edgelist=jq_graph.edges, edge_color="blue", width=2
    )
    nx.draw_networkx_labels(
        jq_graph, pos, labels=node_labels, font_size=10, font_color="black"
    )
    nx.draw_networkx_edge_labels(
        jq_graph, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    # Title & output
    plt.title(f"jq_graph {jq_graph_id} in {db_id}")
    image_dir = Path(output_dir) / db_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(image_dir / f"jq_graph_{jq_graph_id}.png")
    plt.close()


def create_jq_per_db(
    dataset: str,
    file_path: str,
    output_dir: str,
    jq_graph_count_by_table: Dict[str, Dict[int, int]],
    query_count_by_db: Dict[str, int],
) -> None:

    with open(file_path, "r") as f:
        queries = json.load(f)

    # Store accumulated graphs by (db_id, num_tables)
    jq_graph_buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for query in queries:
        query_w_graph = query.copy()
        db_id = query["db_id"]
        query_count_by_db[db_id] = query_count_by_db.get(db_id, 0) + 1

        table_names = query.get("tables", [])
        num_tables = len(table_names)

        jq_graph = nx.Graph()

        for table in table_names:
            table_name = table["table_name"]
            table_alias = table.get("table_alias", "")

            node_id = (table_name, table_alias)
            jq_graph.add_node(
                node_id,
                name=table_name,
                alias=table_alias,
            )

            # Build edges
        edges_w_joins = {}
        for relation in query["jr_w_aliases"]:
            left_key, right_key = map(str.strip, relation.split("="))
            left_node, right_node = left_key.split(".")[0], right_key.split(".")[0]

            for target_node in (left_node, right_node):
                matched = next(
                    node
                    for node in jq_graph.nodes
                    if node[0].lower() == target_node.lower()
                    or node[1].lower() == target_node.lower()
                )
                if target_node == left_node:
                    matched_left_node = matched
                else:
                    matched_right_node = matched

            edge = (matched_left_node, matched_right_node)
            edges_w_joins.setdefault(edge, []).append(relation)

        for edge, joins in edges_w_joins.items():
            jq_graph.add_edge(*edge, joins=joins)

        # Store graph in in-memory bucket
        jq_graph_count_by_table.setdefault(db_id, {}).setdefault(num_tables, 0)
        jq_graph_count_by_table[db_id][num_tables] += 1

        key = (db_id, num_tables)
        save_jq_graph_image(
            jq_graph=jq_graph,
            db_id=db_id,
            jq_graph_id=jq_graph_count_by_table[db_id][num_tables],
            node_labels={
                node: f"{node[0]} ({node[1]})" if node[1] else node[0]
                for node in jq_graph.nodes
            },
            edge_labels={
                edge: str(jq_graph.edges[edge].get("joins", []))
                for edge in jq_graph.edges
                if "joins" in jq_graph.edges[edge]
            },
            output_dir=output_dir,
        )
        query_w_graph["jq_graph"] = jq_graph
        jq_graph_buckets.setdefault(key, []).append(query_w_graph)

    # Save each bucket only once
    for (db_id, num_tables), graph_list in jq_graph_buckets.items():
        db_id_output_dir = Path(output_dir) / db_id
        db_id_output_dir.mkdir(parents=True, exist_ok=True)

        bucket_file = db_id_output_dir / f"jq_graphs_{num_tables}_tables.pkl"

        # If file exists, merge existing data
        if bucket_file.exists():
            with open(bucket_file, "rb") as f:
                graph_list = pickle.load(f) + graph_list

        with open(bucket_file, "wb") as f:
            pickle.dump(graph_list, f)


def create_jq_graphs(dataset: str) -> None:
    jq_graph_count_by_table: Dict[str, Dict[int, int]] = {}
    query_count_by_db: Dict[str, int] = {}

    interm_queries_dir = (
        Path(os.getenv("DATA_FOLDER"))
        / "new_parsing"
        / f"{dataset.lower()}_parsed_queries"
    )
    output_dir = Path(os.getenv("DATA_FOLDER")) / "rule_inputs"
    # Process each pattern file
    for file_name in os.listdir(interm_queries_dir):
        if file_name.endswith(".json"):
            file_path = Path(interm_queries_dir) / file_name
            create_jq_per_db(
                dataset,
                file_path,
                output_dir,
                jq_graph_count_by_table,
                query_count_by_db,
            )

    # Print summary
    for db_id, table_counts in jq_graph_count_by_table.items():
        for num_tables, count in table_counts.items():
            logger.log(
                "info",
                f"Database {db_id}: {count} jq_graphs with {num_tables} tables",
            )
    logger.log("info", f"Total queries processed per database: {query_count_by_db}")

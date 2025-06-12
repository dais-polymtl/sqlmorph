import os
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from src.core.logger.logger import Logger

logger = Logger(__name__)


def load_schema_graph(dataset: str, db_id: str) -> nx.Graph:
    graphs_folder_path = (
        Path(os.getenv("DATA_FOLDER")) / "graph_data" / f"{dataset}_graphs" / "pickles"
    )
    schema_graph_path = Path(graphs_folder_path) / f"{db_id}_graph.pkl"
    with open(schema_graph_path, "rb") as f:
        return pickle.load(f)


def save_subgraph_image(
    main_graph: nx.Graph,
    subgraph: nx.Graph,
    db_id: str,
    jq_graph_id: int,
    table_names: List[str],
    edge_labels: Dict,
    output_dir: str,
) -> None:

    pos = nx.spring_layout(main_graph)
    plt.figure(figsize=(12, 9))

    nx.draw(
        main_graph,
        pos,
        with_labels=True,
        node_color="lightgray",
        edge_color="lightgray",
        node_size=2000,
        font_size=10,
    )
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=subgraph.nodes, node_color="lightblue", node_size=2500
    )
    nx.draw_networkx_edges(
        main_graph, pos, edgelist=subgraph.edges, edge_color="blue", width=2
    )
    nx.draw_networkx_edge_labels(
        main_graph, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    plt.title(f"jq_graph {jq_graph_id} (Tables: {len(table_names)}) in DB {db_id}")
    output_dir = Path(os.getenv("DATA_FOLDER")) / "rule_inputs" / db_id
    image_dir = Path(output_dir) / "images"
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

    for i, query in enumerate(queries):
        db_id = query["db_id"]
        query_count_by_db[db_id] = query_count_by_db.get(db_id, 0) + 1
        main_graph = load_schema_graph(dataset, db_id)

        if "tables" in query and len(query["tables"]) == 1:
            table_name = query["tables"][0]["table_name"]
            subgraph = nx.Graph()
            subgraph.add_node(table_name)
            num_tables = 1
            jq_graph_count_by_table.setdefault(db_id, {}).setdefault(num_tables, 0)

            db_id_output_dir = Path(output_dir) / db_id

            jq_graph_signature = frozenset([table_name.lower()])
            bucket_dir = db_id_output_dir / f"jq_graphs_{num_tables}_tables.pkl"

            # Check if pattern already exists
            if bucket_dir.exists():
                with open(bucket_dir, "rb") as f:
                    all_jq_graphs = pickle.load(f)
            else:
                all_jq_graphs = []

            # Save new pattern and subgraph if not already saved
            seen_graph_signatures = [
                jq_graph["jq_graph_signature"] for jq_graph in all_jq_graphs
            ]
            if jq_graph_signature in seen_graph_signatures:
                jq_graph_index = seen_graph_signatures.index(jq_graph_signature)
                all_jq_graphs[jq_graph_index]["equivalent_queries"].append(query)

            else:
                subgraph_data = {
                    "jq_graph_signature": jq_graph_signature,
                    "subgraph": subgraph,
                    "equivalent_queries": [query],
                }
                all_jq_graphs.append(subgraph_data)

                edge_labels = {}
                save_subgraph_image(
                    main_graph,
                    subgraph,
                    db_id,
                    i,
                    [table_name],
                    edge_labels,
                    output_dir,
                )
                jq_graph_count_by_table[db_id][num_tables] += 1
            with open(bucket_dir, "wb") as f:
                pickle.dump(all_jq_graphs, f)
        else:
            join_relations = query["jr_wo_aliases"]
            table_names = set()
            columns_in_relations = set()
            edge_labels = {}
            edges_to_include = set()

            for relation in join_relations:
                left_col, right_col = relation.split("=")
                left_col, right_col = left_col.strip(), right_col.strip()
                table_names.update([x.split(".")[0] for x in [left_col, right_col]])
                columns_in_relations.add(
                    frozenset([left_col.lower(), right_col.lower()])
                )
                left_node, right_node = left_col.split(".")[0], right_col.split(".")[0]
                matched_left_node = next(
                    (
                        node
                        for node in main_graph.nodes
                        if node.lower() == left_node.lower()
                    ),
                    left_node,
                )
                matched_right_node = next(
                    (
                        node
                        for node in main_graph.nodes
                        if node.lower() == right_node.lower()
                    ),
                    right_node,
                )

                edge = (matched_left_node, matched_right_node)

                if edge not in edge_labels:
                    edge_labels[edge] = [relation]
                else:
                    if relation not in edge_labels[edge]:
                        edge_labels[edge].append(relation)
                edges_to_include.add(edge)

                if main_graph.has_edge(*edge):
                    main_graph[edge[0]][edge[1]]["label"] = "; ".join(edge_labels[edge])

            subgraph_nodes = list(table_names)
            subgraph = nx.Graph()
            subgraph.add_nodes_from(subgraph_nodes)
            for edge in edges_to_include:
                if main_graph.has_edge(*edge):
                    subgraph.add_edge(*edge, **main_graph.get_edge_data(*edge))

            num_tables = len(subgraph_nodes)
            jq_graph_count_by_table.setdefault(db_id, {}).setdefault(num_tables, 0)

            if num_tables == 0:
                print("No tables found in the subgraph")
                print(query)
                break

            db_id_output_dir = Path(output_dir) / db_id

            jq_graph_signature = (
                frozenset([node.lower() for node in subgraph_nodes]),
                frozenset(columns_in_relations),
            )
            bucket_dir = db_id_output_dir / f"jq_graphs_{num_tables}_tables.pkl"

            # Check if pattern already exists
            if bucket_dir.exists():
                with open(bucket_dir, "rb") as f:
                    all_jq_graphs = pickle.load(f)
            else:
                all_jq_graphs = []

            # Save new pattern and subgraph if not already saved
            seen_graph_signatures = [
                jq_graph["jq_graph_signature"] for jq_graph in all_jq_graphs
            ]
            if jq_graph_signature in seen_graph_signatures:
                jq_graph_index = seen_graph_signatures.index(jq_graph_signature)
                all_jq_graphs[jq_graph_index]["equivalent_queries"].append(query)
            else:
                subgraph_data = {
                    "jq_graph_signature": jq_graph_signature,
                    "subgraph": subgraph,
                    "equivalent_queries": [query],
                }
                all_jq_graphs.append(subgraph_data)

                save_subgraph_image(
                    main_graph,
                    subgraph,
                    db_id,
                    i,
                    table_names,
                    edge_labels,
                    output_dir,
                )
                jq_graph_count_by_table[db_id][num_tables] += 1
            with open(bucket_dir, "wb") as f:
                pickle.dump(all_jq_graphs, f)


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
                f"Database {db_id}: {count} unique jq_graphs with {num_tables} tables",
            )
    logger.log("info", f"Total queries processed per database: {query_count_by_db}")

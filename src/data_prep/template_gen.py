import os
import pickle
import json
import logging
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_graph(graph_folder_path: str) -> Dict[str, nx.Graph]:
    """
    Loads graph data from pickle files in the specified folder.

    Args:
        graph_folder_path (str): Path to the folder containing graph pickle files.

    Returns:
        Dict[str, nx.Graph]: A dictionary mapping db_id to the graph data.
    """
    graphs = {}
    for filename in os.listdir(graph_folder_path):
        if filename.endswith("_graph.pkl"):
            db_id = filename.replace("_graph.pkl", "")
            file_path = os.path.join(graph_folder_path, filename)
            with open(file_path, "rb") as file:
                graphs[db_id] = pickle.load(file)
    logging.info(f"Loaded {len(graphs)} graphs from {graph_folder_path}")
    return graphs


def load_patterns(pattern_folder_path: str) -> Dict[int, List[nx.Graph]]:
    """
    Loads pattern data from pickle files for various node sizes.

    Args:
        pattern_folder_path (str): Path to the folder containing pattern pickle files.

    Returns:
        Dict[int, List[nx.Graph]]: A dictionary mapping number of nodes to pattern graphs.
    """
    patterns = {}
    for num_nodes in range(3, 8):
        pattern_file = os.path.join(
            pattern_folder_path, f"patterns_{num_nodes}_nodes.pkl"
        )
        if os.path.exists(pattern_file):
            with open(pattern_file, "rb") as file:
                patterns[num_nodes] = pickle.load(file)
    logging.info(f"Loaded patterns for node sizes 3 to 7.")
    return patterns


def generate_single_table_queries(graph: nx.Graph) -> List[Dict[str, Any]]:
    """
    Generates query templates for single-table queries based on the graph.

    Args:
        graph (nx.Graph): The graph data representing the schema.

    Returns:
        List[Dict[str, Any]]: A list of query templates for single-table queries.
    """
    queries = []
    for table in graph.nodes:
        num_columns = len(graph.nodes[table]["column_names"])
        query_info = {
            "from": [table],
            "tables": [
                {
                    "table": table,
                    "column_names": graph.nodes[table]["column_names"],
                    "column_types": graph.nodes[table]["column_types"],
                }
            ],
            "joinable_relations": [],
            "num_tables": 1,
            "num_joinable_keys": 0,
            "joinable_keys_types": {},
            "num_output_columns": num_columns,
            "pattern": ["A"],
        }
        queries.append(query_info)
    return queries


def generate_two_table_queries(graph: nx.Graph) -> List[Dict[str, Any]]:
    """
    Generates query templates for two-table queries based on the graph.

    Args:
        graph (nx.Graph): The graph data representing the schema.

    Returns:
        List[Dict[str, Any]]: A list of query templates for two-table queries.
    """
    queries = []
    edges = list(graph.edges(data=True))
    for edge in edges:
        table1, table2 = edge[:2]
        joinable_relations = []
        join_keys_count = 0
        joinable_keys_types = {}
        total_columns = 0

        data = graph.get_edge_data(table1, table2)
        for join_clause in data.get("label").split(";"):
            joinable_relations.append(join_clause.strip())
            join_keys_count += 1
            columns = join_clause.split(" = ")
            table_name = columns[0].strip().split(".")[0]
            col_name = columns[0].strip().split(".")[1]
            pos = graph.nodes[table_name]["column_names"].index(col_name)
            fk_type = graph.nodes[table_name]["column_types"][pos]
            joinable_keys_types[fk_type] = joinable_keys_types.get(fk_type, 0) + 1

        table_info = [
            {
                "table": table,
                "column_names": graph.nodes[table]["column_names"],
                "column_types": graph.nodes[table]["column_types"],
            }
            for table in [table1, table2]
        ]

        total_columns = sum(
            len(graph.nodes[table]["column_names"]) for table in [table1, table2]
        )
        num_output_columns = total_columns - 1
        query_info = {
            "from": [table1, table2],
            "tables": table_info,
            "joinable_relations": joinable_relations,
            "num_tables": 2,
            "num_joinable_keys": join_keys_count,
            "joinable_keys_types": joinable_keys_types,
            "num_output_columns": num_output_columns,
            "pattern": ["A", "B"],
        }
        queries.append(query_info)

    return queries


def generate_multiple_table_queries(
    graph: nx.Graph, pattern: nx.Graph
) -> Dict[str, Any]:
    """
    Generates query templates for multi-table queries based on the graph and pattern.

    Args:
        graph (nx.Graph): The graph data representing the schema.
        pattern (nx.Graph): The pattern to match in the graph.

    Returns:
        Dict[str, Any]: A dictionary with the query information for multi-table queries.
    """
    edges = list(graph.edges(data=True))
    tables = set()
    joinable_relations = []
    join_keys_count = 0
    joinable_keys_types = {}
    total_columns = 0
    for edge in edges:
        table1, table2 = edge[:2]
        tables.add(table1)
        tables.add(table2)
        data = graph.get_edge_data(table1, table2)
        for join_clause in data.get("label").split(";"):
            joinable_relations.append(join_clause.strip())
            join_keys_count += 1
            columns = join_clause.split(" = ")
            table_name = columns[0].strip().split(".")[0]
            col_name = columns[0].strip().split(".")[1]
            pos = graph.nodes[table_name]["column_names"].index(col_name)
            fk_type = graph.nodes[table_name]["column_types"][pos]
            joinable_keys_types[fk_type] = joinable_keys_types.get(fk_type, 0) + 1
    table_info = [
        {
            "table": table,
            "column_names": graph.nodes[table]["column_names"],
            "column_types": graph.nodes[table]["column_types"],
        }
        for table in list(tables)
    ]

    total_columns = sum(
        len(graph.nodes[table]["column_names"]) for table in list(tables)
    )
    num_output_columns = total_columns - len(list(pattern.edges))
    query_info = {
        "from": list(tables),
        "tables": table_info,
        "joinable_relations": joinable_relations,
        "num_tables": len(list(tables)),
        "num_joinable_keys": join_keys_count,
        "joinable_keys_types": joinable_keys_types,
        "num_output_columns": num_output_columns,
        "pattern": list(pattern.edges),
    }

    return query_info


def generate_query_templates(
    graph: nx.Graph, patterns: Dict[int, List[nx.Graph]]
) -> List[Dict[str, Any]]:
    """
    Generates a list of query templates for a given graph.

    Args:
        graph (nx.Graph): The graph data representing the schema.
        patterns (Dict[int, List[nx.Graph]]): The loaded pattern data.

    Returns:
        List[Dict[str, Any]]: A list of generated query templates.
    """
    query_templates = []
    for i in range(1, 8):
        if i == 1:
            query_templates.extend(generate_single_table_queries(graph))
        elif i == 2:
            query_templates.extend(generate_two_table_queries(graph))
        else:
            pattern_elements = patterns.get(i)
            for subgraph_nodes in combinations(graph.nodes, i):
                subgraph = graph.subgraph(subgraph_nodes)
                for pattern in pattern_elements:
                    GM = GraphMatcher(subgraph, pattern)
                    if GM.is_isomorphic():
                        query_info = generate_multiple_table_queries(subgraph, pattern)
                        query_templates.append(query_info)

    return query_templates


def process_graph(db_id: str, graph: nx.Graph, graph_folder_path: str) -> None:
    """
    Processes the graph and generates query templates.

    Args:
        db_id (str): The database identifier.
        graph (nx.Graph): The graph representing the schema.
        graph_folder_path (str): Path to the folder where the templates will be stored.
    """
    query_templates = generate_query_templates(graph, load_patterns("patterns"))
    templates_dir = os.path.join(graph_folder_path, "templates")
    output_file = os.path.join(templates_dir, f"{db_id}_query_templates.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(query_templates, file, indent=4)
    logging.info(f"Processed graph for db_id {db_id} and saved query templates.")


def process(graph_folder_path: str) -> None:
    """
    Processes all graphs in the specified folder.

    Args:
        graph_folder_path (str): Path to the folder containing graph data.
    """
    graphs = load_graph(graph_folder_path)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_graph, db_id, graph, graph_folder_path)
            for db_id, graph in graphs.items()
        ]
        for future in futures:
            future.result()

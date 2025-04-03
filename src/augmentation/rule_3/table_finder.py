import pickle
import re
from pathlib import Path

import networkx as nx


def find_central_table_and_components(subgraph, schema):
    """Find the central table in a subgraph and extract its components."""
    if subgraph.number_of_nodes() < 3:
        return "", []

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(subgraph)
    max_betweenness = max(betweenness_centrality.values(), default=0)

    # Get candidate tables
    candidates = [node for node, score in betweenness_centrality.items() if score == max_betweenness]

    if not candidates:
        return "", []

    # Resolve ties using schema centrality
    if len(candidates) > 1:
        schema_centrality = nx.degree_centrality(schema)
        central_table = max(candidates, key=lambda node: schema_centrality.get(node, 0))
    else:
        central_table = candidates[0]

    # Extract components from table name
    components = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|\d+[a-zA-Z]*', central_table)
    components = [comp.lower() for comp in components if comp != '_']

    return central_table, components


def process_subgraphs(subgraphs, schema):
    """Find central tables, extract components, and filter equivalent queries."""
    kept_subgraphs = []

    for subgraph in subgraphs:
        nx_subgraph = subgraph.get("subgraph")
        if not nx_subgraph:
            continue

        central_table, components = find_central_table_and_components(nx_subgraph, schema)
        if not central_table:
            continue

        # Assign central table and components
        subgraph["central_table"] = central_table
        subgraph["components"] = components
        subgraph['equivalent_queries'] = [
            query for query in subgraph['equivalent_queries']
            if any(comp in query['question'].lower() for comp in components)
        ]
        if subgraph['equivalent_queries']:  # Only keep if queries remain
            kept_subgraphs.append(subgraph)

    return kept_subgraphs


def process_dataset(dataset_subgraphs):
    """Process train/dev/test datasets by loading schemas and filtering subgraphs."""
    processed_subgraphs = {}

    for db_id, subgraphs in dataset_subgraphs.items():
        schema_path = Path(f'data/graph_data/bird_graphs/pickles/{db_id}_graph.pkl')

        if not schema_path.exists():
            print(f"Warning: Missing schema file for {db_id}")
            continue

        with schema_path.open('rb') as f:
            schema = pickle.load(f)

        processed_subgraphs[db_id] = process_subgraphs(subgraphs, schema)

    return processed_subgraphs

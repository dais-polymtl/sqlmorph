import pickle
import re
from pathlib import Path

import networkx as nx


def find_central_table_and_components(jqg, schema):
    """Find the central table in a subgraph and extract its components."""
    if jqg.number_of_nodes() < 3:
        return "", []

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(jqg)
    max_betweenness = max(betweenness_centrality.values(), default=0)

    # Get candidate tables
    candidates = [
        node
        for node, score in betweenness_centrality.items()
        if score == max_betweenness
    ]

    if not candidates:
        return "", []

    schema_lookup = {node.lower(): node for node in schema.nodes()}

    mapped_candidates = [
        schema_lookup[node[0].lower()]
        for node in candidates
        if node[0].lower() in schema_lookup
    ]

    if not mapped_candidates:
        return "", []

    # Resolve ties using schema centrality
    if len(mapped_candidates) > 1:
        schema_centrality = nx.degree_centrality(schema)
        central_table = max(
            mapped_candidates, key=lambda node: schema_centrality.get(node, 0)
        )
    else:
        central_table = mapped_candidates[0]

    # Extract components from table name
    components = re.findall(
        r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|\d+[a-zA-Z]*", central_table
    )
    components = [comp.lower() for comp in components if comp != "_"]

    return central_table, components


def retrieve_linker_table(jqg, schema):
    """Find central tables, extract components, and filter equivalent queries."""
    jqg_copy = jqg.copy()
    central_table, components = find_central_table_and_components(
        jqg["jq_graph"], schema
    )
    if not central_table:
        return None

    else:
        # Assign central table and components
        jqg_copy["central_table"] = central_table
        jqg_copy["components"] = components
        for comp in components:
            if comp in jqg["question"].lower():
                return jqg_copy

    return None


def process_dataset(dataset_jqgs):
    """Process train/dev/test datasets by loading schemas and filtering subgraphs."""
    dataset_jqgs_w_lt = []

    for jqg in dataset_jqgs:
        db_id = jqg["db_id"]
        schema_path = Path(f"data/graph_data/bird_graphs/pickles/{db_id}_graph.pkl")

        if not schema_path.exists():
            print(f"Warning: Missing schema file for {db_id}")
            continue

        with schema_path.open("rb") as f:
            schema = pickle.load(f)

        new_jqg = retrieve_linker_table(jqg, schema)
        if new_jqg is not None:
            dataset_jqgs_w_lt.append(new_jqg)

    return dataset_jqgs_w_lt

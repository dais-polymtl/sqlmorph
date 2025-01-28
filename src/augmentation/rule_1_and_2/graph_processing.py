import networkx as nx
from itertools import combinations


def find_candidate_table(schema, subgraph):
    """
    Find candidate tables for extending the subgraph.
    """
    subgraph_nodes = set(subgraph.nodes())
    candidate_tables = {}

    for node in schema.nodes():
        if node not in subgraph_nodes:
            connections_to_subgraph = list(filter(lambda n: n in subgraph_nodes, schema.neighbors(node)))
            if connections_to_subgraph:
                candidate_tables[node] = connections_to_subgraph

    if not candidate_tables:
        return []

    def sort_key(item):
        table, connections = item
        num_connections = len(connections)
        centrality = nx.degree_centrality(schema).get(table, 0)
        return (-num_connections, -centrality)

    return sorted(candidate_tables.items(), key=sort_key)


def is_pattern_in_list(pattern, pattern_list):
    """
    Check if a pattern or its isomorphic equivalent exists in the given list.
    """
    for existing_pattern in pattern_list:
        if nx.is_isomorphic(pattern, existing_pattern):
            return True
    return False


def generate_connected_subgraphs(schema, nodes):
    """
    Generate all connected subgraphs from a given set of nodes.
    """
    edges = list(combinations(nodes, 2))
    valid_edges = [edge for edge in edges if schema.has_edge(*edge)]
    subgraphs = []

    for i in range(1, len(valid_edges) + 1):
        for edge_subset in combinations(valid_edges, i):
            subgraph = nx.Graph()
            subgraph.add_nodes_from(nodes)
            subgraph.add_edges_from(edge_subset)
            if nx.is_connected(subgraph):
                subgraphs.append(subgraph)

    return subgraphs


def extend_and_filter_subgraphs(pre_rule_subgraphs, schema):
    """
    Extend subgraphs based on rules and filter unique patterns.

    Args:
        pre_rule_subgraphs (list): List of initial subgraphs.
        schema (nx.Graph): The main schema graph.

    Returns:
        tuple: (rule_1_patterns, rule_2_patterns)
    """
    unique_patterns = []  # Store all unique patterns across iterations

    # Step 1: Extend each subgraph and filter for unique patterns
    for subgraph in pre_rule_subgraphs:
        candidate_tables = find_candidate_table(schema, subgraph)

        for candidate_table, connections in candidate_tables:
            # Create the extended subgraph
            extended_subgraph = subgraph.copy()
            extended_subgraph.add_node(candidate_table)
            extended_subgraph.add_edges_from([(candidate_table, conn) for conn in connections])

            # Check isomorphic conditions
            if not is_pattern_in_list(extended_subgraph, pre_rule_subgraphs):
                equivalent_patterns = generate_connected_subgraphs(schema, list(extended_subgraph.nodes()))
                valid_candidates = [
                    pattern for pattern in [extended_subgraph] + equivalent_patterns
                    if not is_pattern_in_list(pattern, pre_rule_subgraphs)
                ]

                # If valid candidates exist, choose the one with the fewest edges
                if valid_candidates:
                    chosen_pattern = min(valid_candidates, key=lambda p: len(p.edges()))
                    if not is_pattern_in_list(chosen_pattern, unique_patterns):
                        unique_patterns.append(chosen_pattern)

    # Step 2: Divide the unique patterns into Rule 1 and Rule 2
    rule_1_patterns = []
    rule_2_patterns = []
    if pre_rule_subgraphs:
        max_nodes = max(len(subgraph.nodes()) for subgraph in pre_rule_subgraphs)
    else:
        max_nodes = 0  # Handle empty pre_rule_subgraphs

    for pattern in unique_patterns:
        if len(pattern.nodes()) == max_nodes + 1:
            rule_1_patterns.append(pattern)
        elif 2 <= len(pattern.nodes()) <= max_nodes:
            rule_2_patterns.append(pattern)

    # Step 3: Sort both lists by number of nodes
    rule_1_patterns.sort(key=lambda p: len(p.nodes()))
    rule_2_patterns.sort(key=lambda p: len(p.nodes()))

    return rule_1_patterns, rule_2_patterns


def calculate_subgraph_centrality(subgraph, centrality_dict):
    """
    Calculate the average centrality for a given subgraph.
    """
    subgraph_nodes = list(subgraph.nodes())
    return sum(centrality_dict.get(node, 0) for node in subgraph_nodes) / len(subgraph_nodes)


def is_cyclic(subgraph):
    """
    Check if a given subgraph contains a cycle.
    """
    try:
        nx.find_cycle(subgraph)
        return True
    except nx.NetworkXNoCycle:
        return False

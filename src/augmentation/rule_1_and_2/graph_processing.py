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
            connections_to_subgraph = list(
                filter(lambda n: n in subgraph_nodes, schema.neighbors(node))
            )
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


def is_pattern_in_old_list(pattern, pattern_list):
    """
    Check if a pattern or its isomorphic equivalent exists in the given list.
    """
    
    for existing_pattern in pattern_list:
        if nx.is_isomorphic(pattern, existing_pattern['subgraph']):
        # if nx.is_isomorphic(pattern, existing_pattern):
            return True
    return False

def is_pattern_in_new_list(pattern, pattern_list):
    """
    Check if a pattern or its isomorphic equivalent exists in the given list.
    """
    for existing_pattern in pattern_list:
        if nx.is_isomorphic(pattern, existing_pattern['extended_subgraph']):
        # if nx.is_isomorphic(pattern, existing_pattern):
            return True
    return False

def graph_to_signature(graph):
    nodes = frozenset(sorted(node.lower() for node in graph.nodes()))
    edges = frozenset(
        tuple(sorted((u.lower(), v.lower()))) for u, v in graph.edges()
    )
    labels = frozenset(
        sorted(
            part.strip()
            for label in nx.get_edge_attributes(graph, "label").values()
            for condition in label.split(";")
            for part in condition.split("=")
        )
    )
    return nodes, edges, labels


def extend_and_filter_subgraphs(pre_rule_subgraphs, schema):
    """
    Extend subgraphs based on rules and filter unique patterns.

    Args:
        pre_rule_subgraphs (list): List of initial subgraphs.
        schema (nx.Graph): The main schema graph.

    Returns:
        tuple: (rule_1_patterns, rule_2_patterns, r1_before, r2_before, r1_after, r2_after)
    """
    unique_patterns = []  # Store all unique patterns across iterations
    unique_extended_graphs = []  # Keep track of extended graphs with no duplicates
    unique_signatures = set()

    # Include signatures of the initial subgraphs
    # Step 1: Extend each subgraph and filter for unique patterns
    for subgraph in pre_rule_subgraphs:
        candidate_tables = find_candidate_table(schema, subgraph['subgraph'])
        # candidate_tables = find_candidate_table(schema, subgraph)
        
        for candidate_table, connections in candidate_tables:
            # Create the extended subgraph
            extended_subgraph = subgraph['subgraph'].copy()
            # extended_subgraph = subgraph.copy()

            # Batch update all existing edges to blue
            nx.set_edge_attributes(extended_subgraph, {edge: {"color": "blue"} for edge in extended_subgraph.edges()})
            for conn in connections:
                extended_subgraph.add_edge(candidate_table, conn, color="red")
                extended_subgraph[candidate_table][conn]["label"] = schema.get_edge_data(candidate_table, conn)["label"]
            
                introduced_cycles = [cycle for cycle in nx.simple_cycles(extended_subgraph) if candidate_table in cycle]
                is_redundant = False
                for cycle in introduced_cycles:
                    cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle) - 1)] + [(cycle[-1], cycle[0])]
                    if (candidate_table, conn) not in cycle_edges and (conn, candidate_table) not in cycle_edges:
                        continue

                    cycle_edge_labels = {edge: extended_subgraph.get_edge_data(*edge)["label"] for edge in cycle_edges}

                    extra_edge = (candidate_table, conn) if (candidate_table, conn) in cycle_edge_labels else (conn, candidate_table)
                    multiple_labels = cycle_edge_labels[extra_edge].split(";")

                    for l in multiple_labels:
                        cycle_edge_labels[extra_edge] = l.strip()
                        join_keys = set()
                        for (t1, t2), condition in cycle_edge_labels.items():
                            if not((t1, t2) == extra_edge):
                                for cond in condition.split(";"):
                                    left, right = cond.strip().split("=")
                                    left, right = left.strip(), right.strip()
                                    join_keys.add(left)
                                    join_keys.add(right)
                        extra_condition = cycle_edge_labels[extra_edge]
                        left, right = extra_condition.split("=")
                        left, right = left.strip(), right.strip()
                        if left in join_keys or right in join_keys:
                            is_redundant = True

                        if not is_redundant:
                            extended_subgraph[candidate_table][conn]["label"] = l
                            break

                    if is_redundant:
                        break

                if is_redundant:
                    extended_subgraph.remove_edge(candidate_table, conn)
                        
            extended_signature = graph_to_signature(extended_subgraph)
            if extended_signature not in unique_signatures:
                unique_extended_graphs.append(extended_subgraph)
                unique_signatures.add(extended_signature)
                
            if not is_pattern_in_old_list(extended_subgraph, pre_rule_subgraphs):
                if not is_pattern_in_new_list(extended_subgraph, unique_patterns):
                    new_subgraph = subgraph.copy()
                    new_subgraph['extended_subgraph'] = extended_subgraph
                    unique_patterns.append(new_subgraph)


    max_nodes = (
        max(len(sg['subgraph'].nodes()) for sg in pre_rule_subgraphs) if pre_rule_subgraphs else 0
    )
    # max_nodes = (
    #     max(len(sg.nodes()) for sg in pre_rule_subgraphs) if pre_rule_subgraphs else 0
    # )
    r1_before = 0
    r2_before = 0
    for subgraph in unique_extended_graphs:
        if len(subgraph.nodes()) == max_nodes + 1:
            r1_before += 1
        elif 2 <= len(subgraph.nodes()) <= max_nodes:
            r2_before += 1

    r1_dev_before = 0
    r2_dev_before = 0
    for subgraph in pre_rule_subgraphs:
        r1_dev_before += len(subgraph['equivalent_queries'])
        r2_dev_before += len(subgraph['equivalent_queries'])

    # Step 2: Divide the unique patterns into Rule 1 and Rule 2
    rule_1_patterns = []
    rule_2_patterns = []

    for pattern in unique_patterns:
        if len(pattern['extended_subgraph'].nodes()) == max_nodes + 1:
        # if len(pattern.nodes()) == max_nodes + 1:
            rule_1_patterns.append(pattern)
        elif 2 <= len(pattern['extended_subgraph'].nodes()) <= max_nodes:
        # elif 2 <= len(pattern.nodes()) <= max_nodes:
            rule_2_patterns.append(pattern)

    # Step 3: Sort both lists by number of nodes
    rule_1_patterns.sort(key=lambda p: len(p['extended_subgraph'].nodes()))
    rule_2_patterns.sort(key=lambda p: len(p['extended_subgraph'].nodes()))
    # rule_1_patterns.sort(key=lambda p: len(p.nodes()))
    # rule_2_patterns.sort(key=lambda p: len(p.nodes()))
    # for pattern in rule_2_patterns:
    #     if set(pattern.nodes()) == {"badges", "users", "posts", "votes"}:
    #         print("Pattern: ", pattern.edges(data=True))
    #         print()

    r1_after = len(rule_1_patterns)
    r2_after = len(rule_2_patterns)

    r1_dev_after = 0
    r2_dev_after = 0
    for pattern in rule_1_patterns:
        r1_dev_after += len(pattern['equivalent_queries'])
    for pattern in rule_2_patterns:
        r2_dev_after += len(pattern['equivalent_queries'])

    return rule_1_patterns, rule_2_patterns, r1_before, r2_before, r1_after, r2_after, r1_dev_before, r2_dev_before, r1_dev_after, r2_dev_after


def calculate_subgraph_centrality(subgraph, centrality_dict):
    """
    Calculate the average centrality for a given subgraph.
    """
    subgraph_nodes = list(subgraph.nodes())
    return sum(centrality_dict.get(node, 0) for node in subgraph_nodes) / len(
        subgraph_nodes
    )


def is_cyclic(subgraph):
    """
    Check if a given subgraph contains a cycle.
    """
    try:
        nx.find_cycle(subgraph)
        return True
    except nx.NetworkXNoCycle:
        return False

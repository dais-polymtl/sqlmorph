import networkx as nx
import sys
from itertools import combinations, product
from jq_graph_augmentation.query_generation import (
    translate_graph_into_query,
    extend_old_query,
)
from jq_graph_augmentation.query_execution import (
    execute_new_queries,
    execute_extended_queries,
    add_values_to_translated_queries,
)
from copy import deepcopy


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
        if nx.is_isomorphic(pattern, existing_pattern["subgraph"]):
            return True
    return False


def is_pattern_in_new_list(pattern, pattern_list):
    """
    Check if a pattern or its isomorphic equivalent exists in the given list.
    """
    for existing_pattern in pattern_list:
        if nx.is_isomorphic(pattern, existing_pattern["extended_subgraph"]):
            return True
    return False


def graph_to_signature(graph):
    nodes = frozenset(sorted(node.lower() for node in graph.nodes()))
    edges = frozenset(tuple(sorted((u.lower(), v.lower()))) for u, v in graph.edges())
    labels = frozenset(
        sorted(
            part.strip()
            for label in nx.get_edge_attributes(graph, "label").values()
            for condition in label.split(";")
            for part in condition.split("=")
        )
    )
    return nodes, edges, labels


def extend_and_filter_subgraphs(
    pre_rule_subgraphs,
    db_id,
    df,
    schema,
    adapter,
    logger,
    query_first,
    graph_first,
    n_tables,
    mode,
):
    """
    Extend subgraphs based on rules and filter unique patterns.

    Args:
        pre_rule_subgraphs (list): List of initial subgraphs.
        schema (nx.Graph): The main schema graph.

    Returns:
        tuple: (rule_1_excluded, rule_2_excluded, rule_1_pruned, rule_2_pruned)
    """
    pruned_extensions = []
    excluded_extensions = []
    max_nodes = max(
        (len(sg["subgraph"].nodes()) for sg in pre_rule_subgraphs), default=0
    )
    if max_nodes + 1 != n_tables:
        logger.log(
            "error",
            "Mismatch in table count. Aborting.",
            {
                "db_id": db_id,
                "expected_tables": n_tables,
                "actual_tables (+1)": max_nodes + 1,
            },
        )
        sys.exit(1)
    if mode == "n":
        # only keep subgraphs with max_tables
        pre_rule_subgraphs = [
            sg for sg in pre_rule_subgraphs if len(sg["subgraph"].nodes()) == max_nodes
        ]
    elif mode == "lt_n":
        # only keep subgraphs with max_tables - 1
        pre_rule_subgraphs = [
            sg
            for sg in pre_rule_subgraphs
            if len(sg["subgraph"].nodes()) == max_nodes - 1
        ]

    for subgraph in pre_rule_subgraphs:
        candidate_tables = find_candidate_table(schema, subgraph["subgraph"])

        for candidate_table, connections in candidate_tables:
            all_candidate_edges = [(candidate_table, conn) for conn in connections]

            for size in range(len(all_candidate_edges), 0, -1):
                for edge_combination in combinations(all_candidate_edges, size):
                    temp_subgraph = subgraph["subgraph"].copy()
                    nx.set_edge_attributes(
                        temp_subgraph,
                        {edge: {"color": "blue"} for edge in temp_subgraph.edges()},
                    )

                    for src, dst in edge_combination:
                        temp_subgraph.add_edge(src, dst, color="red")
                        temp_subgraph[src][dst]["label"] = schema.get_edge_data(
                            src, dst
                        )["label"]

                    introduced_cycles = [
                        cycle
                        for cycle in nx.simple_cycles(temp_subgraph)
                        if candidate_table in cycle
                    ]
                    is_redundant = False

                    for cycle in introduced_cycles:
                        cycle_edges = [
                            (cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)
                        ] + [(cycle[-1], cycle[0])]
                        cycle_edge_labels = {
                            edge: temp_subgraph.get_edge_data(*edge)["label"]
                            for edge in cycle_edges
                        }

                        extra_edges_caused_cycle = [
                            edge
                            for edge in cycle_edges
                            if edge in edge_combination
                            or (edge[1], edge[0]) in edge_combination
                        ]
                        if not extra_edges_caused_cycle:
                            continue

                        all_label_list = [
                            cycle_edge_labels.get(edge, "").split(";")
                            for edge in extra_edges_caused_cycle
                        ]
                        label_combinations = list(product(*all_label_list))

                        join_keys = {
                            key.strip().lower()
                            for (t1, t2), condition in cycle_edge_labels.items()
                            for cond in condition.split(";")
                            if (t1, t2) not in extra_edges_caused_cycle
                            for key in cond.strip().split("=")
                        }

                        for label_set in label_combinations:
                            for i, label in enumerate(label_set):
                                edge = extra_edges_caused_cycle[i]
                                left, right = map(str.strip, label.split("="))

                                if i != len(label_set) - 1:
                                    join_keys.update([left.lower(), right.lower()])
                                    temp_subgraph[edge[0]][edge[1]][
                                        "label"
                                    ] = label.strip()

                                elif (
                                    left.lower() in join_keys
                                    and right.lower() in join_keys
                                ):
                                    is_redundant = True

                                else:
                                    temp_subgraph[edge[0]][edge[1]][
                                        "label"
                                    ] = label.strip()
                                    is_redundant = False

                            if not is_redundant:
                                break

                        if is_redundant:
                            break

                    if not is_redundant:
                        extended_subgraph = temp_subgraph
                        break

                if not is_redundant:
                    break

            new_subgraph = subgraph.copy()
            new_subgraph["extended_subgraph"] = extended_subgraph
            new_subgraph["db_id"] = db_id

            new_queries = translate_graph_into_query(
                pattern=new_subgraph["extended_subgraph"],
                db_id=db_id,
                df=df,
                schema=schema,
            )
            add_values_to_translated_queries(new_queries, adapter)

            extended_old_queries = deepcopy(extend_old_query(new_subgraph))
            new_queries, new_queries_validity = execute_new_queries(
                new_queries, adapter
            )
            extended_old_queries, extended_queries_validity = execute_extended_queries(
                extended_old_queries, adapter
            )

            if new_queries_validity and extended_queries_validity:
                if graph_first:
                    new_subgraph["graph_first"] = new_queries
                if query_first:
                    new_subgraph["query_first"] = extended_old_queries

            else:
                continue

            if not is_pattern_in_old_list(
                new_subgraph["extended_subgraph"], pre_rule_subgraphs
            ):
                if not is_pattern_in_new_list(
                    new_subgraph["extended_subgraph"], pruned_extensions
                ):
                    pruned_extensions.append(new_subgraph)
            else:
                excluded_extensions.append(new_subgraph)

    rule_1_pruned_extensions = [
        extension
        for extension in pruned_extensions
        if len(extension["extended_subgraph"].nodes()) == max_nodes + 1
    ]
    rule_2_pruned_extensions = [
        extension
        for extension in pruned_extensions
        if 2 <= len(extension["extended_subgraph"].nodes()) <= max_nodes
    ]
    rule_1_excluded_extensions = [
        extension
        for extension in excluded_extensions
        if len(extension["extended_subgraph"].nodes()) == max_nodes + 1
    ]
    rule_2_excluded_extensions = [
        extension
        for extension in excluded_extensions
        if 2 <= len(extension["extended_subgraph"].nodes()) <= max_nodes
    ]

    rule_1_pruned_extensions.sort(key=lambda p: len(p["extended_subgraph"].nodes()))
    rule_1_excluded_extensions.sort(key=lambda p: len(p["extended_subgraph"].nodes()))
    rule_2_pruned_extensions.sort(key=lambda p: len(p["extended_subgraph"].nodes()))
    rule_2_excluded_extensions.sort(key=lambda p: len(p["extended_subgraph"].nodes()))

    return (
        rule_1_pruned_extensions,
        rule_2_pruned_extensions,
        rule_1_excluded_extensions,
        rule_2_excluded_extensions,
    )


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

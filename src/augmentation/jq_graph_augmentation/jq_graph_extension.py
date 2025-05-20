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


def check_cycle_redundancy(temp_subgraph, candidate_table, edge_combination):
    """
    Check if adding the candidate table with the given edge combination introduces a redundant cycle.

    A cycle is redundant if all join keys in the last edge of a cycle are already present
    in the join keys of other edges in the cycle.

    Args:
        temp_subgraph (nx.Graph): The subgraph with the candidate table and edges
        candidate_table (str): The table being added
        edge_combination (list): List of edges being added

    Returns:
        bool: True if any cycle introduced is redundant, False otherwise
    """
    # Find cycles that include the candidate table
    introduced_cycles = [
        cycle for cycle in nx.simple_cycles(temp_subgraph) if candidate_table in cycle
    ]

    for cycle in introduced_cycles:
        # Extract cycle edges (each edge connects consecutive nodes in the cycle)
        cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)] + [
            (cycle[-1], cycle[0])
        ]

        # Get labels for each edge in the cycle
        cycle_edge_labels = {
            edge: temp_subgraph.get_edge_data(*edge)["label"] for edge in cycle_edges
        }

        # Identify which cycle edges are from our new edge_combination
        extra_edges_caused_cycle = [
            edge
            for edge in cycle_edges
            if edge in edge_combination or (edge[1], edge[0]) in edge_combination
        ]

        # Skip cycles without any edge from the edge_combination
        if not extra_edges_caused_cycle:
            continue

        # Get all possible label combinations for the extra edges
        all_label_list = [
            cycle_edge_labels.get(edge, "").split(";")
            for edge in extra_edges_caused_cycle
        ]
        label_combinations = list(product(*all_label_list))

        # Collect join keys from existing edges in the cycle
        join_keys = {
            key.strip().lower()
            for (t1, t2), condition in cycle_edge_labels.items()
            for cond in condition.split(";")
            if (t1, t2) not in extra_edges_caused_cycle
            for key in cond.strip().split("=")
        }

        # Check each label combination for redundancy
        for label_set in label_combinations:
            is_redundant = False
            # Use a copy to avoid modifying the original set during iterations
            temp_join_keys = join_keys.copy()

            for i, label in enumerate(label_set):
                edge = extra_edges_caused_cycle[i]
                left, right = map(str.strip, label.split("="))

                # For all but the last edge, add its keys to the join keys
                if i != len(label_set) - 1:
                    temp_join_keys.update([left.lower(), right.lower()])
                    temp_subgraph[edge[0]][edge[1]]["label"] = label.strip()

                # For the last edge, check if both keys are already in the join keys
                elif left.lower() in temp_join_keys and right.lower() in temp_join_keys:
                    is_redundant = True

                # If not, the cycle is not redundant
                else:
                    temp_subgraph[edge[0]][edge[1]]["label"] = label.strip()
                    is_redundant = False

            # If this combination is not redundant, we can stop checking
            if not is_redundant:
                break

        # If all combinations for this cycle are redundant, return True
        if is_redundant:
            return True

    # No redundant cycles found
    return False


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
        db_id (str): Database identifier.
        df: Database data.
        schema (nx.Graph): The database schema graph.
        adapter: Database adapter for query execution.
        logger: Logger instance.
        query_first (bool): Whether to use query-first approach.
        graph_first (bool): Whether to use graph-first approach.
        n_tables (int): Expected number of tables.
        mode (str): Mode for filtering subgraphs ("n", "lt_n", or other).

    Returns:
        tuple: (rule_1_pruned, rule_2_pruned, rule_1_excluded, rule_2_excluded)
    """
    pruned_extensions = []
    excluded_extensions = []

    # Find the maximum number of nodes in any subgraph
    max_nodes = max(
        (len(sg["subgraph"].nodes()) for sg in pre_rule_subgraphs), default=0
    )

    # Validate table count
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

    # Filter subgraphs based on mode
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

    # Process each subgraph
    for subgraph in pre_rule_subgraphs:
        # Find potential tables to add to the subgraph
        candidate_tables = find_candidate_table(schema, subgraph["subgraph"])

        # For each candidate table, try to add it to the subgraph
        for candidate_table, connections in candidate_tables:
            all_candidate_edges = [(candidate_table, conn) for conn in connections]
            extended_subgraph = None

            # Try different numbers of edges (start with max, reduce if needed)
            for size in range(len(all_candidate_edges), 0, -1):
                # For each possible combination of edges of this size
                for edge_combination in combinations(all_candidate_edges, size):
                    # Create a copy of the subgraph and add the new edges
                    temp_subgraph = subgraph["subgraph"].copy()

                    # Mark existing edges as blue
                    nx.set_edge_attributes(
                        temp_subgraph,
                        {edge: {"color": "blue"} for edge in temp_subgraph.edges()},
                    )

                    # Add new edges and mark them as red
                    for src, dst in edge_combination:
                        temp_subgraph.add_edge(src, dst, color="red")
                        temp_subgraph[src][dst]["label"] = schema.get_edge_data(
                            src, dst
                        )["label"]

                    # Check if extension introduces a redundant cycle
                    is_redundant = check_cycle_redundancy(
                        temp_subgraph, candidate_table, edge_combination
                    )

                    if not is_redundant:
                        extended_subgraph = temp_subgraph
                        break

                if extended_subgraph:
                    break

            # Skip if we couldn't find a non-redundant extension
            if not extended_subgraph:
                continue

            # Create new subgraph with the extension
            new_subgraph = subgraph.copy()
            new_subgraph["extended_subgraph"] = extended_subgraph
            new_subgraph["db_id"] = db_id

            # Generate queries for the extended subgraph
            new_queries = translate_graph_into_query(
                pattern=new_subgraph["extended_subgraph"],
                db_id=db_id,
                df=df,
                schema=schema,
            )
            add_values_to_translated_queries(new_queries, adapter)

            extended_old_queries = extend_old_query(new_subgraph)

            # Execute the queries
            new_queries, new_queries_validity = execute_new_queries(
                new_queries, adapter
            )
            extended_old_queries, extended_queries_validity = execute_extended_queries(
                extended_old_queries, adapter
            )

            # Skip if queries are not valid
            if not (new_queries_validity and extended_queries_validity):
                continue

            # Store query results based on the approach
            if graph_first:
                new_subgraph["graph_first"] = new_queries
            if query_first:
                new_subgraph["query_first"] = extended_old_queries

            # Categorize the extension based on uniqueness
            if not is_pattern_in_old_list(
                new_subgraph["extended_subgraph"], pre_rule_subgraphs
            ):
                if not is_pattern_in_new_list(
                    new_subgraph["extended_subgraph"], pruned_extensions
                ):
                    pruned_extensions.append(new_subgraph)
            else:
                excluded_extensions.append(new_subgraph)

    # Categorize and sort extensions based on the number of nodes
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

    # Sort the extensions by node count
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

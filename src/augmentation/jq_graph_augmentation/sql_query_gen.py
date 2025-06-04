import networkx as nx
import os
import pandas as pd
from itertools import combinations, product
from query_generation import (
    translate_graph_into_query,
    extend_old_query,
)
from data_retrieval import (
    retrieve_all_dev_subgraphs,
    load_schema,
)
from query_execution import (
    execute_new_queries,
    execute_extended_queries,
    add_values_to_translated_queries,
)
from src.core.logger.logger import Logger

logger = Logger(__name__)


def find_candidate_table(schema, subgraph):
    """
    Find candidate tables for extending the subgraph.
    """
    subgraph_nodes = set(subgraph.nodes())
    candidate_tables = {}  # turn into array, return sorted[]

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

    # move out the sorting function such that, generate the possible combinations for each extension, sort them, do your redundancy check
    # do the actual extension to generate sql, then possibly graph-first

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
        if nx.is_isomorphic(pattern, existing_pattern):
            return True
    return False


def canonical_form(candidate_table, edge_combination):
    normalized_edges = []
    for u, v in edge_combination:
        if u <= v:
            normalized_edges.append((u, v))
        else:
            normalized_edges.append((v, u))

    normalized_edges.sort()
    edge_strs = [f"({a}, {b})" for a, b in normalized_edges]
    canonical = candidate_table + ": " + " -- ".join(edge_strs)
    return canonical


def sort_edge_combinations(edge_combinations):
    canon_form = edge_combinations[0]
    num_edges = canon_form.count("(")
    return (-num_edges, canon_form)


def check_cycle_redundancy(temp_subgraph, edge_combination):
    """
    Determines if the added edges create a redundant cycle in the subgraph.

    A cycle is considered redundant if the added edges don't introduce new
    join paths beyond what's already implied by existing join keys.

    Args:
        temp_subgraph (networkx.DiGraph): The subgraph with added edges.
        edge_combination (Tuple): Candidate extension edges and their canonical form.

    Returns:
        bool: True if the cycle is redundant, False otherwise.
    """
    introduced_cycles = [
        cycle
        for cycle in nx.simple_cycles(temp_subgraph)
        if edge_combination[0] in cycle
    ]

    for cycle in introduced_cycles:
        cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)] + [
            (cycle[-1], cycle[0])
        ]

        cycle_edge_labels = {
            edge: temp_subgraph.get_edge_data(*edge)["label"] for edge in cycle_edges
        }

        extra_edges_caused_cycle = [
            edge
            for edge in cycle_edges
            if edge in edge_combination[1] or (edge[1], edge[0]) in edge_combination[1]
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
            is_redundant = False
            for i, label in enumerate(label_set):
                edge = extra_edges_caused_cycle[i]
                left, right = map(str.strip, label.split("="))

                if i != len(label_set) - 1:
                    join_keys.update([left.lower(), right.lower()])
                    temp_subgraph[edge[0]][edge[1]]["label"] = label.strip()
                elif left.lower() in join_keys and right.lower() in join_keys:
                    is_redundant = True
                else:
                    temp_subgraph[edge[0]][edge[1]]["label"] = label.strip()
                    is_redundant = False

            if not is_redundant:
                break

        if is_redundant:
            return True

    return False


def extend_graphs_and_gen_queries(
    pre_aug_subgraphs,
    db_id,
    df,
    schema,
    db_file_path,
    graph_first,
):
    augmented_qf_filtered_queries = []
    augmented_qf_discarded_queries = []
    augmented_gf_filtered_queries = []
    augmented_gf_discarded_queries = []

    kept_ext_subgraphs = []

    aug_fil_queries = {}
    aug_dis_queries = {}

    aug_fil_queries["db_id"] = db_id
    aug_dis_queries["db_id"] = db_id
    for i, subgraph in enumerate(pre_aug_subgraphs):
        candidate_tables = find_candidate_table(schema, subgraph["subgraph"])

        all_edge_combinations = []
        for candidate_table, connections in candidate_tables:
            all_candidate_edges = [(candidate_table, conn) for conn in connections]

            for size in range(len(all_candidate_edges), 0, -1):
                for edge_combination in combinations(all_candidate_edges, size):
                    canon_form = canonical_form(candidate_table, edge_combination)
                    all_edge_combinations.append(
                        (canon_form, (candidate_table, edge_combination))
                    )
            all_edge_combinations.sort(key=sort_edge_combinations)

        seen_extra_node = set()
        for i, (canon_form, edge_combination) in enumerate(all_edge_combinations):

            if canon_form.partition(":")[0].strip() in seen_extra_node:
                continue
            temp_subgraph = subgraph["subgraph"].copy()
            nx.set_edge_attributes(
                temp_subgraph,
                {edge: {"color": "blue"} for edge in temp_subgraph.edges()},
            )

            for src, dst in edge_combination[1]:
                temp_subgraph.add_edge(src, dst, color="red")
                temp_subgraph[src][dst]["label"] = schema.get_edge_data(src, dst)[
                    "label"
                ]
            is_redundant = check_cycle_redundancy(temp_subgraph, edge_combination)

            if not is_redundant:
                extended_subgraph = temp_subgraph
                seen_extra_node.add(canon_form.partition(":")[0].strip())
            else:
                continue

            new_query_first = {}
            new_graph_first = {}
            new_query_first["id"] = "new_" + str(i)
            new_graph_first["id"] = "new_" + str(i)

            extended_old_queries = extend_old_query(
                equivalent_queries=subgraph["equivalent_queries"],
                old_subgraph=subgraph["subgraph"],
                extended_subgraph=extended_subgraph,
            )
            extended_old_queries, extended_queries_validity = execute_extended_queries(
                extended_old_queries, db_file_path
            )
            if extended_queries_validity:
                new_query_first.update(
                    {
                        "ext_id": extended_old_queries[0]["question_id"],
                        "evidence": extended_old_queries[0]["evidence"],
                        "difficulty": extended_old_queries[0]["difficulty"],
                        "SQL": extended_old_queries[0]["new_query"],
                    }
                )

                if graph_first:
                    new_queries = translate_graph_into_query(
                        pattern=extended_subgraph,
                        db_id=db_id,
                        df=df,
                        schema=schema,
                    )
                    add_values_to_translated_queries(new_queries, db_file_path)
                    graph_first_queries = execute_new_queries(new_queries, db_file_path)
                    new_graph_first.update(
                        {
                            "SQL": graph_first_queries["main_query"],
                            "difficulty": "challenging",
                        }
                    )
            else:
                continue

            if is_pattern_in_old_list(
                extended_subgraph, pre_aug_subgraphs
            ) or is_pattern_in_new_list(extended_subgraph, kept_ext_subgraphs):

                augmented_qf_discarded_queries.append(new_query_first)
                augmented_gf_discarded_queries.append(new_graph_first)

            else:
                augmented_qf_filtered_queries.append(new_query_first)
                augmented_gf_filtered_queries.append(new_graph_first)
                kept_ext_subgraphs.append(extended_subgraph)

            aug_fil_queries["queries"] = augmented_qf_filtered_queries
            aug_dis_queries["queries"] = augmented_qf_discarded_queries
            aug_fil_queries["graph_first"] = augmented_gf_filtered_queries
            aug_dis_queries["graph_first"] = augmented_gf_discarded_queries

    return (aug_fil_queries, aug_dis_queries)


def validate_paths(aug_inputs_path, schema_path, db_file_path, query_stats_path):
    """
    Validate that all paths exist.
    """
    required_paths = {
        "aug_inputs_path": aug_inputs_path,
        "schema_path": schema_path,
        "db_file_path": db_file_path,
        "query_stats_path": query_stats_path,
    }

    missing_paths = [
        path for path in required_paths.values() if not os.path.exists(path)
    ]
    if missing_paths:
        logger.log(
            level="error",
            action="Required files for SQL generation are missing.",
            details={"missing_paths": missing_paths},  # <- FIXED
        )
        return False
    return True


def aug_n_table_sql_queries(db_id, num_tables, graph_first=False):
    rule_inputs_base = os.getenv("RULE_INPUTS_BASE")
    graph_data_base = os.getenv("GRAPH_DATA_BASE")
    data_folder = os.getenv("DATA_FOLDER")

    aug_inputs_path = os.path.join(rule_inputs_base, db_id)
    schema_path = os.path.join(graph_data_base, f"{db_id}_graph.pkl")
    db_file_path = os.path.join(
        data_folder, "benchmarks", "Bird", "dev_databases", db_id, f"{db_id}.sqlite"
    )
    query_stats_path = os.path.join(rule_inputs_base, "query_statistics.csv")

    # Check that all necessary files exist
    validate_paths(aug_inputs_path, schema_path, db_file_path, query_stats_path)

    # Load subgraphs
    pre_aug_subgraphs = retrieve_all_dev_subgraphs(aug_inputs_path, num_tables)
    logger.log(
        "info",
        "Pre-augmented subgraphs loaded successfully.",
        {"db_id": db_id, "number_of_subgraphs": len(pre_aug_subgraphs)},
    )

    # Load schema and query statistics
    schema = load_schema(schema_path)
    query_col_stats = pd.read_csv(query_stats_path)

    # Generate extended subgraphs and SQL queries
    filtered_aug, discarded_aug = extend_graphs_and_gen_queries(
        pre_aug_subgraphs,
        db_id,
        query_col_stats,
        schema,
        db_file_path,
        graph_first=graph_first,
    )

    return filtered_aug, discarded_aug

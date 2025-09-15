import networkx as nx
import os
import pandas as pd
from itertools import combinations, product
from .query_generation import (
    translate_graph_into_query,
    extend_old_query,
)
from .data_retrieval import (
    retrieve_n_node_jqgs,
    retrieve_all_dev_jq_graphs,
    load_schema,
)
from .query_execution import (
    execute_new_queries,
    execute_extended_queries,
    add_values_to_translated_queries,
)
from src.core.logger.logger import Logger

logger = Logger(__name__)


def find_candidate_tables(schema_graph, jq_graph):
    """
    Identify candidate tables from the schema graph that are not already in jq_graph,
    but are connected to at least one node in jq_graph (based on table name only).

    Returns:
        A list of (candidate_table, reconstructed_connections), sorted by:
        - number of connections to jq_graph
        - centrality in schema_graph
    """
    jq_nodes = set(jq_graph.nodes())

    table_to_nodes = {}
    for table_name, alias in jq_nodes:
        key = table_name.lower()
        table_to_nodes.setdefault(key, []).append((table_name, alias))

    candidate_tables = []

    for node in schema_graph.nodes():
        node_lower = node.lower()
        if node_lower not in table_to_nodes:
            connections = []
            for neighbor in schema_graph.neighbors(node):
                neighbor_lower = neighbor.lower()
                if neighbor_lower in table_to_nodes:
                    for matched_node in table_to_nodes[neighbor_lower]:
                        connections.append(((node, "et"), matched_node))

            if connections:
                candidate_tables.append((node, connections))

    if not candidate_tables:
        return []

    centrality_scores = nx.degree_centrality(schema_graph)

    def sort_key(item):
        table, connections = item
        return (-len(connections), -centrality_scores.get(table[0], 0))

    return sorted(candidate_tables, key=sort_key)


def is_pattern_in_old_list(pattern, pattern_list):
    """
    Check if a pattern or its isomorphic equivalent exists in the given list.
    """

    for existing_pattern in pattern_list:
        if nx.is_isomorphic(pattern, existing_pattern["jq_graph"]):
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
    def normalize_edge(u, v):
        return (u, v) if u <= v else (v, u)

    def format_node(node):
        if isinstance(node, str):
            return node
        table, alias = node
        return f"{table} AS {alias}" if alias else table

    normalized_edges = sorted([normalize_edge(*edge) for _, edge in edge_combination])

    edge_strs = [f"({format_node(a)}, {format_node(b)})" for a, b in normalized_edges]
    return f"{format_node(candidate_table)}: " + " -- ".join(edge_strs)


def sort_edge_combinations(edge_combinations):
    canon_form = edge_combinations[0]
    num_edges = len(edge_combinations[1])
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
    tables = list(temp_subgraph.nodes())
    # alias_to_node = {node[1].lower(): node for node in temp_subgraph.nodes()}
    canon_node = edge_combination[0]
    added_edges = set(edge_combination[1])

    for cycle in nx.simple_cycles(temp_subgraph):
        if canon_node not in cycle:
            continue

        cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)] + [
            (cycle[-1], cycle[0])
        ]
        cycle_edge_labels = {
            edge: temp_subgraph.get_edge_data(*edge).get("joins", [])
            for edge in cycle_edges
        }

        extra_edges = [
            edge
            for edge in cycle_edges
            if edge in added_edges or (edge[1], edge[0]) in added_edges
        ]

        if not extra_edges:
            continue

        all_label_list = [cycle_edge_labels.get(edge, []) for edge in extra_edges]
        label_combinations = list(product(*all_label_list))

        join_keys = set()
        for (
            edge,
            labels,
        ) in cycle_edge_labels.items():
            if edge in extra_edges or (edge[1], edge[0]) in extra_edges:
                continue
            for label in labels:
                for key in map(str.strip, label.strip().split("=")):
                    table, col = key.lower().split(".", 1)
                    node = next(
                        (
                            (t, a)
                            for t, a in tables
                            if a.lower() == table or t.lower() == table
                        ),
                        (table, ""),
                    )

                    join_keys.add(f"{node[0] or node[1].lower()}.{col}")

        for label_set in label_combinations:
            is_redundant = False
            for i, label in enumerate(label_set):
                edge = extra_edges[i]
                try:
                    left, right = map(str.strip, label.split("="))
                    l_tbl, l_col = left.lower().split(".", 1)
                    r_tbl, r_col = right.lower().split(".", 1)

                    l_node = next(
                        (
                            (t, a)
                            for t, a in tables
                            if a.lower() == l_tbl or t.lower() == l_tbl
                        ),
                        (l_tbl, ""),
                    )
                    r_node = next(
                        (
                            (t, a)
                            for t, a in tables
                            if a.lower() == r_tbl or t.lower() == r_tbl
                        ),
                        (r_tbl, ""),
                    )

                    l_key = f"{l_node[0]}.{l_col}"
                    r_key = f"{r_node[0]}.{r_col}"

                except ValueError:
                    is_redundant = False
                    continue

                if i != len(label_set) - 1:
                    join_keys.update([l_key, r_key])
                    temp_subgraph[edge[0]][edge[1]]["joins"] = [label]
                    is_redundant = False
                elif l_key in join_keys and r_key in join_keys:
                    is_redundant = True
                    continue
                else:
                    temp_subgraph[edge[0]][edge[1]]["joins"] = [label]
                    is_redundant = False
                    break

            if is_redundant:
                return True

    return False

    # introduced_cycles = [
    #     cycle
    #     for cycle in nx.simple_cycles(temp_subgraph)
    #     if edge_combination[0] in cycle
    # ]

    # for cycle in introduced_cycles:
    #     cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)] + [
    #         (cycle[-1], cycle[0])
    #     ]

    #     cycle_edge_labels = {
    #         edge: temp_subgraph.get_edge_data(*edge)["label"] for edge in cycle_edges
    #     }

    #     extra_edges_caused_cycle = [
    #         edge
    #         for edge in cycle_edges
    #         if edge in edge_combination[1] or (edge[1], edge[0]) in edge_combination[1]
    #     ]

    #     if not extra_edges_caused_cycle:
    #         continue

    #     all_label_list = [
    #         cycle_edge_labels.get(edge, [])
    #         for edge in extra_edges_caused_cycle
    #     ]
    #     label_combinations = list(product(*all_label_list))

    #     alias_to_node = {
    #         node[1].lower(): node
    #         for node in temp_subgraph.nodes()
    #     }

    #     join_keys = set()

    #     for (t1, t2), condition in cycle_edge_labels.items():
    #         if (t1, t2) in extra_edges_caused_cycle:
    #             continue

    #         for cond in condition:
    #             cond = cond.strip()
    #             for key in cond.split("="):
    #                 key = key.strip().lower()
    #                 table_part, column_part = key.split(".", 1)
    #                 table_name_w_alias = alias_to_node.get(table_part, table_part)
    #                 join_keys.add(f"{table_name_w_alias[1] or table_name_w_alias[0].lower()}.{column_part.lower()}")

    #     for label_set in label_combinations:
    #         is_redundant = False
    #         for i, label in enumerate(label_set):
    #             edge = extra_edges_caused_cycle[i]
    #             left_table, left_col, right_table, right_col = label.replace("=", ".").split(".")
    #             if i != len(label_set) - 1:
    #                 left = f"{alias_to_node.get(left_table.lower())[0] or left_table.lower()}.{left_col.strip()}"
    #                 right = f"{alias_to_node.get(right_table.lower())[0] or right_table.lower()}.{right_col.strip()}"
    #                 join_keys.update([left.lower(), right.lower()])
    #                 temp_subgraph[edge[0]][edge[1]]["label"] = label.strip()
    #             elif left.lower() in join_keys and right.lower() in join_keys:
    #                 is_redundant = True
    #             else:
    #                 temp_subgraph[edge[0]][edge[1]]["label"] = label.strip()
    #                 is_redundant = False

    #         if not is_redundant:
    #             break

    #     if is_redundant:
    #         return True

    # return False


def make_new_id(ext_id, i):
    ext_id_str = str(ext_id).zfill(4)  # pad to 4 digits
    i_str = str(i).zfill(3)  # pad to 3 digits
    return int(f"11{ext_id_str}{i_str}")


def extend_graphs_and_gen_queries(
    pre_aug_graphs,
    all_dev_jqgs,
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

    filtered_ext_graphs = []
    all_extensions = []

    aug_fil_queries = {"queries": [], "graph_first": []}
    aug_dis_queries = {"queries": [], "graph_first": []}

    aug_fil_queries["db_id"] = db_id
    aug_dis_queries["db_id"] = db_id
    for jq_graph in pre_aug_graphs:
        candidate_tables = find_candidate_tables(schema, jq_graph["jq_graph"])

        all_edge_combinations = []
        for candidate_table, connections in candidate_tables:
            all_candidate_edges = [conn for conn in connections]

            for size in range(len(all_candidate_edges), 0, -1):
                for edge_combination in combinations(all_candidate_edges, size):
                    canon_form = canonical_form(
                        (candidate_table, "et"), edge_combination
                    )
                    all_edge_combinations.append(
                        (canon_form, ((candidate_table, "et"), edge_combination))
                    )
            all_edge_combinations.sort(key=sort_edge_combinations)

        edge_label_cache = {}
        for i, (canon_form, edge_combination) in enumerate(all_edge_combinations):
            temp_graph = jq_graph["jq_graph"].copy()
            nx.set_edge_attributes(
                temp_graph, {e: {"color": "blue"} for e in temp_graph.edges()}
            )
            for src, dst in edge_combination[1]:
                for node in (src, dst):
                    if node not in temp_graph:
                        temp_graph.add_node(node, name=node[0], alias=node[1])

                temp_graph.add_edge(src, dst, color="red")
                schema_node_map = {n.lower(): n for n in schema.nodes}

                schema_labels = edge_label_cache.setdefault(
                    (src[0], dst[0]),
                    [
                        lbl.strip()
                        for lbl in (
                            schema.get_edge_data(
                                schema_node_map.get(src[0].lower(), src[0]),
                                schema_node_map.get(dst[0].lower(), dst[0]),
                            )
                            or {}
                        )
                        .get("label", "")
                        .split(";")
                        if lbl.strip()
                    ],
                )

                temp_graph[src][dst]["joins"] = []

                for lbl in schema_labels:
                    l_tbl, l_col, r_tbl, r_col = tuple(
                        map(str.strip, lbl.replace("=", ".").split("."))
                    )

                    l_tbl_lower, r_tbl_lower = l_tbl.lower(), r_tbl.lower()
                    src_tbl, dst_tbl = src[0].lower(), dst[0].lower()

                    if l_tbl_lower == src_tbl and r_tbl_lower == dst_tbl:
                        join = (
                            f"{(src[1] or l_tbl)}.{l_col} = {(dst[1] or r_tbl)}.{r_col}"
                        )
                    elif r_tbl_lower == src_tbl and l_tbl_lower == dst_tbl:
                        join = (
                            f"{(src[1] or r_tbl)}.{r_col} = {(dst[1] or l_tbl)}.{l_col}"
                        )
                    else:
                        continue
                    temp_graph[src][dst]["joins"].append(join)

            is_redundant = check_cycle_redundancy(temp_graph, edge_combination)

            if not is_redundant:
                extended_graph = temp_graph.copy()
                for _, _, d in extended_graph.edges(data=True):
                    if d.get("color") == "red" and len(d.get("joins", [])) > 1:
                        d["joins"] = [d["joins"][0]]

            else:
                continue

            new_query_first = {}
            new_graph_first = {}
            new_query_first["id"] = make_new_id(jq_graph["question_id"], i)
            new_graph_first["id"] = make_new_id(jq_graph["question_id"], i)

            ext_jqg = extend_old_query(
                jqg=jq_graph,
                extended_graph=extended_graph,
            )
            ext_jqg["ext_jq_graph"] = extended_graph
            valid_ext_jqg, ext_is_valid = execute_extended_queries(
                ext_jqg, db_file_path
            )
            if ext_is_valid:
                new_query_first.update(
                    {
                        "ext_id": valid_ext_jqg["question_id"],
                        "evidence": valid_ext_jqg["evidence"],
                        "difficulty": valid_ext_jqg["difficulty"],
                        "SQL": valid_ext_jqg["new_query"],
                    }
                )

                if graph_first:
                    new_queries = translate_graph_into_query(
                        ext_jqg=extended_graph,
                        db_id=db_id,
                        df=df,
                    )
                    add_values_to_translated_queries(new_queries, db_file_path)
                    graph_first_queries = execute_new_queries(new_queries, db_file_path)
                    new_graph_first.update(
                        {
                            "SQL": graph_first_queries["main_query"],
                            "difficulty": "challenging",
                        }
                    )
                    logged_ext_jqg = {
                        **{
                            k: v
                            for k, v in valid_ext_jqg.items()
                            if k not in {"new_query", "main_query"}
                        },
                        "query_first": valid_ext_jqg["new_query"],
                        "graph_first": graph_first_queries["main_query"],
                    }
            else:
                continue

            if is_pattern_in_old_list(
                extended_graph, all_dev_jqgs
            ) or is_pattern_in_new_list(extended_graph, filtered_ext_graphs):
                if graph_first:
                    all_extensions.append({**logged_ext_jqg, "status": "discarded"})
                augmented_qf_discarded_queries.append(new_query_first)
                augmented_gf_discarded_queries.append(new_graph_first)

            else:
                augmented_qf_filtered_queries.append(new_query_first)
                augmented_gf_filtered_queries.append(new_graph_first)
                if graph_first:
                    all_extensions.append({**logged_ext_jqg, "status": "filtered"})
                filtered_ext_graphs.append(extended_graph)

            aug_fil_queries["queries"] = augmented_qf_filtered_queries
            aug_dis_queries["queries"] = augmented_qf_discarded_queries
            aug_fil_queries["graph_first"] = augmented_gf_filtered_queries
            aug_dis_queries["graph_first"] = augmented_gf_discarded_queries

    return (aug_fil_queries, aug_dis_queries, all_extensions)


# def bootstrap_queries_from_existing(
#     pre_aug_graphs,
#     db_id,
#     df,
#     schema,
#     db_file_path,
#     graph_first,
# ):


#     aug_fil_queries = {"queries": []}
#     aug_dis_queries = {"queries": []}


#     aug_fil_queries["db_id"] = db_id
#     aug_dis_queries["db_id"] = db_id
#     for jq_graph in pre_aug_graphs:
#         candidate_tables = find_candidate_tables(schema, jq_graph["jq_graph"])

#         all_edge_combinations = []
#         for candidate_table, connections in candidate_tables:
#             all_candidate_edges = [conn for conn in connections]

#             for size in range(len(all_candidate_edges), 0, -1):
#                 for edge_combination in combinations(all_candidate_edges, size):
#                     canon_form = canonical_form((candidate_table, "et"), edge_combination)
#                     all_edge_combinations.append(
#                         (canon_form, ((candidate_table, "et"), edge_combination))
#                     )
#             all_edge_combinations.sort(key=sort_edge_combinations)

#         edge_label_cache = {}
#         for i, (canon_form, edge_combination) in enumerate(all_edge_combinations):
#             temp_graph = jq_graph["jq_graph"].copy()
#             nx.set_edge_attributes(temp_graph, {e: {"color": "blue"} for e in temp_graph.edges()})
#             for src, dst in edge_combination[1]:
#                 for node in (src, dst):
#                     if node not in temp_graph:
#                         temp_graph.add_node(node, name=node[0], alias=node[1])
#                 temp_graph.add_edge(src, dst, color="red")

#                 schema_labels = edge_label_cache.setdefault(
#                     (src[0], dst[0]),
#                     [lbl.strip() for lbl in schema.get_edge_data(src[0], dst[0])["label"].split(";") if lbl.strip()]
#                 )

#                 temp_graph[src][dst]["joins"] = []

#                 for lbl in schema_labels:
#                     l_tbl, l_col, r_tbl, r_col = tuple(map(str.strip, lbl.replace("=", ".").split(".")))

#                     l_tbl_lower, r_tbl_lower = l_tbl.lower(), r_tbl.lower()
#                     src_tbl, dst_tbl = src[0].lower(), dst[0].lower()

#                     if (l_tbl_lower == src_tbl and r_tbl_lower == dst_tbl):
#                         join = f"{(src[1] or l_tbl)}.{l_col} = {(dst[1] or r_tbl)}.{r_col}"
#                     elif (r_tbl_lower == src_tbl and l_tbl_lower == dst_tbl):
#                         join = f"{(src[1] or r_tbl)}.{r_col} = {(dst[1] or l_tbl)}.{l_col}"
#                     else:
#                         continue
#                     temp_graph[src][dst]["joins"].append(join)

#             is_redundant = check_cycle_redundancy(temp_graph, edge_combination)

#             if not is_redundant:
#                 extended_graph = temp_graph
#             else:
#                 continue

#             new_query_first = {}
#             new_graph_first = {}
#             new_query_first["id"] = make_new_id(jq_graph["question_id"], i)
#             new_graph_first["id"] = make_new_id(jq_graph["question_id"], i)

#             ext_jqg = extend_old_query(
#                 jqg=jq_graph,
#                 extended_graph=extended_graph,
#             )
#             ext_jqg["ext_jq_graph"] = extended_graph

#             valid_ext_jqg, ext_is_valid = execute_extended_queries(
#                 ext_jqg, db_file_path
#             )
#             if ext_is_valid:
#                 new_query_first.update(
#                     {
#                         "ext_id": valid_ext_jqg["question_id"],
#                         "evidence": valid_ext_jqg["evidence"],
#                         "difficulty": valid_ext_jqg["difficulty"],
#                         "SQL": valid_ext_jqg["new_query"],
#                     }
#                 )

#                 if graph_first:
#                     new_queries = translate_graph_into_query(
#                         ext_jqg=extended_graph,
#                         db_id=db_id,
#                         df=df,
#                     )
#                     add_values_to_translated_queries(new_queries, db_file_path)
#                     graph_first_queries = execute_new_queries(new_queries, db_file_path)
#                     new_graph_first.update(
#                         {
#                             "SQL": graph_first_queries["main_query"],
#                             "difficulty": "challenging",
#                         }
#                     )
#             else:
#                 continue

#             if is_pattern_in_old_list(
#                 extended_graph, pre_aug_graphs
#             ) or is_pattern_in_new_list(extended_graph, filtered_ext_graphs):
#                 ext_jqg["status"] = "discarded"
#                 all_extensions.append(ext_jqg)
#                 augmented_qf_discarded_queries.append(new_query_first)
#                 augmented_gf_discarded_queries.append(new_graph_first)

#             else:
#                 augmented_qf_filtered_queries.append(new_query_first)
#                 augmented_gf_filtered_queries.append(new_graph_first)
#                 ext_jqg["status"] = "filtered"
#                 all_extensions.append(ext_jqg)
#                 filtered_ext_graphs.append(extended_graph)

#             aug_fil_queries["queries"] = augmented_qf_filtered_queries
#             aug_dis_queries["queries"] = augmented_qf_discarded_queries
#             aug_fil_queries["graph_first"] = augmented_gf_filtered_queries
#             aug_dis_queries["graph_first"] = augmented_gf_discarded_queries

#     return (aug_fil_queries, aug_dis_queries, all_extensions)


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

    aug_inputs_path = os.path.join(rule_inputs_base, "jq_augmentation", db_id)
    schema_path = os.path.join(graph_data_base, f"{db_id}_graph.pkl")
    db_file_path = os.path.join(
        data_folder, "benchmarks", "Bird", "bird_databases", db_id, f"{db_id}.sqlite"
    )
    query_stats_path = os.path.join(
        rule_inputs_base, "jq_augmentation", "query_statistics.csv"
    )

    # Check that all necessary files exist
    validate_paths(aug_inputs_path, schema_path, db_file_path, query_stats_path)

    # Load subgraphs
    pre_aug_graphs = retrieve_n_node_jqgs(aug_inputs_path, num_tables)
    all_dev_jqgs = retrieve_all_dev_jq_graphs(aug_inputs_path)
    logger.log(
        "info",
        "Pre-augmented subgraphs loaded successfully.",
        {"db_id": db_id, "number_of_subgraphs": len(pre_aug_graphs)},
    )

    # Load schema and query statistics
    schema = load_schema(schema_path)
    query_col_stats = pd.read_csv(query_stats_path)

    # Generate extended subgraphs and SQL queries
    filtered_aug, discarded_aug, all_extensions = extend_graphs_and_gen_queries(
        pre_aug_graphs,
        all_dev_jqgs,
        db_id,
        query_col_stats,
        schema,
        db_file_path,
        graph_first=graph_first,
    )

    return filtered_aug, discarded_aug, all_extensions


# def bootstrap_and_expand_query_graphs(db_id, num_tables):
#     rule_inputs_base = os.getenv("RULE_INPUTS_BASE")
#     graph_data_base = os.getenv("GRAPH_DATA_BASE")
#     data_folder = os.getenv("DATA_FOLDER")

#     aug_inputs_path = os.path.join(rule_inputs_base, "bootstrapping", db_id)
#     schema_path = os.path.join(graph_data_base, f"{db_id}_graph.pkl")
#     db_file_path = os.path.join(
#         data_folder, "benchmarks", "Bird", "bird_databases", db_id, f"{db_id}.sqlite"
#     )
#     query_stats_path = os.path.join(rule_inputs_base, "query_statistics.csv")

#     # Check that all necessary files exist
#     validate_paths(aug_inputs_path, schema_path, db_file_path, query_stats_path)

#     # Load subgraphs
#     pre_aug_graphs = retrieve_all_dev_subgraphs(aug_inputs_path, num_tables)
#     logger.log(
#         "info",
#         "Pre-augmented subgraphs loaded successfully.",
#         {"db_id": db_id, "number_of_subgraphs": len(pre_aug_graphs)},
#     )

#     # Load schema and query statistics
#     schema = load_schema(schema_path)
#     query_col_stats = pd.read_csv(query_stats_path)

#     # Generate extended subgraphs and SQL queries
#     filtered_aug, discarded_aug, all_extensions = extend_graphs_and_gen_queries(
#         pre_aug_graphs,
#         db_id,
#         query_col_stats,
#         schema,
#         db_file_path,
#     )

#     return filtered_aug, discarded_aug, all_extensions

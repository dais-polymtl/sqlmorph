import networkx as nx
import os
import pandas as pd
from itertools import combinations, product, chain
from query_generation import (
    translate_graph_into_query,
    extend_old_query,
)
from data_retrieval import (
    retrieve_n_node_jqgs,
    retrieve_all_dev_jq_graphs,
    load_schema,
)
from query_execution import (
    execute_new_queries,
    execute_extended_queries,
    add_values_to_translated_queries,
)
from src.core.logger.logger import Logger

logger = Logger(__name__)


def sort_condition_combinations(join_combinations, sort_asc=False):
    """
    Normalize edges inside each candidate_edge and sort combinations:
        1. Number of edges (descending)
        2. Lexicographical order of normalized edges
        3. Lexicographical order of all joins
    All sorted ascending/descending based on sort_asc flag.
    """

    def normalize_edge(edge):
        """
        Convert [["League","t5"], ["Match","t3"]] into 'League=Match'.
        """
        tables = [edge[0][0], edge[1][0]]  # table names only
        tables.sort()
        return "=".join(tables)

    # Preprocess: normalize edges in-place
    for combo in join_combinations:
        for edge_entry in combo["candidate_edges"]:
            if isinstance(edge_entry["edge"], list):  # not already normalized
                edge_entry["edge"] = normalize_edge(edge_entry["edge"])

    def canonical_form(combo):
        # use normalized edge strings directly
        edges_normalized = sorted(
            (e["edge"] for e in combo["candidate_edges"]), reverse=not sort_asc
        )

        # flatten joins and sort
        joins = []
        for edge_entry in combo["candidate_edges"]:
            joins.extend(edge_entry.get("joins", []))
        joins_sorted = sorted(joins, reverse=not sort_asc)

        num_edges = len(combo["candidate_edges"])
        if not sort_asc:
            num_edges = -num_edges  # for descending order

        return (num_edges, tuple(edges_normalized), tuple(joins_sorted))

    # sort the combinations using the canonical form
    sorted_combos = sorted(join_combinations, key=canonical_form)
    return sorted_combos


def create_condition_combinations(candidate_tables):
    """
    Create a flat list of combinations where each combination is an element with:
      - candidate_table: str
      - candidate_edges: list of dicts, each dict = { "edge": edge_name, "joins": [label1, label2, ...] }

    Returns: List of dicts
    Example:
    [
      {
        "candidate_table": "account",
        "candidate_edges": [
          {"edge": "account.district_id = district.district_id", "joins": ["district.district_id"]},
          {"edge": "account.customer_id = customer.customer_id", "joins": ["customer.customer_id"]}
        ]
      },
      ...
    ]
    """
    all_combinations = []

    for candidate in candidate_tables:
        table_name = candidate["node"]
        connections = candidate["connections"]

        # all non-empty combinations of edges
        edge_combos = chain.from_iterable(
            combinations(connections, r) for r in range(1, len(connections) + 1)
        )

        for edge_combo in edge_combos:
            # for each edge, generate all possible non-empty subsets of its labels
            label_subsets_per_edge = []
            for edge in edge_combo:
                subsets = [
                    list(subset)
                    for r in range(1, len(edge["labels"]) + 1)
                    for subset in combinations(edge["labels"], r)
                ]
                label_subsets_per_edge.append(subsets)

            # Cartesian product across edges → each edge gets one subset of labels
            for choice in product(*label_subsets_per_edge):
                candidate_edges_with_joins = []
                for edge, chosen_labels in zip(edge_combo, choice):
                    candidate_edges_with_joins.append(
                        {"edge": edge["edge"], "joins": chosen_labels}
                    )
                all_combinations.append(
                    {
                        "candidate_table": table_name,
                        "candidate_edges": candidate_edges_with_joins,
                    }
                )

    return all_combinations


def find_candidate_expansions(schema_graph, jq_graph, sort_asc=False):
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
                    edge_data = schema_graph.get_edge_data(node, neighbor)
                    for matched_node in table_to_nodes[neighbor_lower]:
                        alias = f"t{len(jq_graph.nodes()) + 1}"
                        edge_labels = edge_data.get("label", "").split(";")
                        new_edge_labels = []
                        for lbl in edge_labels:
                            tab1, col1, tab2, col2 = tuple(
                                map(str.strip, lbl.replace("=", ".").split("."))
                            )
                            if tab1.lower() == node_lower:
                                new_lbl = f"{alias}.{col1} = {(matched_node[1] or tab2)}.{col2}"
                            else:
                                new_lbl = f"{(matched_node[1] or tab2)}.{col2} = {alias}.{col1}"

                            new_edge_labels.append(new_lbl)

                        connections.append(
                            {
                                "edge": ((node, alias), matched_node),
                                "labels": new_edge_labels,
                            }
                        )

            if connections:
                candidate_tables.append(
                    {"node": (node, alias), "connections": connections}
                )

    if not candidate_tables:
        return []

    join_combinations = create_condition_combinations(candidate_tables)
    join_combinations_sorted = sort_condition_combinations(
        join_combinations, sort_asc=sort_asc
    )

    return join_combinations_sorted


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


def check_cycle_redundancy(temp_subgraph, candidate_table, candidate_edges):
    """
    Determines if the added edges create a redundant cycle in the subgraph.

    A cycle is considered redundant if the added edges don't introduce new
    join paths beyond what's already implied by existing join keys.

    Args:
        temp_subgraph (networkx.DiGraph or networkx.Graph): The subgraph with added edges.
        candidate_table: The canonical node (table/alias tuple or name) being extended.
        candidate_edges (List[dict]): Candidate edges, each expected to contain an "edge" key.

    Returns:
        bool: True if a redundant cycle (involving candidate_table) is found, False otherwise.
    """
    tables = list(temp_subgraph.nodes())
    canon_node = candidate_table
    added_edges = [edge["edge"] for edge in candidate_edges]

    def _resolve_node_name(tbl_name):
        """Return canonical node name (prefer node[0], else node[1].lower())."""
        node = next(
            (
                (t, a)
                for t, a in tables
                if a.lower() == tbl_name or t.lower() == tbl_name
            ),
            (tbl_name, ""),
        )
        return node[0] or node[1].lower()

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

        print("Cycle edge labels: ", cycle_edge_labels)  # --- IGNORE ---
        join_keys = set()
        for edge, labels in cycle_edge_labels.items():
            if edge in extra_edges or (edge[1], edge[0]) in extra_edges:
                continue
            for label in labels:
                try:
                    for key in map(str.strip, label.strip().split("=")):
                        table, col = key.lower().split(".", 1)
                        canon = _resolve_node_name(table)
                        join_keys.add(f"{canon}.{col}")
                except Exception as e:
                    try:
                        logger.log(
                            "error",
                            "Error parsing join label while collecting join_keys",
                            {"label": label, "error": str(e)},
                        )
                    except Exception:
                        pass
        print("Join keys so far: ", join_keys)  # --- IGNORE ---
        for extra_edge in extra_edges[:-1]:
            labels = cycle_edge_labels.get(extra_edge, []) or cycle_edge_labels.get(
                (extra_edge[1], extra_edge[0]), []
            )
            for label in labels:
                try:
                    left, right = map(str.strip, label.split("="))
                    l_tbl, l_col = left.lower().split(".", 1)
                    r_tbl, r_col = right.lower().split(".", 1)

                    l_canon = _resolve_node_name(l_tbl)
                    r_canon = _resolve_node_name(r_tbl)

                    join_keys.add(f"{l_canon}.{l_col}")
                    join_keys.add(f"{r_canon}.{r_col}")
                except Exception as e:
                    try:
                        logger.log(
                            "error",
                            "Error parsing join label from extra edge",
                            {"label": label, "error": str(e)},
                        )
                    except Exception:
                        pass
        print("Join keys after extra edges: ", join_keys)  # --- IGNORE ---

        # 3) Check redundancy for the closing edge: if any of its join labels is fully implied,
        #    treat the cycle as redundant. Break when True.
        closing_edge = extra_edges[-1]
        labels = cycle_edge_labels.get(closing_edge, []) or cycle_edge_labels.get(
            (closing_edge[1], closing_edge[0]), []
        )

        for label in labels:
            try:
                left, right = map(str.strip, label.split("="))
                l_tbl, l_col = left.lower().split(".", 1)
                r_tbl, r_col = right.lower().split(".", 1)

                l_canon = _resolve_node_name(l_tbl)
                r_canon = _resolve_node_name(r_tbl)

                l_key = f"{l_canon}.{l_col}"
                r_key = f"{r_canon}.{r_col}"
                print(
                    "Checking redundancy for closing edge label: ",
                    label,
                    " with keys: ",
                    l_key,
                    r_key,
                )  # --- IGNORE ---
                if l_key in join_keys and r_key in join_keys:
                    print(
                        "Redundant cycle detected involving candidate_table: ",
                        candidate_table,
                    )  # --- IGNORE ---
                    return True  # confirmed redundant cycle involving candidate_table
            except Exception as e:
                try:
                    logger.log(
                        "error",
                        "Error parsing join label for redundancy check",
                        {"label": label, "error": str(e)},
                    )
                except Exception:
                    pass

    return False


def make_new_id(ext_id, i):
    ext_id_str = str(ext_id).zfill(4)  # pad to 4 digits
    i_str = str(i).zfill(3)  # pad to 3 digits
    return int(f"11{ext_id_str}{i_str}")


def get_table_alias(table_name, alias_counters):
    base = table_name[0].lower()
    alias_counters[table_name] += 1
    count = alias_counters[table_name]
    return f"{base}{count if count > 1 else ''}"


def extend_graphs_and_gen_queries(
    pre_aug_graphs,
    all_dev_jqgs,
    db_id,
    df,
    schema,
    db_file_path,
    graph_first,
    max_queries=1000,
    max_isomorphic=5,
    sort_asc=False,
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

    remaining_queries = max_queries
    iso_counters = {}

    for jq_graph in pre_aug_graphs:
        if remaining_queries <= 0:
            break

        candidate_combinations = find_candidate_expansions(
            schema, jq_graph["jq_graph"], sort_asc
        )
        for i, combo in enumerate(candidate_combinations):
            if remaining_queries <= 0:
                break

            candidate_table = combo["candidate_table"]
            candidate_edges = combo["candidate_edges"]
            temp_graph = jq_graph["jq_graph"].copy()
            nx.set_edge_attributes(
                temp_graph, {e: {"color": "blue"} for e in temp_graph.edges()}
            )
            temp_graph.add_node(
                candidate_table,
                name=candidate_table[0],
                alias=candidate_table[1],
                color="red",
            )
            for candidate_edge in candidate_edges:
                temp_graph.add_edge(
                    candidate_edge["edge"][0],
                    candidate_edge["edge"][1],
                    color="red",
                    joins=candidate_edge["joins"],
                )

            is_redundant = check_cycle_redundancy(
                temp_graph, candidate_table, candidate_edges
            )

            if not is_redundant:
                extended_graph = temp_graph.copy()

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
            if not ext_is_valid:
                continue

            remaining_queries -= 1
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

            matched_graph = None
            for stored_graph in filtered_ext_graphs:
                if nx.is_isomorphic(extended_graph, stored_graph):
                    matched_graph = stored_graph
                    break
            iso_count = iso_counters.get(
                matched_graph if matched_graph else extended_graph, 0
            )

            if (
                is_pattern_in_old_list(extended_graph, all_dev_jqgs)
                or is_pattern_in_new_list(extended_graph, filtered_ext_graphs)
            ) and iso_count >= max_isomorphic:

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

                iso_counters[matched_graph if matched_graph else extended_graph] = (
                    iso_count + 1
                )

            aug_fil_queries["queries"] = augmented_qf_filtered_queries
            aug_dis_queries["queries"] = augmented_qf_discarded_queries
            aug_fil_queries["graph_first"] = augmented_gf_filtered_queries
            aug_dis_queries["graph_first"] = augmented_gf_discarded_queries

    return (aug_fil_queries, aug_dis_queries, all_extensions)


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


def aug_n_table_sql_queries(
    db_id,
    num_tables,
    graph_first=False,
    max_queries=1000,
    max_isomorphic=5,
    sort_asc=True,
):
    rule_inputs_base = os.getenv("RULE_INPUTS_BASE") + "/jq_augmentation"
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
        max_queries=max_queries,
        max_isomorphic=max_isomorphic,
        sort_asc=sort_asc,
    )

    return filtered_aug, discarded_aug, all_extensions

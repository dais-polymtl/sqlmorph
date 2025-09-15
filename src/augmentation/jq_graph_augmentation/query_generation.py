import sqlglot
from sqlglot import expressions as exp
from functools import reduce
import operator
import re
from typing import Dict, Any
import networkx as nx


def replace_order_identifier(query):
    # Pattern to match 'order' except when it's the keyword ORDER in 'ORDER BY'
    pattern = r"""(?<!["\w])         # not preceded by quote or word char
                  (?:\.|\b)          # either a dot or word boundary before
                  (order)            # the word 'order'
                  (?!["\w])          # not followed by quote or word char
               """

    def replacement(m):
        # Check if this 'order' is followed by 'by' (with optional whitespace)
        start = m.start()
        after = query[start:].lower()
        if (
            after.startswith("order by")
            or after.startswith("order\tby")
            or after.startswith("order\nby")
        ):
            # It's the ORDER BY keyword, do NOT replace
            return m.group(0)
        # Otherwise, replace with quoted "order", preserving leading dot if any
        return (m.group(0)[0] if m.group(0).startswith(".") else "") + '"order"'

    return re.sub(pattern, replacement, query, flags=re.IGNORECASE | re.VERBOSE)


def retrieve_projection_filter_columns(db_id, df, tables):
    """
    Retrieve the projection and filtering columns along with their corresponding tables based on occurrences for a given db_id.

    Args:
        db_id: The database ID to filter (e.g., 'card_games').
        df: The pandas DataFrame containing the data.
        tables: List of table names and aliases involved in the current query model.

    Returns:
        projection_columns_with_tables: List of tuples (table_name, projection_column, table_alias) for the top p projection columns.
        filtering_columns_with_tables: List of tuples (table_name, filtering_column, table_alias) for the top w filtering columns.
    """
    # Step 1: Retrieve avg_projections_per_query (p) and avg_conditions_per_query (w)
    avg_projection = df.loc[df["db_id"] == db_id, "avg_projections_per_query"].values[0]
    avg_conditions = df.loc[df["db_id"] == db_id, "avg_conditions_per_query"].values[0]

    p = int(avg_projection)  # Number of projection columns to include
    w = int(avg_conditions)  # Number of filtering columns to include

    # Step 2: Filter rows where table_name is in tables_involved
    tables_involved = list({table[0].lower() for table in tables})

    involved_tables_df = df[
        (df["db_id"] == db_id) & (df["table_name"].isin(tables_involved))
    ].copy()

    # Step 3: Handle NaN values and split occurrence strings
    involved_tables_df["projection_occurrences"] = involved_tables_df[
        "projection_occurrences"
    ].fillna("0/1")
    involved_tables_df["projection_occurrences"] = (
        involved_tables_df["projection_occurrences"].str.split("/").str[0].astype(int)
    )

    # Step 4: Select the top p projection columns based on projection_occurrences
    top_projection_columns_df = involved_tables_df.sort_values(
        by="projection_occurrences", ascending=False
    ).head(p)
    # Get the projection columns with their corresponding tables
    projection_columns_with_tables = list(
        top_projection_columns_df[["table_name", "projection_column"]].itertuples(
            index=False, name=None
        )
    )

    # Step 5: Handle filtering columns while excluding those in the projection columns
    if "filtering_column" in involved_tables_df.columns:
        involved_tables_df["filtering_occurrences"] = involved_tables_df[
            "filtering_occurrences"
        ].fillna("0/1")
        involved_tables_df["filtering_occurrences"] = (
            involved_tables_df["filtering_occurrences"]
            .str.split("/")
            .str[0]
            .astype(int)
        )

        # Exclude columns that are in the projection from filtering columns
        projection_columns = set(
            col[1] for col in projection_columns_with_tables
        )  # Get the names of projection columns
        filtering_candidates_df = involved_tables_df[
            ~involved_tables_df["filtering_column"].isin(projection_columns)
        ]

        # Select the top w filtering columns based on filtering_occurrences
        top_filtering_columns_df = filtering_candidates_df.sort_values(
            by="filtering_occurrences", ascending=False
        ).head(w)
        filtering_columns_with_tables = list(
            top_filtering_columns_df[["table_name", "filtering_column"]].itertuples(
                index=False, name=None
            )
        )
    else:
        filtering_columns_with_tables = []

    filtering_columns_with_tables = [
        (
            table,
            f'"{col}"' if " " in col else col,
            next((alias for t, alias in tables if t.lower() == table.lower()), None),
        )
        for table, col in filtering_columns_with_tables
    ]

    projection_columns_with_tables = [
        (
            table,
            f'"{col}"' if " " in col else col,
            next((alias for t, alias in tables if t.lower() == table.lower()), None),
        )
        for table, col in projection_columns_with_tables
    ]

    return projection_columns_with_tables, filtering_columns_with_tables


def remove_simple_condition_parentheses(sql):
    pattern = r"\(([\w\.]+)\s*=\s*([\w\.?]+)\)"

    def replacer(match):
        left = match.group(1)
        right = match.group(2)
        # Return without parentheses
        return f"{left} = {right}"

    prev_sql = None
    new_sql = sql
    while new_sql != prev_sql:
        prev_sql = new_sql
        new_sql = re.sub(pattern, replacer, new_sql)
    return new_sql


def translate_graph_into_query(ext_jqg, db_id, df):
    """
    Generate a SQL query and a test query from a join query graph (JQG).

    Parameters:
        ext_jqg (nx.Graph): The extended join query graph with table and join information.
        db_id (str): The database ID used to extract relevant columns from the DataFrame.
        df (pd.DataFrame): The metadata containing projection and filtering columns.

    Returns:
        dict: {
            "main_query": str,
            "test_query": str,
            "projection_columns": list,
            "filtering_columns": list
        }
    """
    tables = list(ext_jqg.nodes())

    query_joins = [
        label
        for _, _, edge_data in ext_jqg.edges(data=True)
        for label in edge_data.get("joins", [])
    ]

    projection_columns_with_tables, filtering_columns_with_tables = (
        retrieve_projection_filter_columns(db_id, df, tables)
    )

    select_columns = [
        exp.Column(this=col, table=alias or table)
        for table, col, alias in projection_columns_with_tables
    ]

    base_query = sqlglot.select(*select_columns).distinct()

    from_clause = ", ".join(
        f"{table} AS {alias}" if alias else table for table, alias in tables
    )

    base_query = base_query.from_(from_clause)

    where_conditions = []

    for join in query_joins:
        left, right = map(str.strip, join.split("="))
        left_table, left_col = left.split(".", 1)
        right_table, right_col = right.split(".", 1)
        where_conditions.append(
            exp.EQ(
                this=exp.Column(this=left_col, table=left_table),
                expression=exp.Column(this=right_col, table=right_table),
            )
        )

    where_conditions.extend(
        exp.EQ(
            this=exp.Column(this=col, table=alias or table),
            expression=exp.Placeholder(),
        )
        for table, col, alias in filtering_columns_with_tables
    )

    if where_conditions:
        base_query = base_query.where(reduce(operator.and_, where_conditions))

    main_query = remove_simple_condition_parentheses(base_query.sql())
    if "order" in main_query:
        main_query = replace_order_identifier(main_query)

    test_select_columns = [
        f"{alias}.{col}" if alias else f"{table}.{col}"
        for table, col, alias in projection_columns_with_tables
        + filtering_columns_with_tables
    ]
    test_query = f"""
    SELECT {", ".join(test_select_columns)}, COUNT(*) AS combination_count
    FROM {from_clause}
    WHERE {" AND ".join(query_joins)}
    GROUP BY {", ".join(test_select_columns)};
    """
    if "order" in test_query:
        test_query = replace_order_identifier(test_query)

    return {
        "main_query": main_query,
        "test_query": test_query,
        "projection_columns": projection_columns_with_tables,
        "filtering_columns": filtering_columns_with_tables,
    }


def extend_old_query(jqg: Dict[str, Any], extended_graph: nx.Graph) -> Dict[str, Any]:
    """
    Extend an existing SQL query by adding a join to an additional table.

    This function takes a join query graph (`jqg`) and a graph (`extended_graph`) that includes
    a new candidate table (with alias 'et'). It parses the original SQL query from the jqg, adds
    the appropriate JOIN conditions based on the labeled edges in the extended graph, and
    reconstructs the extended SQL query.

    Args:
        jqg (Dict[str, Any]): A join query graph containing the original SQL under "flattened_query" or "SQL".
        extended_graph (nx.Graph): A graph containing the candidate table and its join edges, with labels.

    Returns:
        Dict[str, Any]: A modified copy of the original `jqg` including the new extended SQL query under "new_query".
    """

    jqg_copy = jqg.copy()
    old_query = jqg_copy.get("flattened_query") or jqg_copy.get("SQL")
    # print("extended graph nodes: ", extended_graph.nodes(data=True))
    # added_table = next((n for n in extended_graph.nodes if n["color"] == "red"), None)
    added_table = next(
        (
            n
            for n, data in extended_graph.nodes(data=True)
            if data.get("color") == "red"
        ),
        None,
    )
    print("added_table: ", added_table)

    added_joins = sum(
        (
            edge_data.get("joins", [])
            for _, _, edge_data in extended_graph.edges(data=True)
            if edge_data.get("color") == "red"
        ),
        [],
    )

    parsed = sqlglot.parse_one(old_query, dialect="mysql")
    if not isinstance(parsed, exp.Select):
        return jqg_copy

    if len(jqg_copy["jq_graph"].nodes) == 1:
        table_name = list(jqg_copy["jq_graph"].nodes)[0][0]

        def qualify_column(expr):
            if isinstance(expr, exp.Column) and not expr.table:
                expr.set("table", exp.to_identifier(table_name))
            elif hasattr(expr, "args"):
                for v in expr.args.values():
                    if isinstance(v, list):
                        for item in v:
                            qualify_column(item)
                    elif isinstance(v, exp.Expression):
                        qualify_column(v)

        for node in parsed.walk():
            qualify_column(node)

    parsed.args.setdefault("joins", [])
    join_conditions = [
        exp.column(left_col, table=left_table).eq(
            exp.column(right_col, table=right_table)
        )
        for join in added_joins
        for left, right in [map(str.strip, join.split("="))]
        for left_table, left_col in [left.split(".")]
        for right_table, right_col in [right.split(".")]
    ]

    if join_conditions and added_table:
        parsed.args["joins"].append(
            exp.Join(
                this=exp.Table(
                    this=exp.Identifier(this=added_table[0], quoted=False),
                    alias=exp.TableAlias(
                        this=exp.Identifier(this=added_table[1], quoted=False)
                    ),
                ),
                on=exp.and_(*join_conditions),
                kind="INNER",
            )
        )

    sql_result = parsed.sql()
    if "order" in sql_result:
        sql_result = replace_order_identifier(sql_result)
    jqg_copy["new_query"] = sql_result

    return jqg_copy

import sqlglot
from sqlglot import expressions as exp
from sqlglot.expressions import Subquery
import networkx as nx


def get_table_data(schema, node_name):
    """
    Retrieve table data from the schema given a node name.

    Parameters:
        schema: The graph schema containing table information.
        node_name: The name of the node (table) to retrieve data for.

    Returns:
        Tuple containing table name, list of column names, and list of column types.
    """
    if node_name in schema.nodes:
        data = schema.nodes[node_name]
        return data["table_name"], data["column_names"], data["column_types"]
    return None, [], []


def retrieve_projection_filter_columns(db_id, df, tables_involved):
    """
    Retrieve the projection and filtering columns along with their corresponding tables based on occurrences for a given db_id.

    Parameters:
        db_id: The database ID to filter (e.g., 'card_games').
        df: The pandas DataFrame containing the data.
        tables_involved: List of table names involved in the current query model.

    Returns:
        projection_columns_with_tables: List of tuples (table_name, projection_column) for the top p projection columns.
        filtering_columns_with_tables: List of tuples (table_name, filtering_column) for the top w filtering columns.
    """
    # Step 1: Retrieve avg_projections_per_query (p) and avg_conditions_per_query (w)
    avg_projection = df.loc[df["db_id"] == db_id, "avg_projections_per_query"].values[0]
    avg_conditions = df.loc[df["db_id"] == db_id, "avg_conditions_per_query"].values[0]

    p = int(avg_projection)  # Number of projection columns to include
    w = int(avg_conditions)  # Number of filtering columns to include

    # Step 2: Filter rows where table_name is in tables_involved
    tables_involved = [table.lower() for table in tables_involved]
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

    return projection_columns_with_tables, filtering_columns_with_tables


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


def translate_graph_into_query(schema, pattern, db_id, df):
    """
    Generate an INNER JOIN SQL query and a test query for validation.

    Parameters:
        schema: The graph schema containing table information.
        pattern: The subgraph pattern to generate the query for.
        db_id: The database ID to filter on for projection/filtering columns.
        df: The DataFrame containing information about projection and filtering columns.

    Returns:
        A dictionary with the original query, the test query, projection columns, filtering columns.
    """
    tables = set()
    join_conditions = []

    # Collect table data for each node in the subgraph using the schema
    for node in pattern.nodes():
        table_name, columns, _ = get_table_data(schema, node)
        if table_name:
            tables.add(table_name)

    # Collect join conditions from the edges of the subgraph
    for edge in pattern.edges(data=True):
        left_table, right_table, edge_data = edge
        if edge_data.get("color") == "red":
            join_condition = edge_data.get("label").split(";")[0]
        elif edge_data.get("color") == "blue":
            join_condition = edge_data.get("label")
        join_conditions.append((left_table, right_table, join_condition))

    # Get the tables involved in the current query
    tables_involved = list(tables)

    # Retrieve projection and filtering columns based on the db_id and involved tables
    projection_columns_with_tables, filtering_columns_with_tables = (
        retrieve_projection_filter_columns(db_id, df, tables_involved)
    )
    # Construct the SELECT columns for the main query
    select_columns = [
        exp.Column(this=col, table=table)
        for table, col in projection_columns_with_tables
    ]

    # Initialize the base query
    base_query = sqlglot.select(
        *select_columns
    ).distinct()  # Add DISTINCT to the main query

    # Construct the FROM clause with all involved tables
    from_clause = ", ".join(tables_involved)
    base_query = base_query.from_(from_clause)

    # Prepare WHERE conditions, starting with join conditions
    where_conditions = []

    # Add join conditions to WHERE
    for _, _, condition in join_conditions:
        separated_conditions = condition.split(";")
        for separated_condition in separated_conditions:
            left_col = separated_condition.split("=")[0].strip().split(".")[1]
            left_table = separated_condition.split("=")[0].strip().split(".")[0]
            right_col = separated_condition.split("=")[1].strip().split(".")[1]
            right_table = separated_condition.split("=")[1].strip().split(".")[0]
            where_conditions.append(
                exp.EQ(
                    this=exp.Column(this=left_col, table=left_table),
                    expression=exp.Column(this=right_col, table=right_table),
                )
            )

    # Add filtering conditions to WHERE (using placeholders)
    for table, col in filtering_columns_with_tables:
        where_conditions.append(
            exp.EQ(
                this=exp.Column(this=col, table=table),
                expression=exp.Placeholder(),  # Placeholder for filtering
            )
        )

    # Apply WHERE conditions without unnecessary parentheses
    if where_conditions:
        base_query = base_query.where(
            where_conditions[0]
        )  # Start with the first condition
        for condition in where_conditions[1:]:
            base_query = base_query.where(condition)  # Add remaining conditions

    # Replace parentheses with an empty character in the SQL query string
    main_query = base_query.sql().replace("(", "").replace(")", "")

    # Now, generate the test query
    test_select_columns = [
        f"{table}.{col}"
        for table, col in projection_columns_with_tables + filtering_columns_with_tables
    ]

    # Create the WHERE clause for the test query
    test_where_conditions = []
    for _, _, condition in join_conditions:
        # if len(condition.split(";")) > 1:
        separated_conditions = condition.split(";")
        for separated_condition in separated_conditions:
            left_col = separated_condition.split("=")[0].strip().split(".")[1]
            left_table = separated_condition.split("=")[0].strip().split(".")[0]
            right_col = separated_condition.split("=")[1].strip().split(".")[1]
            right_table = separated_condition.split("=")[1].strip().split(".")[0]
            test_where_conditions.append(
                f"{left_table}.{left_col} = {right_table}.{right_col}"
            )

      
    # Construct the test query
    test_query = f"""
    SELECT {", ".join(test_select_columns)}, COUNT(*) AS combination_count
    FROM {from_clause}
    WHERE {" AND ".join(test_where_conditions)}
    GROUP BY {", ".join(test_select_columns)};
    """

    # Return both queries and relevant columns
    return {
        "main_query": main_query,
        "test_query": test_query,
        "projection_columns": projection_columns_with_tables,
        "filtering_columns": filtering_columns_with_tables,
    }

def extend_old_query(pattern):
    extended_subgraph = pattern['extended_subgraph']
    equivalent_queries = pattern['equivalent_queries']
    old_subgraph = pattern['subgraph']
    added_table = list(set(extended_subgraph.nodes) - set(old_subgraph.nodes))[0]
    added_alias = "extra_table"
    
    added_joins = []
    for edge in extended_subgraph.edges(data=True):
        if edge[2]['color'] == 'red':
            edge_label = edge[2]['label'].split(";")[0]
            added_joins.append(edge_label)


    extended_old_queries = []
    for query in equivalent_queries:
        old_flattened_query = query['flattened_query']
        old_sql = query['SQL']
        old_flattened_query = old_flattened_query.replace("DATETIME()", '"DATETIME()"')
        old_sql = old_sql.replace("DATETIME()", '"DATETIME()"')
        old_query = old_flattened_query if sqlglot.parse_one(old_sql, dialect="mysql").find(Subquery) is not None and old_flattened_query != "" else old_sql
        
        # Step 1: Track all tables and their aliases
        old_query_tables = {}
        for table in sqlglot.parse_one(old_query, dialect="mysql").find_all(exp.Table):
            table_name = table.name.lower()
            table_alias = table.alias if table.alias else table_name
            if table_name not in old_query_tables:
                old_query_tables[table_name] = []
            old_query_tables[table_name].append(table_alias)

        
        # Step 3: Parse and modify the old query
        parsed_old_query = sqlglot.parse_one(old_query, dialect="mysql")
        if isinstance(parsed_old_query, exp.Select):
            if "joins" not in parsed_old_query.args:
                parsed_old_query.args["joins"] = []
            
            
            # alias_counter = 1  # Counter for dynamically generating aliases for the new table
            join_conditions = []
            for join in added_joins:
                left, right = join.split("=")
                left_table, left_col = left.strip().split(".")
                right_table, right_col = right.strip().split(".")

                if right_table == added_table:
                    old_table, old_column, added_col = left_table, left_col, right_col
                else:
                    old_table, old_column, added_col = right_table, right_col, left_col
                
                old_table_alias = old_query_tables.get(old_table.lower(), [old_table])[0]
                join_conditions.append(exp.column(added_col, table=added_alias).eq(exp.column(old_column, table=old_table_alias)))                

                
            if join_conditions:
                join_expr = exp.Join(
                    this=exp.Table(
                        this=exp.Identifier(this=added_table, quoted=False),
                        alias=exp.TableAlias(this=exp.Identifier(this=added_alias, quoted=False))
                    ),
                    on=exp.and_(*join_conditions),
                    kind="INNER"
                )
                parsed_old_query.args["joins"].append(join_expr)

        # Step 4: Convert back to SQL
        new_query = parsed_old_query.sql()
        if 'order' in new_query:
            new_query = new_query.replace('order', "'order'")
        query['new_query'] = new_query
        extended_old_queries.append(query)

    return extended_old_queries
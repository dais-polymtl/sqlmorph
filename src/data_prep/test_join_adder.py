import sqlglot
from sqlglot import expressions as exp

def add_join(query, join_table, join_alias, join_condition, join_type="LEFT"):
    parsed = sqlglot.parse_one(query)
    if not isinstance(parsed, exp.Select):
        return query
    
    # Create table with alias
    join_table_expr = exp.Table(
        this=exp.Identifier(this=join_table, quoted=False),
        alias=exp.TableAlias(this=exp.Identifier(this=join_alias, quoted=False))
    )
    
    # Create join expression
    join_expr = exp.Join(
        this=join_table_expr,
        on=sqlglot.parse_one(join_condition),
        kind=join_type
    )
    
    # Initialize joins if needed
    if "joins" not in parsed.args:
        parsed.set("joins", [])
    
    parsed.args["joins"].append(join_expr)
    return parsed.sql()

# Usage
original_query = "SELECT a.id, a.name FROM table_a a"
modified_query = add_join(
    original_query,
    join_table="table_b",
    join_alias="b",
    join_condition="a.id = b.a_id",
    join_type="INNER"
)
print(modified_query)
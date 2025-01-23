import json
import re
import os
import sqlparse
from sqlparse.sql import Identifier, Comparison
from sqlglot import parse_one, exp
import logging
from typing import List, Tuple, Dict

# Setting up basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to replace backticks with single quotes and handle DATETIME()
def sanitize_sql(sql: str) -> str:
    """
    Sanitize SQL query by replacing backticks with single quotes and handling DATETIME().

    Args:
        sql (str): The SQL query to sanitize.

    Returns:
        str: The sanitized SQL query.
    """
    sql = sql.replace("`", "'")  # Replace backticks with single quotes
    sql = re.sub(
        r"\bDATETIME\(\)", "'DATETIME()'", sql, flags=re.IGNORECASE
    )  # Replace DATETIME() with 'DATETIME()'
    return sql


# Function to check if a query is nested using regex
def is_nested_query(sql: str) -> bool:
    """
    Check if the SQL query contains a nested query.

    Args:
        sql (str): The SQL query to check.

    Returns:
        bool: True if the query is nested, otherwise False.
    """
    # Use regex to detect a subquery (SELECT inside another SELECT)
    pattern = re.compile(
        r"\bSELECT\b.*?\bFROM\b.*?\bSELECT\b", re.IGNORECASE | re.DOTALL
    )
    return bool(pattern.search(sql))


# Function to extract tables and aliases using sqlglot
def extract_tables(sql: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Extract tables and their aliases from the SQL query.

    Args:
        sql (str): The SQL query to extract tables from.

    Returns:
        Tuple: A tuple containing a list of table information and a dictionary of aliases to table names.
    """
    sanitized_sql = sanitize_sql(sql)  # Sanitize the SQL before parsing
    tables_info = []
    aliases_to_table = {}
    try:
        parsed_query = parse_one(sanitized_sql)  # Parse the query
        if parsed_query:  # Check if parsing was successful
            for table in parsed_query.find_all(exp.Table):
                table_name = table.name
                table_alias = table.alias
                tables_info.append(
                    {"table_name": table_name, "table_alias": table_alias}
                )
                if table_alias:
                    aliases_to_table[table_alias] = table_name
    except Exception as e:
        logging.error(f"Error parsing tables in SQL: {sql}")
        logging.error(f"Exception: {e}")
    return tables_info, aliases_to_table


# Function to replace aliases with table names in join conditions
def replace_aliases_in_join_conditions(
    join_conditions: List[str], aliases_to_table: Dict[str, str]
) -> List[str]:
    """
    Replace aliases in join conditions with actual table names.

    Args:
        join_conditions (List[str]): The list of join conditions.
        aliases_to_table (Dict[str, str]): A dictionary of aliases to table names.

    Returns:
        List[str]: The updated join conditions with aliases replaced by table names.
    """
    updated_conditions = []
    for condition in join_conditions:
        for alias, table in aliases_to_table.items():
            condition = condition.replace(alias, table)
        updated_conditions.append(condition)
    return updated_conditions


# Function to extract join conditions using sqlparse
def extract_joins(token_list) -> List[Comparison]:
    """
    Extract join conditions from the SQL query tokens.

    Args:
        token_list: The list of SQL query tokens.

    Returns:
        List[Comparison]: A list of join conditions.
    """
    join_pattern = re.compile(r"\b(\w+\.\w+)\s*=\s*(\w+\.\w+)\b", re.IGNORECASE)
    join_conditions = []
    for token in token_list:
        if isinstance(token, Comparison):
            parent_token = token.parent
            if any(str(t).upper() == "ON" for t in parent_token.tokens):
                # Check if the token matches the join pattern
                if join_pattern.match(str(token)):
                    join_conditions.append(token)
        elif token.is_group:
            join_conditions.extend(extract_joins(token.tokens))
    return join_conditions


# Function to extract join conditions from SQL query using sqlparse
def get_joins_and_tables(sql: str) -> List[str]:
    """
    Extract join conditions from the SQL query.

    Args:
        sql (str): The SQL query to extract join conditions from.

    Returns:
        List[str]: A list of join conditions extracted from the SQL query.
    """
    sanitized_sql = sanitize_sql(sql)  # Sanitize the SQL before parsing
    parsed = sqlparse.parse(sanitized_sql)
    join_conditions = []
    for statement in parsed:
        join_conditions.extend([str(t) for t in extract_joins(statement.tokens)])
    return join_conditions


# Main function to process the JSON file and save the result
def process_queries(input_file: str, output_folder: str):
    """
    Process the input queries and save the results divided by db_id, and save nested queries separately.

    Args:
        input_file (str): The input JSON file containing the SQL queries.
        output_folder (str): The folder where the output files should be saved.
    """
    with open(input_file, "r") as file:
        data = json.load(file)

    # Dictionaries to hold data for each db_id
    db_data = {}
    nested_queries = []

    for entry in data:
        sql_query = entry.get("SQL", "")
        db_id = entry.get("db_id", "")

        # Skip nested queries and store them in a separate list
        if is_nested_query(sql_query):
            nested_queries.append(entry)
            continue

        # Extract tables and their aliases
        tables, aliases_to_table = extract_tables(sql_query)

        # Extract join conditions
        join_conditions = get_joins_and_tables(sql_query)

        # Replace aliases in join conditions
        join_conditions_without_aliases = replace_aliases_in_join_conditions(
            join_conditions, aliases_to_table
        )

        # Prepare the processed data
        processed_entry = {
            "Tables": tables,
            "Join_relations_with_aliases": join_conditions,
            "Join_relations_without_aliases": join_conditions_without_aliases,
            "db_id": db_id,
            "SQL": sql_query,
            "question": entry["question"],
            "evidence": entry["evidence"],
            "difficulty": entry["difficulty"],
        }

        # Add to the corresponding db_id
        if db_id not in db_data:
            db_data[db_id] = []
        db_data[db_id].append(processed_entry)

    # Save queries grouped by db_id
    os.makedirs(output_folder, exist_ok=True)
    for db_id, queries in db_data.items():
        output_file = os.path.join(output_folder, f"{db_id}_unnested_queries.json")
        with open(output_file, "w") as outfile:
            json.dump(queries, outfile, indent=4)
        logging.info(f"Saved {len(queries)} queries to {output_file}")

    # Save nested queries to a separate file
    nested_output_file = os.path.join(output_folder, "nested_queries.json")
    with open(nested_output_file, "w") as outfile:
        json.dump(nested_queries, outfile, indent=4)
    logging.info(f"Saved {len(nested_queries)} nested queries to {nested_output_file}")

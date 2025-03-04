import json
import re
import os
import sqlparse
from sqlparse.sql import Identifier, Comparison
from sqlglot import parse_one, exp
import logging
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import openai
import subprocess


# Setting up basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def run_evaluation(
    predicted_sql_path,
    ground_truth_path,
    db_root_path,
    diff_json_path,
    num_cpus=1,
    meta_time_out=30.0,
):
    """
    Run the evaluation script and return execution accuracy (EX score).
    """
    try:
        result = subprocess.run(
            [
                "python",
                "evaluation.py",
                "--predicted_sql_path",
                predicted_sql_path,
                "--ground_truth_path",
                ground_truth_path,
                "--data_mode",
                "dev",
                "--db_root_path",
                db_root_path,
                "--num_cpus",
                str(num_cpus),
                "--meta_time_out",
                str(meta_time_out),
                "--mode_gt",
                "gt",
                "--mode_predict",
                "gpt",
                "--diff_json_path",
                diff_json_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
        ex_score = parse_ex_score(output)
        return ex_score
    except subprocess.CalledProcessError as e:
        logging.error(f"Evaluation failed: {e}")
        return 0


def parse_ex_score(output):
    """
    Parse the EX score from the evaluation script output.
    """
    try:
        # Split the output into lines
        lines = output.splitlines()

        # Find the line with the accuracy values
        for i, line in enumerate(lines):
            if "accuracy" in line:
                # Extract the accuracy values
                accuracy_values = line.split()
                # The last value in the line corresponds to the total accuracy
                total_accuracy = float(accuracy_values[-1])
                return total_accuracy
    except Exception as e:
        logging.error(f"Failed to parse EX score: {e}")
    return 0


def extract_flattened_query(sql_query):
    """
    Transform a nested SQL query into a single flat SQL query with only one SELECT statement.

    Args:
        sql_query (str): The SQL query to process.

    Returns:
        str: The flattened SQL query as a continuous string without newlines.
    """
    prompt = f"""
    Flatten the following nested SQL query into a single SQL query with only one SELECT statement:
    - Avoid using subqueries.
    - If the query involves finding the maximum or minimum value (using MAX(), MIN(), etc.), replace the subquery with ORDER BY and LIMIT to retrieve the top result.
    - Use JOIN only when necessary to combine tables. Avoid using subqueries to aggregate data that can be done with window functions or joins.
    - Use CASE statements for conditional logic when needed, especially for conditional aggregation.
    - If the query requires fetching the maximum, minimum, or other aggregate values that should affect the entire result set, consider using window functions (e.g., OVER()), if applicable.
    - Ensure the flattened query returns the same result as the original query.
    Return only the SQL query with one SELECT statement, without subqueries, no additional explanations or text, and no newlines. The query should be a single line.

    Example 1:
    Original query:
    SELECT budget.budget_id FROM budget JOIN (SELECT MAX(amount) AS max_amount FROM budget) AS max_budget ON budget.amount = max_budget.max_amount WHERE budget.category = 'Food';
    Flattened query:
    SELECT budget.budget_id FROM budget WHERE budget.category = 'Food' ORDER BY budget.amount DESC LIMIT 1;

    Example 2:
    Original query:
    SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.overall_rating = (SELECT MAX(overall_rating) FROM Player_Attributes) ORDER BY t2.overall_rating DESC LIMIT 1;
    Flattened query:
    SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id ORDER BY t2.overall_rating DESC LIMIT 1;

    Example 3:
    Original query:
    SELECT COUNT(DISTINCT T.element) FROM ( SELECT DISTINCT T2.molecule_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = '-' ) AS T
    Flattened query:
    SELECT COUNT(DISTINCT T1.element) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = '-'

    Example 4: 
    Original query: 
    WITH SubQuery AS (SELECT DISTINCT T1.atom_id, T1.element, T1.molecule_id, T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = 'TR006') SELECT CAST(COUNT(CASE WHEN element = 'h' THEN atom_id ELSE NULL END) AS REAL) / (CASE WHEN COUNT(atom_id) = 0 THEN NULL ELSE COUNT(atom_id) END) AS ratio, label FROM SubQuery GROUP BY label
    Flattened query:
    SELECT CAST(COUNT(CASE WHEN T1.element = 'h' THEN T1.atom_id ELSE NULL END) AS REAL) / NULLIF(COUNT(T1.atom_id), 0) AS ratio, T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = 'TR006' GROUP BY T2.label

    Example 5: 
    Original query:
    SELECT AVG(single_bond_count) FROM (SELECT T3.molecule_id, COUNT(T1.bond_type) AS single_bond_count FROM bond AS T1  INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN molecule AS T3 ON T3.molecule_id = T2.molecule_id WHERE T1.bond_type = '-' AND T3.label = '+' GROUP BY T3.molecule_id) AS subquery
    flattened_query: 
    SELECT AVG(COUNT(T1.bond_type)) OVER() FROM bond AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN molecule AS T3 ON T3.molecule_id = T2.molecule_id WHERE T1.bond_type = '-' AND T3.label = '+' GROUP BY T3.molecule_id;

    Example 6:
    Original query:
    WITH MaxBanned AS (SELECT format, COUNT(*) AS count_banned FROM legalities WHERE status = 'Banned' GROUP BY format ORDER BY COUNT(*) DESC LIMIT 1) SELECT T2.format, T1.name FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid INNER JOIN MaxBanned MB ON MB.format = T2.format WHERE T2.status = 'Banned'
    Flattened query:
    SELECT T2.format, T1.name FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid WHERE T2.status = 'Banned' AND T2.format = (SELECT format FROM legalities WHERE status = 'Banned' GROUP BY format ORDER BY COUNT(*) DESC LIMIT 1);

    Input SQL query: {sql_query}
    Output: 
    Just give the SQL query with one SELECT, no subqueries, no additional explanation or text, just the query in a single string, no newlines. THE FLATTENED QUERY SHOULD RETURN THE SAME VALUES AS THE ORIGINAL QUERY.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Ensure you're using the appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # Remove any newlines and extra spaces from the response
        return response.choices[0].message.content.strip().replace("\n", " ")

    except Exception as e:
        print(f"Error occurred while generating query: {e}")
        return sql_query  # Return the original query in case of error


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
        # Split the condition into left and right sides of the join
        left, right = condition.split("=")
        left, right = left.strip(), right.strip()

        # Extract table or alias and column names
        table_or_alias1, col1 = left.split(".")
        table_or_alias2, col2 = right.split(".")

        # Use aliases if they exist, otherwise keep the original names
        table_1 = aliases_to_table.get(table_or_alias1, table_or_alias1)
        table_2 = aliases_to_table.get(table_or_alias2, table_or_alias2)

        # Format the condition with the resolved table names
        updated_condition = f"{table_1}.{col1} = {table_2}.{col2}"
        updated_conditions.append(updated_condition)

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


def evaluate_query(query: Dict[str, Any], new_query: str) -> float:
    """
    Evaluates a query by running a custom evaluation and returns the score.
    """
    pred_file = "./predict_dev.json"
    gt_file = "./dev_gold.sql"
    db_root_path = "./dev_databases/"
    diff_json_path = "./diff.json"

    pred_queries = {
        query["question_id"]: f"{new_query}\t----- bird -----\t{query['db_id']}"
    }
    difficulties = [
        {"sql_idx": query["question_id"], "difficulty": query["difficulty"]}
    ]

    with open(pred_file, "w") as f:
        json.dump(pred_queries, f, indent=4)

    with open(gt_file, "w") as f:
        f.write(f"{query['SQL']}\t{query['db_id']}")

    with open(diff_json_path, "w") as f:
        json.dump(difficulties, f, indent=4)

    return run_evaluation("./", "./", db_root_path, diff_json_path)


def process_queries(input_file: str, output_folder: str):
    """
    Processes SQL queries from the input JSON file, categorizes them by db_id,
    and iteratively flattens and processes nested queries. The results are saved
    in the specified output folder.

    Args:
        input_file (str): Path to the input JSON file containing SQL queries.
        output_folder (str): Directory where processed query results will be saved.
    """
    with open(input_file, "r") as file:
        queries = json.load(file)

    db_queries = {}
    nested_queries = [
        entry for entry in queries if is_nested_query(entry.get("SQL", ""))
    ]
    non_nested_queries = [entry for entry in queries if entry not in nested_queries]

    # Process non-nested queries
    for query in non_nested_queries:
        process_and_store_query(query, db_queries)

    # Iteratively flatten and process nested queries
    pending_nested_queries = nested_queries
    max_iterations = 8

    for iteration in range(1, max_iterations + 1):
        print(
            f"Iteration {iteration}, pending nested queries: {len(pending_nested_queries)}"
        )
        if not pending_nested_queries or len(pending_nested_queries) <= 3:
            break

        next_pending_nested_queries = []

        for query in pending_nested_queries:
            sql_query = query.get("SQL", "")
            flattened_query = extract_flattened_query(sql_query)

            try:
                if is_nested_query(flattened_query):
                    next_pending_nested_queries.append(query)

                else:
                    evaluation_score = evaluate_query(query, flattened_query)
                    if evaluation_score < 100.00:
                        next_pending_nested_queries.append(query)
                    else:
                        process_and_store_query(
                            query, db_queries, is_nested=True, new_query=flattened_query
                        )
            except Exception as error:
                logging.error(f"Error processing nested query: {error}")
                next_pending_nested_queries.append(query)

        pending_nested_queries = next_pending_nested_queries

        # Final processing of any remaining nested queries
        for query in pending_nested_queries:
            process_and_store_query(
                query, db_queries, is_nested=True, new_query=query["SQL"]
            )

        # Save processed queries by db_id
        os.makedirs(output_folder, exist_ok=True)
        for db_id, queries in db_queries.items():
            output_file = os.path.join(output_folder, f"{db_id}_parsed_queries.json")
            with open(output_file, "w") as outfile:
                json.dump(queries, outfile, indent=4)
            logging.info(f"Saved {len(queries)} queries to {output_file}")


def process_and_store_query(
    query_entry: Dict,
    database_queries: Dict[str, List[Dict]],
    is_nested_query: bool = False,
    flattened_query: str = "",
) -> None:
    """
    Extract tables and join conditions from a SQL query and store the processed data.

    Args:
        query_entry (Dict): A dictionary containing the SQL query and associated metadata.
        database_queries (Dict[str, List[Dict]]): A dictionary mapping db_id to a list of processed queries.
        is_nested_query (bool, optional): Whether the query is nested. Defaults to False.
        flattened_query (str, optional): The flattened version of a nested query. Defaults to "".
    """
    sql_query = flattened_query if is_nested_query else query_entry.get("SQL", "")
    database_id = query_entry.get("db_id", "")
    original_sql_query = query_entry.get("SQL", "")

    tables, aliases_to_table_map = extract_tables(sql_query)
    join_conditions_with_aliases = get_joins_and_tables(sql_query)
    join_conditions_without_aliases = replace_aliases_in_join_conditions(
        join_conditions_with_aliases, aliases_to_table_map
    )

    # Remove duplicates from tables
    seen_tables = set()
    unique_tables = []
    for table in tables:
        table_name = table["table_name"]
        if table_name not in seen_tables:
            seen_tables.add(table_name)
            unique_tables.append(
                {"table_name": table_name, "table_alias": table["table_alias"]}
            )

    # Filter and deduplicate join relations
    seen_joins = set()
    deduplicated_join_conditions = []
    for condition in join_conditions_without_aliases:
        left_side, right_side = map(str.strip, condition.split("="))
        table1, column1 = map(str.strip, left_side.split("."))
        table2, column2 = map(str.strip, right_side.split("."))

        if table1 != table2:  # Skip self-joins
            join_pair = frozenset([(table1, column1), (table2, column2)])
            if join_pair not in seen_joins:
                seen_joins.add(join_pair)
                deduplicated_join_conditions.append(
                    f"{table1}.{column1} = {table2}.{column2}"
                )

    # Prepare processed entry
    processed_entry_data = {
        "Tables": unique_tables,
        "Join_relations_with_aliases": join_conditions_with_aliases,
        "Join_relations_without_aliases": join_conditions_without_aliases,
        "db_id": database_id,
        "SQL": original_sql_query,
        "question": query_entry.get("question", ""),
        "evidence": query_entry.get("evidence", ""),
        "difficulty": query_entry.get("difficulty", ""),
        "flattened_query": flattened_query if is_nested_query else "",
    }

    # Store in db_data
    database_queries.setdefault(database_id, []).append(processed_entry_data)


def main() -> None:
    input_file = ""
    output_folder = ""

    process_queries(input_file=input_file, output_folder=output_folder)


if __name__ == "__main__":
    main()

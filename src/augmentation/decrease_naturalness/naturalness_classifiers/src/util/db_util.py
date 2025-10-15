"""
Copyright 2024 Kyle Luoma

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import pyodbc
import json


def do_query(
    query: str,
    database_name: str,
    debug: bool = False,
    convert_bytes_to_str: bool = False,
    db_list_file: str = ".local/dbinfo.json",
) -> pd.DataFrame:
    """
    Executes a SQL query on a specified database and returns the result as a pandas DataFrame.
    Parameters:
    query (str): The SQL query to execute.
    database_name (str): The name of the database to connect to.
    debug (bool, optional): If True, prints debug information. Defaults to False.
    convert_bytes_to_str (bool, optional): If True, converts byte values to strings using cp437 encoding. Defaults to False.
    db_list_file (str, optional): The path to the database list file. Defaults to ".local/dbinfo.json".
    Returns:
    pd.DataFrame: A DataFrame containing the query results. If the query returns no results, an empty DataFrame is returned.
    """
    if debug:
        print("DB UTIL.DO_QUERY: Creating connection")
    conn = connect_to_db(database_name, db_list_file=db_list_file)
    cursor = conn.cursor()
    query_string = query

    if debug:
        print("DB UTIL.DO_QUERY: Executing query")
        print(query)
    cursor.execute(query_string)
    if debug:
        print("DB UTIL.DO_QUERY: Executing query FINISHED")
    if cursor.description == None:
        return pd.DataFrame()
    columns = [column[0] for column in cursor.description]

    # make sure column names are unique
    for i in range(0, len(columns)):
        if i > 0:
            if columns[i] in columns[:i]:
                columns[i] = columns[i] + str(i)

    result_dict = {}
    for column in columns:
        result_dict[column] = []

    for row in cursor:
        for i in range(len(columns)):
            value = row[i]
            if convert_bytes_to_str and isinstance(value, bytes):
                value = value.decode("cp437")
            result_dict[columns[i]].append(row[i])

    cursor.close()
    conn.close()

    try:
        df = pd.DataFrame(result_dict)
    except ValueError as e:
        print("Encountered a problem attempting to translate results to a DataFrame")
        print(e)
        for key in result_dict:
            print(key)
            print(len(result_dict[key]))
        df = pd.DataFrame()

    return df


def connect_to_db(
    database_name: str, db_list_file=".local/dbinfo.json"
) -> pyodbc.Connection:
    """
    Establishes a connection to the specified database using the provided database name and a JSON file containing database information.
    Args:
        database_name (str): The name of the database to connect to.
        db_list_file (str, optional): The path to the JSON file containing database connection information. Defaults to '.local/dbinfo.json'.
    Returns:
        pyodbc.Connection: A connection object to the specified database.
    Raises:
        FileNotFoundError: If the db_list_file does not exist.
        KeyError: If the database information is not found in the db_list_file.
        pyodbc.Error: If there is an error connecting to the database.
    """

    database = database_name

    db_list = json.load(open(db_list_file))

    for entry in db_list:
        if entry["database"] == database:
            db_info = entry
            break

    if "credential_type" not in db_info.keys():
        connection_string = "DSN=%s; database=%s" % (
            db_info["DSN"],
            db_info["database"],
        )
    elif db_info["credential_type"] == "username":
        connection_string = "DRIVER={" + db_info["driver"] + "};"
        connection_string += (
            f"SERVER={db_info['server']};DATABASE={db_info['database']};"
        )
        connection_string += (
            f"UID={db_info['username']};PWD={db_info['password']};Encrypt=no"
        )

    conn = pyodbc.connect(connection_string, timeout=30, autocommit=True)
    conn.timeout = 10
    return conn


def get_tables_and_columns_from_sql_server_db(
    db_name: str,
    append_col_types: bool = True,
    table_list: list = None,
    uppercase: bool = False,
    schema: str = "dbo",
) -> dict:
    """
    Returns a dictionary of tables and columns from a SQL Server database.

    Parameters
    ----------
    db_name : str
        The name of the database to use for the schema.
    append_col_types : bool, optional
        Whether to append the column types to the column names. The default is True.
    table_list : list, optional
        A list of tables to limit the request to. The default is None. If a value is passed, only the columns of the tables in the list will be returned.
    schema : schema name, defaults to dbo. Modify to query natural views
    """

    if table_list != None:
        table_list = [t.upper() for t in table_list]

    conn = connect_to_db(db_name)
    cursor = conn.cursor()
    schema = schema
    table_type = "BASE TABLE"
    query_string = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = '{table_type}' AND table_schema = '{schema}' and TABLE_NAME not like '%_cross_reference';"
    cursor.execute(query_string)
    tables_and_columns = {}
    for row in cursor:
        if uppercase:
            table_name = row[0].upper()
        else:
            table_name = row[0]
        if table_list == None or table_name in table_list:
            tables_and_columns[table_name] = []

    for table in tables_and_columns:
        query_string = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' AND TABLE_SCHEMA = '{schema}';"
        cursor.execute(query_string)
        for row in cursor:
            if uppercase:
                column_name = row[0].upper()
            else:
                column_name = row[0]
            if append_col_types:
                tables_and_columns[table].append(column_name + " " + row[1])
            else:
                tables_and_columns[table].append(column_name)
    return tables_and_columns


def get_tables_and_columns_df(tables_and_col_dict: dict) -> pd.DataFrame:
    """
    Converts a dictionary of tables and their columns into a pandas DataFrame.
    Args:
        tables_and_col_dict (dict): A dictionary where keys are table names and values are lists of column names.
    Returns:
        pd.DataFrame: A DataFrame with two columns: 'table' and 'column', where each row represents a table-column pair.
    """

    tables = []
    columns = []
    for table in tables_and_col_dict:
        for column in tables_and_col_dict[table]:
            tables.append(table)
            columns.append(column)
    return pd.DataFrame({"table": tables, "column": columns})


def make_db_schema_prompt(
    db_name: str,
    db_type: str = "MS SQL Server",
    task: str = "query",  # or 'tables'
    table_list: list = None,
    column_list: list = None,
    identifier_tags: bool = False,
    schema: str = "dbo",
    table_col_function=get_tables_and_columns_from_sql_server_db,
):
    """Creates a zero shot prompt with schema knowledge for a LLM query,
        using the schema of the database as a starting point.

    Parameters
    ----------
    db_name : str
        The name of the database to use for the schema.
    db_type : str, optional
        The type of database. The default is 'MS SQL Server'.
    task : str, optional
        The task to be performed. The default is 'query'.
        Options include:
            - query: a query to be executed on the database
            - tables: retrieve the most-likely tables required to form a query
    table_list : list, optional
        A list of tables to include in the prompt. The default is None.
    column_list : list, optional
        A list of columns to include in the prompt. The default is None.
    identifier_tags : bool, optional
        Whether to include tags around table and column names. The default is False.

    Returns
    -------
    prompt : str
        The prompt to be used for the LLM query.
    """
    if table_list != None:
        table_list = [t.upper() for t in table_list]
    if column_list != None:
        column_list = [c.upper() for c in column_list]

    tables_and_columns = table_col_function(
        db_name, table_list=table_list, uppercase=True, schema=schema
    )

    if identifier_tags:
        table_tags = ("<TABLE_NAME>", "</TABLE_NAME>")
        column_tags = ("<COLUMN_NAME>", "</COLUMN_NAME>")
    else:
        table_tags = column_tags = ("", "")

    task_text = json.load(open("./prompts/gpt-query-generation.json"))
    task_info = {
        "task": "query",
        "preamble": "For the database described next, provide only a sql query. do not include any text that is not valid SQL.\n",
        "taskdescription": "#\n### a sql query, written in the {} dialect, to answer the question: ".format(
            db_type
        ),
    }
    for tt in task_text:
        if tt["task"] == task:
            task_info = tt
            break

    prompt = task_info["preamble"].replace("__db_type__", db_type)

    prompt += "#\n#Database: " + db_name + "\n#\n"
    prompt += "#{} tables, with their properties:\n#\n".format(db_type)
    for table in tables_and_columns:
        if table_list is not None:
            if table not in table_list and table.upper() not in table_list:
                # print(f"Skipping {table} because it is not in the table list")
                continue
        prompt += f"#{table_tags[0]}{table}{table_tags[1]}("
        column_count = 0
        for column in tables_and_columns[table]:
            if len(column.split(" ")) >= 2:
                word_list = column.split(" ")
                column_name = word_list[0 : len(word_list) - 1]
                column_name = " ".join(column_name)
                datatype = word_list[len(word_list) - 1]
            elif len(column.split(" ")) == 1:
                column_name = column
                datatype = ""
            if column_list is not None and column_name not in column_list:
                # print(f"Skipping {column_name} because it is not in the column list")
                continue
            prompt += f" {column_tags[0]}{column_name}{column_tags[1]} {datatype},"
            column_count += 1
        if column_count == 0:
            prompt = prompt + "*"
        else:
            prompt = prompt[: len(prompt) - 1]
        prompt += ")\n"
    prompt += task_info["taskdescription"].replace("__db_type__", db_type)

    return prompt


def get_table_cardinality(db_name, table_name) -> int:
    """
    Retrieves the number of rows (cardinality) in a specified table within a given database.
    Args:
        db_name (str): The name of the database to connect to.
        table_name (str): The name of the table whose cardinality is to be retrieved.
    Returns:
        int: The number of rows in the specified table.
    """
    conn = connect_to_db(db_name)
    cursor = conn.cursor()
    query_string = f"SELECT COUNT(*) FROM [{table_name}];"
    cursor.execute(query_string)
    for row in cursor:
        return row[0]


def get_db_tables_cardinality(db_name, save_to_temp_folder=False) -> pd.DataFrame:
    """
    Retrieves the cardinality (number of rows) for each table in the specified database.
    Args:
        db_name (str): The name of the database to query.
        save_to_temp_folder (bool, optional): If True, saves the resulting DataFrame to a CSV file in the './temp_data/' directory. Defaults to False.
    Returns:
        pd.DataFrame: A DataFrame containing the table names and their corresponding cardinalities.
    Raises:
        Exception: If there is an error retrieving the cardinality for a table, it will be caught and printed.
    """

    db_tables = get_tables_and_columns_from_sql_server_db(db_name)
    tables = []
    cardinalities = []
    print(f"Getting cardinalities for tables in {db_name}")
    for table in db_tables:
        print(f"Getting cardinality for {table}")
        try:
            cardinalities.append(get_table_cardinality(db_name, table))
            tables.append(table)
        except Exception as e:
            print(f"Error getting cardinality for {table}")
            print(e)

    df = pd.DataFrame({"table": tables, "cardinality": cardinalities})
    if save_to_temp_folder:
        df.to_csv(f"./temp_data/{db_name}-cardinality.csv", index=False)


def make_db_schema_prompt_test():
    prompt = make_db_schema_prompt(
        "NTSB", task="query", identifier_tags=False, use_natural_schema=False
    )
    print(prompt)


if __name__ == "__main__":
    # get_db_tables_cardinality('SBODemoUS', True)
    # res = get_tables_and_columns_from_sql_server_db("NTSB")
    make_db_schema_prompt_test()
    # conn = connect_to_db("NTSB")
    # cursor = conn.cursor()
    # cursor.columns("TIREDAMAGE")
    # for row in cursor:
    #     print(tuple(row))

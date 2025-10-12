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
from util import db_util
import sqlite3
import json


def do_query(
    query: str, database_name: str, db_list_file: str = ".local/spider_dbinfo.json"
) -> pd.DataFrame:
    conn = connect_to_db(database_name=database_name, db_list_file=db_list_file)
    cursor = conn.cursor()
    cursor.execute(query)
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
            result_dict[columns[i]].append(row[i])

    cursor.close()
    conn.close()
    try:
        df = pd.DataFrame(result_dict)
    except ValueError as e:
        print("db_util_sqlite.do_query raised an exception", e)
        print(result_dict)
        df = None
    return df


def connect_to_db(
    database_name, db_list_file=".local/spider_dbinfo.json"
) -> sqlite3.Connection:

    db_list = json.load(open(db_list_file))

    db_info = None
    for entry in db_list:
        if entry["database"] == database_name:
            db_info = entry
            break
    if db_info == None:
        raise FileNotFoundError

    conn = sqlite3.connect(db_info["path"])
    return conn


def get_tables_and_columns_from_sqlite_db(
    db_name: str,
    append_col_types: bool = True,
    table_list=None,
    uppercase=False,
    schema="",
    db_list_file=".local/spider_dbinfo.json",
):
    conn = connect_to_db(db_name, db_list_file=db_list_file)
    tables = do_query(
        "select tbl_name from sqlite_master where type='table'",
        database_name=db_name,
        db_list_file=db_list_file,
    )
    tables_and_columns = {}
    for table_name in tables.tbl_name:
        table_info = do_query(
            f"PRAGMA table_info({table_name})",
            database_name=db_name,
            db_list_file=db_list_file,
        )
        columns = []
        for row in table_info.itertuples():
            column = row.name
            if append_col_types:
                column += " " + row.type
            columns.append(column)
        tables_and_columns[table_name] = columns
    return tables_and_columns


def make_db_schema_prompt(
    db_name,
    db_type="sqlite",
    task="query",  # or 'tables'
    table_list=None,
    column_list=None,
    identifier_tags=False,
    use_natural_schema=False,
):
    return db_util.make_db_schema_prompt(
        db_name,
        db_type,
        task,
        table_list,
        column_list,
        identifier_tags,
        use_natural_schema,
        get_tables_and_columns_from_sqlite_db,
    )


def get_tables_and_columns_df(tables_and_col_dict):
    return db_util.get_tables_and_columns_df(tables_and_col_dict)


if __name__ == "__main__":
    info = do_query("PRAGMA table_info(Student)", database_name="pets_1")
    print(info)
    res = get_tables_and_columns_from_sqlite_db(db_name="pets_1")
    prompt = make_db_schema_prompt(db_name="pets_1")
    print(prompt)

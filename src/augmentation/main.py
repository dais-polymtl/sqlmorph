import os

import pandas as pd

from rule_1_and_2 import (
    retrieve_all_dev_patterns,
    load_schema,
    extend_and_filter_subgraphs,
    ensure_directory
)
from rule_1_and_2.persistence import save_rule_data, process_rule_folder
from rule_1_and_2.query_execution import process_queries_for_rules
from rule_1_and_2.query_generation import generate_inner_join_query_with_test
from rule_1_and_2.visualization import save_pre_rule_subgraphs


def run_rule_1_and_2():
    # Configuration
    db_id = 'financial'
    data_folder = 'data/'
    rule_inputs_folder = os.path.join(data_folder, 'rule_inputs', 'rules_1_2', db_id)
    graph_data_folder = os.path.join(data_folder, 'graph_data', 'bird_graphs', 'pickles')
    rule_outputs_folder = os.path.join(data_folder, 'rule_outputs', 'rules_1_2')
    images_output_base = os.path.join(rule_outputs_folder, 'images')
    ensure_directory(images_output_base)

    # Retrieve all subgraphs
    database_subgraphs = retrieve_all_dev_patterns(rule_inputs_folder)
    print(f"Found {len(database_subgraphs)} subgraphs.")

    # Load the schema
    schema_path = os.path.join(graph_data_folder, f'{db_id}_graph.pkl')
    schema = load_schema(schema_path)

    # Save pre-rule subgraphs visualization
    save_pre_rule_subgraphs(
        schema=schema,
        subgraphs=database_subgraphs,
        rule_name='rule_1',
        db_id=db_id,
        output_base_dir=images_output_base
    )
    save_pre_rule_subgraphs(
        schema=schema,
        subgraphs=database_subgraphs,
        rule_name='rule_2',
        db_id=db_id,
        output_base_dir=images_output_base
    )

    # Extend and filter subgraphs based on rules
    rule_1_patterns, rule_2_patterns = extend_and_filter_subgraphs(database_subgraphs, schema)
    print(f"Rule 1 Patterns: {len(rule_1_patterns)}")
    print(f"Rule 2 Patterns: {len(rule_2_patterns)}")

    # Load query statistics
    query_stats_path = os.path.join(data_folder, 'rule_inputs', 'rules_1_2', 'query_statistics.csv')
    df = pd.read_csv(query_stats_path)

    # Generate queries for Rule 1 and Rule 2
    rule_1_data = []
    rule_2_data = []

    for pattern in rule_1_patterns:
        result = generate_inner_join_query_with_test(schema, pattern, db_id, df)
        rule_1_data.append({
            'subgraph': pattern,
            'main_query': result['main_query'],
            'test_query': result['test_query'],
            'projection_columns': result['projection_columns'],
            'filtering_columns': result['filtering_columns']
        })

    for pattern in rule_2_patterns:
        result = generate_inner_join_query_with_test(schema, pattern, db_id, df)
        rule_2_data.append({
            'subgraph': pattern,
            'main_query': result['main_query'],
            'test_query': result['test_query'],
            'projection_columns': result['projection_columns'],
            'filtering_columns': result['filtering_columns']
        })

    # Execute and update queries
    database_path_template = os.path.join(data_folder, 'benchmarks', 'Bird', 'dev_databases', db_id, f'{db_id}.sqlite')
    process_queries_for_rules(rule_1_data, db_id, database_path_template)
    process_queries_for_rules(rule_2_data, db_id, database_path_template)

    # Save outputs
    save_rule_data(rule_1_data, 'rule_1', rule_outputs_folder, db_id)
    save_rule_data(rule_2_data, 'rule_2', rule_outputs_folder, db_id)
    print("Data saved successfully.")

    # Global ordering and final output generation
    json_output_rule_1 = os.path.join(rule_outputs_folder, 'dev_1_1.json')
    sql_output_rule_1 = os.path.join(rule_outputs_folder, 'dev_1_1.sql')
    json_output_rule_2 = os.path.join(rule_outputs_folder, 'dev_2_1.json')
    sql_output_rule_2 = os.path.join(rule_outputs_folder, 'dev_2_1.sql')

    process_rule_folder(rule_outputs_folder, graph_data_folder, 'rule_1', json_output_rule_1, sql_output_rule_1)
    process_rule_folder(rule_outputs_folder, graph_data_folder, 'rule_2', json_output_rule_2, sql_output_rule_2)

    print("Files dev_1_1.json, dev_1_1.sql, dev_2_1.json, and dev_2_1.sql generated successfully.")


if __name__ == "__main__":
    run_rule_1_and_2()

import os
import pickle
import json

from .llm_inference import generate_explicit_question, generate_dev_set_like_question

def save_rule_data(rule_data, rule, folder, db_id):
    """
    Save rule data to a pickle file with the rule name appended.

    Args:
        rule_data (list): The rule data to save.
        rule (str): The rule name, either 'rule_1' or 'rule_2'.
        folder (str): The folder path to save the file.
        db_id (str): The database ID.
    """
    # Create output file path with rule name
    output_file = os.path.join(folder, f"{db_id}_outputs_{rule}.pkl")

    # Save the list to a pickle file without modifying or adding extra fields
    with open(output_file, 'wb') as f:
        pickle.dump(rule_data, f)


def process_rule_folder(rule_folder, graph_data_folder, rule_name, json_output_file, sql_output_file, question_type):
    """
    Process and compute metrics for all databases in a rule folder.

    Args:
        rule_folder (str): Path to the folder containing rule pickle files.
        graph_data_folder (str): Path to the folder containing graph data.
        rule_name (str): Name of the rule ('rule_1' or 'rule_2').
        json_output_file (str): Path to save the JSON output.
        sql_output_file (str): Path to save the SQL output.
        question_type (str): Type of question to generate ('explicit' or 'dev_set_like').
    """
    import networkx as nx  # Imported here to avoid circular dependencies
    from .graph_processing import calculate_subgraph_centrality, is_cyclic
    from .data_retrieval import load_schema

    # Collect all pickle files in the folder
    if rule_name == 'rule_1':
        files = [f for f in os.listdir(rule_folder) if f.endswith('outputs_rule_1.pkl')]
    else:
        files = [f for f in os.listdir(rule_folder) if f.endswith('outputs_rule_2.pkl')]

    global_rule_data = []

    # Process each pickle file
    for file in files:
        # Extract db_id by removing '_outputs_rule_X.pkl'
        if rule_name == 'rule_1':
            db_id = file.replace('_outputs_rule_1.pkl', '')
        else:
            db_id = file.replace('_outputs_rule_2.pkl', '')
        schema_path = os.path.join(graph_data_folder, f'{db_id}_graph.pkl')

        # Load the schema graph
        schema = load_schema(schema_path)

        # Compute centrality for each node in the schema
        centrality = nx.degree_centrality(schema)

        # Load the rule data from pickle file
        with open(os.path.join(rule_folder, file), 'rb') as f:
            data_list = pickle.load(f)

        # Compute metrics for each subgraph in the pickle data
        for data in data_list:
            subgraph = data.get('subgraph', None)

            # If there's no subgraph, skip this entry
            if subgraph is None:
                continue

            # Extract and calculate subgraph metrics
            num_nodes = subgraph.number_of_nodes()
            num_connections = subgraph.number_of_edges()
            cyclic = is_cyclic(subgraph)
            centrality_score = calculate_subgraph_centrality(subgraph, centrality)

            # Append relevant data to the global list, with metrics
            global_rule_data.append({
                'db_id': db_id,
                'main_query': data['main_query'],
                'num_nodes': num_nodes,
                'num_connections': num_connections,
                'cyclic': cyclic,
                'centrality_score': centrality_score
            })

    # Sort all data globally in descending order based on the subgraph metrics
    global_rule_data_sorted = sorted(
        global_rule_data,
        key=lambda x: (x['num_nodes'], x['num_connections'], x['cyclic'], x['centrality_score']),
        reverse=True  # Sort from biggest to smallest
    )

    # Generate JSON and SQL files
    json_data = []
    sql_queries = []

    for idx, entry in enumerate(global_rule_data_sorted):
        if question_type == 'explicit':
            question = ""
            # question = generate_explicit_question(entry['main_query'])
        if question_type == 'dev_set_like':
            question = ""
            # question = generate_dev_set_like_question(entry['main_query'])
        json_data.append({
            "question_id": idx,
            "db_id": entry['db_id'],
            "question": question,  # Placeholder
            "evidence": "",  # Placeholder
            "SQL": entry['main_query'],
            "difficulty": "challenging"  # Default difficulty
        })
        sql_queries.append(f"{entry['main_query']}\t{entry['db_id']}")

    # Write JSON output
    with open(json_output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    # Write SQL output
    with open(sql_output_file, 'w') as sql_file:
        sql_file.write("\n".join(sql_queries))

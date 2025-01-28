import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime


def display_and_save_subgraph(schema, subgraph, subgraph_number, rule, time_period, db_id, output_base_dir):
    """
    Display and save a subgraph within the main schema.

    Args:
        schema (nx.Graph): The main graph schema.
        subgraph (nx.Graph): The subgraph to display.
        subgraph_number (int): The number of the subgraph for labeling.
        rule (str): The rule this subgraph belongs to (either 'rule_1' or 'rule_2').
        time_period (str): Either 'pre_rule' or 'post_rule' indicating the time period.
        db_id (str): The database ID.
        output_base_dir (str): Base directory to save images.
    """
    # Create the directory structure: images/rule_x/db_id/pre_rule or post_rule
    base_dir = os.path.join(output_base_dir, rule, db_id, time_period)
    os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists even if the subgraph is None or empty

    # Create a figure for the plot
    plt.figure(figsize=(10, 10))

    # Draw the entire schema in light gray
    pos = nx.spring_layout(schema)  # Node positions using spring layout
    nx.draw(schema, pos, with_labels=True, node_color='lightgray', edge_color='black',
            node_size=1000, font_size=10)

    # Check if the subgraph is None or has nodes
    if subgraph is not None and subgraph.number_of_nodes() > 0:  # Draw the subgraph only if it contains nodes
        # Draw the subgraph nodes and edges
        nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=1000)
        nx.draw_networkx_edges(subgraph, pos, edge_color='blue', width=2)

        # Create edge labels for the subgraph, fallback to schema if needed
        edge_labels = {
            (u, v): edge_data.get('label', schema[u][v].get('label', 'No Label'))
            for u, v, edge_data in subgraph.edges(data=True)
        }

        # Draw the edge labels
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red', font_size=7)

    # Add title indicating the subgraph number
    plt.title(f'Subgraph {subgraph_number}')

    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f'subgraph_{subgraph_number}_{timestamp}.png'
    image_path = os.path.join(base_dir, image_filename)

    # Save the image
    plt.savefig(image_path)
    plt.close()  # Close the plot to free up memory

    print(f'Subgraph {subgraph_number} saved at: {image_path}')


def save_pre_rule_subgraphs(schema, subgraphs, rule_name, db_id, output_base_dir):
    """
    Save and display pre-rule subgraphs for a specific rule.

    Parameters:
    - schema (nx.Graph): The main schema graph.
    - subgraphs (list): List of subgraphs to process.
    - rule_name (str): Name of the rule (e.g., 'rule_1', 'rule_2').
    - db_id (str): Database identifier for saving images.
    - output_base_dir (str): Base directory to save images.
    """
    subgraphs_sorted = sorted(subgraphs, key=lambda x: x.number_of_nodes())  # Sort by number of nodes
    for i, subgraph in enumerate(subgraphs_sorted, start=1):
        display_and_save_subgraph(
            schema=schema,
            subgraph=subgraph,
            subgraph_number=i,
            rule=rule_name,
            time_period='pre_rule',
            db_id=db_id,
            output_base_dir=output_base_dir
        )

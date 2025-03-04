import json
from column import Column
from table import Table
from functional_dependencies import ForeignKey
from collections import defaultdict
import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle


class Beaver_Schema:
    def __init__(self, database_name: str, absolute_file_path: str):
        """
        Represents a database schema.

        Args:
            database_name (str): The name of the database.
            absolute_file_path (str): Absolute file path containing schema information in JSON format.
        """
        self.database_name = database_name
        self.absolute_file_path = absolute_file_path
        self.tables = []
        self.foreign_keys = []

        with open(self.absolute_file_path, "r") as file:
            data = json.load(file)

        # Filter data for the given database_name
        relevant_data = {
            k: v for k, v in data.items() if v.get("db_id") == self.database_name
        }

        if relevant_data:
            self._process_data(relevant_data)

    def _process_data(self, relevant_data):
        """
        Process the new data format and populate tables and foreign keys.
        """
        for table_key, table_info in relevant_data.items():
            table_name = table_info["table_name_original"]
            column_names_original = table_info["column_names_original"]
            column_types = table_info["column_types"]
            primary_keys = table_info.get("primary_key", [])  # Default to empty list if missing

            # Create Column objects
            columns = [
                Column(table_name, column_names_original[i], column_types[i])
                for i in range(len(column_names_original))
            ]

            # Identify primary key columns
            table_primary_keys = [
                col for col in columns if col.column_name in primary_keys
            ]

            # Create Table object
            table = Table(table_name, columns, table_primary_keys)
            self.tables.append(table)

            # Process foreign keys 
        for table_key, table_info in relevant_data.items():
            table_name = table_info["table_name_original"]
            foreign_keys = table_info.get("foreign_key", [])

            for fk_info in foreign_keys:
                referencing_column_name = fk_info["column_name"]
                referenced_table_name = fk_info["referenced_table_name"].split("#sep#")[-1].strip()
                referenced_column_name = fk_info["referenced_column_name"].strip()
                referencing_column = next(
                    (
                        col
                        for table in self.tables
                        for col in table.columns
                        if table.table_name == table_name
                        and col.column_name == referencing_column_name
                    ),
                    None,
                )
                referenced_column = next(
                    (
                        col
                            for table in self.tables
                            for col in table.columns
                            if table.table_name == referenced_table_name
                            and col.column_name == referenced_column_name
                        ),
                        None,
                    )

                if referencing_column and referenced_column:
                    fk = ForeignKey(
                        table_name,
                        referenced_table_name,
                        referencing_column,
                        referenced_column,
                    )
                    self.foreign_keys.append(fk)
                

    def __str__(self) -> str:
        schema_description = f"The database schema '{self.database_name}' consists of {len(self.tables)} tables:\n"
        for table in self.tables:
            schema_description += (
                f"\nTable '{table.table_name}' has {len(table.columns)} columns:\n"
            )
            for column in table.columns:
                schema_description += (
                    f"  - {column.column_name} ({column.column_type})\n"
                )

            table_primary_keys = defaultdict(list)
            for pk in table.primary_keys:
                if pk.table_name == table.table_name:
                    table_primary_keys[pk.table_name].append(pk.column_name)

            for table_name, primary_keys in table_primary_keys.items():
                schema_description += f"  Primary key(s) ({len(primary_keys)}): {', '.join(primary_keys)}\n"

            table_foreign_keys = [
                (
                    fk.column.column_name,
                    fk.referenced_table,
                    fk.referenced_column.column_name,
                )
                for fk in self.foreign_keys
                if fk.referencing_table == table.table_name
            ]

            if table_foreign_keys:
                schema_description += f"  Foreign key(s) ({len(table_foreign_keys)}):\n"
                for fk_column, fk_table, fk_referenced_column in table_foreign_keys:
                    schema_description += f"    - {fk_column} referencing column: {fk_referenced_column} in table: {fk_table}\n"

        return schema_description

    def create_graph(self):
        G = nx.Graph()

        # Add nodes with attributes
        for table in self.tables:
            attributes = {
                "table_name": table.table_name,
                "column_names": [column.column_name for column in table.columns],
                "column_types": [column.column_type for column in table.columns],
                "primary_keys": [pk.column_name for pk in table.primary_keys],
            }
            G.add_node(table.table_name, **attributes)

        # Add edges with labels
        edge_labels = {}
        for fk in self.foreign_keys:
            if fk.referencing_table != fk.referenced_table:  # Avoid self-loops
                edge_key = (fk.referencing_table, fk.referenced_table)
                edge_label = f"{fk.referencing_table}.{fk.column.column_name} = {fk.referenced_table}.{fk.referenced_column.column_name}"
                if edge_key in edge_labels:
                    edge_labels[edge_key].append(edge_label)
                else:
                    edge_labels[edge_key] = [edge_label]

        for i, fk1 in enumerate(self.foreign_keys):
            for fk2 in self.foreign_keys[i + 1 :]:
                if (
                    fk1 != fk2
                    and fk1.referenced_table == fk2.referenced_table
                    and fk1.referencing_table != fk2.referencing_table
                ):
                    referencing_table_1 = fk1.referencing_table
                    referencing_table_2 = fk2.referencing_table
                    edge_key = (referencing_table_1, referencing_table_2)
                    edge_label = f"{referencing_table_1}.{fk1.column.column_name} = {referencing_table_2}.{fk2.column.column_name}"
                    if (
                        edge_key not in edge_labels
                        and edge_key[::-1] not in edge_labels
                    ):
                        edge_labels[edge_key] = [edge_label]
                    else:
                        if edge_key in edge_labels:
                            edge_labels[edge_key].append(edge_label)
                        else:
                            edge_labels[edge_key[::-1]].append(edge_label)
        
        for (table1, table2), labels in edge_labels.items():
            combined_label = "; ".join(labels)
            G.add_edge(table1, table2, label=combined_label)

        return G

    def save_graph_and_image(self, output_dir):
        graph = self.create_graph()

        pickle_dir = os.path.join(output_dir, "pickles")
        os.makedirs(pickle_dir, exist_ok=True)
        pickle_file_path = os.path.join(pickle_dir, f"{self.database_name}_graph.pkl")
        with open(pickle_file_path, "wb") as file:
            pickle.dump(graph, file)

        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_file_path = os.path.join(image_dir, f"{self.database_name}_graph.png")

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=1500,
            node_color="lightblue",
            font_size=12,
            font_weight="bold",
            edge_color="blue",
        )

        plt.title(f"Graph for Database '{self.database_name}'")
        plt.tight_layout(pad=1.0)
        plt.savefig(image_file_path)
        plt.close()

 
def update_dw_graph(output_dir):
    pickle_file_path = os.path.join(output_dir, "pickles", "dw_graph.pkl")
    with open(pickle_file_path, "rb") as file:
        G = pickle.load(file)
    
    with open("dw_join_keys.json", "r") as file:
        dw_joins = json.load(file)
    
    edge_labels = nx.get_edge_attributes(G, "label")
    for join_pair in dw_joins:
        table1, key1 = join_pair[0].split(".")
        table2, key2 = join_pair[1].split(".")

        # Skip self-joins (where both tables are the same)
        if table1 == table2:
            continue
        
        # Define the edge key and label
        edge_key_1 = (table1, table2)
        edge_key_2 = (table2, table1)

        edge_label = f"{table1}.{key1} = {table2}.{key2}"
        # Check if the reversed edge already exists
        if edge_key_2 in edge_labels:
            continue  # Skip this iteration if reversed edge exists
        
        if edge_key_1 in edge_labels:
            edge_labels[edge_key_1] += "; " + edge_label
        else:
            edge_labels[edge_key_1] = edge_label
        G.add_edge(table1, table2, label=edge_labels[edge_key_1])
    
    with open(pickle_file_path, "wb") as file:
        pickle.dump(G, file)
    
    image_file_path = os.path.join(output_dir, "images", "dw_graph.png")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="lightblue",
        font_size=12,
        font_weight="bold",
        edge_color="blue",
    )
    plt.title("Updated Graph for Database 'dw'")
    plt.tight_layout(pad=1.0)
    plt.savefig(image_file_path)
    plt.close()

def main():
    with open("dev_tables.json", "r") as file:
        data = json.load(file)

        for db_info in data.values():
            database_name = db_info.get("db_id", "")
            schema = Beaver_Schema(database_name, "dev_tables.json")
            schema.save_graph_and_image("beaver_graphs")
    
    update_dw_graph("beaver_graphs")


if __name__ == "__main__":
    main()
import json
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from column import Column
from .functional_dependencies import ForeignKey
from .table import Table


class Spider_Schema:
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

        relevant_data = next(
            (db_info for db_info in data if db_info.get("db_id") == self.database_name),
            None,
        )

        if relevant_data:
            self._process_data(relevant_data)

    def _process_data(self, relevant_data):
        table_names_original = relevant_data.get("table_names_original", [])
        column_names_original = relevant_data.get("column_names_original", [])
        column_types = relevant_data.get("column_types", [])
        foreign_keys_original = relevant_data.get("foreign_keys", [])
        primary_keys_original = relevant_data.get("primary_keys", [])
        columns = [
            Column(table_names_original[i[0]], i[1], column_types[j])
            for j, i in enumerate(column_names_original[1:], start=1)
        ]

        self.tables = [
            Table(
                table_name,
                [column for column in columns if column.table_name == table_name],
                [
                    column
                    for column in columns
                    for pk in primary_keys_original
                    if (
                        column.table_name
                        == table_names_original[column_names_original[pk][0]]
                        and column.column_name == column_names_original[pk][1]
                    )
                ],
            )
            for table_name in table_names_original
        ]

        self.foreign_keys = [
            ForeignKey(
                columns[i[0] - 1].table_name,
                columns[i[1] - 1].table_name,
                columns[i[0] - 1],
                columns[i[1] - 1],
            )
            for i in foreign_keys_original
        ]

    def create_graph(self):
        G = nx.Graph()

        for table in self.tables:
            attributes = {
                "table_name": table.table_name,
                "column_names": [column.column_name for column in table.columns],
                "column_types": [column.column_type for column in table.columns],
                "primary_keys": [pk.column_name for pk in table.primary_keys],
            }
            G.add_node(table.table_name, **attributes)

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

        plt.figure(figsize=(16, 16))  # Increase figure size
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=800,  # Smaller nodes
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            edge_color="blue",
        )
        # Remove edge labels from the visualization
        plt.title(f"Graph for Database '{self.database_name}'")
        plt.tight_layout(pad=1.0)
        plt.savefig(image_file_path)
        plt.close()


def main():
    with open("tables.json", "r") as file:
        data = json.load(file)
        for db_info in data:
            database_name = db_info["db_id"]
            schema = Spider_Schema(database_name, "tables.json")
            schema.save_graph_and_image("spider_graphs")


if __name__ == "__main__":
    main()

import json
from column import Column
from table import Table
from functional_dependencies import ForeignKey
from collections import defaultdict
import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle


class BIRD_Schema:
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

        self.tables = []
        table_columns_dict = {table_name: [] for table_name in table_names_original}
        for column in columns:
            table_columns_dict[column.table_name].append(column)

        for table_name in table_names_original:
            table_columns = table_columns_dict[table_name]

            table_primary_keys = []
            for pk in primary_keys_original:
                if isinstance(pk, list):
                    for pki in pk:
                        pk_table_name = table_names_original[
                            column_names_original[pki][0]
                        ]
                        if pk_table_name == table_name:
                            pk_column_name = column_names_original[pki][1]
                            for column in table_columns:
                                if column.column_name == pk_column_name:
                                    table_primary_keys.append(column)
                                    break
                else:
                    pk_table_name = table_names_original[column_names_original[pk][0]]
                    if pk_table_name == table_name:
                        pk_column_name = column_names_original[pk][1]
                        for column in table_columns:
                            if column.column_name == pk_column_name:
                                table_primary_keys.append(column)
                                break

            table = Table(table_name, table_columns, table_primary_keys)
            self.tables.append(table)

        self.foreign_keys = [
            ForeignKey(
                columns[i[0] - 1].table_name,
                columns[i[1] - 1].table_name,
                columns[i[0] - 1],
                columns[i[1] - 1],
            )
            for i in foreign_keys_original
        ]

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


def main():
    with open("data/benchmarks/Bird/dev_tables.json", "r") as file:
        data = json.load(file)

        for db_info in data:
            database_name = db_info.get("db_id", "")

            schema = BIRD_Schema(database_name, "data/benchmarks/Bird/dev_tables.json")
            if database_name == "toxicology":
                print(schema)
    #         schema.save_graph_and_image("bird_graphs")

    # with open("train_tables.json", "r") as file:
    #     data = json.load(file)

    #     for db_info in data:
    #         database_name = db_info.get("db_id", "")
    #         schema = BIRD_Schema(database_name, "train_tables.json")
    #         print(schema)

    #         schema.save_graph_and_image("bird_graphs")


if __name__ == "__main__":
    main()

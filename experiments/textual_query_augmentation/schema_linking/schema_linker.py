import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.core.model_manager import OpenAIModel
from src.core.prompt_renderer import PromptRenderer
from src.core.model_manager.model_manager import ModelManager, ModelProvider, ModelType
from src.core.model_manager.ollama_model import OllamaModel
from src.core.model_manager.utils import compose_chat_messages

from src.core.logger import logger

logger = logger.Logger(__name__)


@dataclass
class Column:
    """
    Represents a database column.
    """

    table_name: str
    column_name: str
    column_type: str
    column_description: str | None = None
    data_format: str | None = None
    value_description: str | None = None

    def __post_init__(self):
        """
        Clean and flatten metadata fields.
        """

        def clean_field(field: str | None) -> str | None:
            if field is None:
                return None
            field = str(field).strip().replace("\n", " ").replace("\r", " ")
            field = field.encode("utf-8", "ignore").decode("utf-8")
            field = " ".join(field.split())
            if field == "" or field.lower() in {"nan", "none"}:
                return None
            return field

        self.column_description = clean_field(self.column_description)
        self.data_format = clean_field(self.data_format)
        self.value_description = clean_field(self.value_description)

        if (
            self.column_description is not None
            and self.column_name.strip() == self.column_description.strip()
        ):
            self.column_description = None


@dataclass
class Table:
    """
    Represents a database table.
    """

    table_name: str
    columns: List[Column]
    primary_keys: List[Column]


@dataclass
class ForeignKey:
    """
    Represents a foreign key constraint.
    """

    referencing_table: str
    referenced_table: str
    column: Column
    referenced_column: Column


class SchemaReader:
    def __init__(self, database_name: str, tables_file_path: str, dbs_dir_path: str):
        self.database_name = database_name
        self.tables_file_path = tables_file_path
        self.dbs_dir_path = dbs_dir_path
        self.tables = []
        self.foreign_keys = []

        with open(self.tables_file_path, "r") as file:
            data = json.load(file)

        relevant_data = next(
            (db_info for db_info in data if db_info.get("db_id") == self.database_name),
            None,
        )

        if not relevant_data:
            logger.log(
                "error",
                "DATABASE_NAME_NOT_FOUND",
                {
                    "database_name": self.database_name,
                    "tables_file_path": self.tables_file_path,
                },
            )
            raise ValueError(
                f"Database name '{self.database_name}' not found in the provided tables file."
            )

        # Load column metadata from CSV files
        self.column_metadata = self._load_column_metadata()
        self._process_data(relevant_data)

    def _load_column_metadata(self) -> dict:
        """
        Load column metadata from CSV files located in the 'database_description' directory.
        """
        metadata = {}
        description_dir = os.path.join(
            self.dbs_dir_path, self.database_name, "database_description"
        )

        if not os.path.exists(description_dir):
            # print(f"Database description directory not found: {description_dir}")
            return metadata

        for csv_file in os.listdir(description_dir):
            if csv_file.endswith(".csv"):
                table_name = os.path.splitext(csv_file)[0]
                csv_path = os.path.join(description_dir, csv_file)

                df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")

                metadata[table_name] = {
                    str(row["original_column_name"]).strip(): {
                        "column_description": row.get("column_description"),
                        "data_format": row.get("data_format"),
                        "value_description": row.get("value_description"),
                    }
                    for _, row in df.iterrows()
                }
        return metadata

    def _process_data(self, relevant_data):
        table_names_original = relevant_data.get("table_names_original", [])
        column_names_original = relevant_data.get("column_names_original", [])
        column_types = relevant_data.get("column_types", [])
        foreign_keys_original = relevant_data.get("foreign_keys", [])
        primary_keys_original = relevant_data.get("primary_keys", [])

        # Process data to create Column, Table, PrimaryKey, and ForeignKey objects
        columns = [
            self._create_column(table_names_original[i[0]], i[1], column_types[j])
            for j, i in enumerate(column_names_original[1:], start=1)
        ]

        self.tables = []
        table_columns_dict = {table_name: [] for table_name in table_names_original}
        for column in columns:
            table_columns_dict[column.table_name].append(column)

        for table_name in table_names_original:
            table_columns = table_columns_dict[table_name]
            # Map column names to columns for quick lookup
            column_map = {column.column_name: column for column in table_columns}

            table_primary_keys = []
            for pk in primary_keys_original:
                pk = pk if isinstance(pk, list) else [pk]
                for pki in pk:
                    pk_table_name = table_names_original[column_names_original[pki][0]]
                    if pk_table_name == table_name:
                        pk_column_name = column_names_original[pki][1]
                        if pk_column_name in column_map:
                            table_primary_keys.append(column_map[pk_column_name])
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

    def _create_column(self, table_name, column_name, column_type) -> Column:
        """
        Create a Column object with metadata if available.
        """
        metadata = self.column_metadata.get(table_name, {}).get(column_name, {})

        data_format = metadata.get("data_format")
        if data_format:
            column_type = data_format

        return Column(
            table_name=table_name,
            column_name=column_name,
            column_type=column_type,
            column_description=metadata.get("column_description"),
            data_format=metadata.get("data_format"),
            value_description=metadata.get("value_description"),
        )

    def _format_schema_description(
        self,
        tables: List[Table],
        columns_filter: dict[str, list[str]] | None = None,
        include_schema_overview: bool = True,
    ) -> str:
        """
        Format schema description for given tables and optional column filters.

        Args:
            tables: List of tables to include in the schema
            columns_filter: Optional dict mapping table names to list of column names to include.
                          If None, all columns are included.
            include_schema_overview: Whether to include the schema overview section

        Returns:
            Formatted schema description string
        """
        schema_description = ""

        # Build table details
        for table in tables:
            schema_description += f"\nTable '{table.table_name}':\n"

            # Filter columns if a filter is provided
            table_columns = table.columns
            if columns_filter and table.table_name in columns_filter:
                table_columns = [
                    col
                    for col in table.columns
                    if col.column_name in columns_filter[table.table_name]
                ]

            for column in table_columns:
                schema_description += (
                    f"  - '{column.column_name}' ({column.column_type})\n"
                )

            # Add primary keys
            table_primary_keys = defaultdict(list)
            for pk in table.primary_keys:
                if pk.table_name == table.table_name:
                    if not columns_filter or (
                        table.table_name in columns_filter
                        and pk.column_name in columns_filter[table.table_name]
                    ):
                        table_primary_keys[pk.table_name].append(pk.column_name)

            if table_primary_keys:
                for table_name, primary_keys in table_primary_keys.items():
                    pks = "', '".join([f"'{pk}'" for pk in primary_keys])
                    schema_description += f"  Primary keys: {pks}\n"

            # Add foreign keys
            table_foreign_keys = [
                (
                    fk.column.column_name,
                    fk.referenced_table,
                    fk.referenced_column.column_name,
                )
                for fk in self.foreign_keys
                if fk.referencing_table == table.table_name
                and (
                    not columns_filter
                    or (
                        table.table_name in columns_filter
                        and fk.column.column_name in columns_filter[table.table_name]
                        and fk.referenced_table in [t.table_name for t in tables]
                    )
                )
            ]

            if table_foreign_keys:
                schema_description += "  Foreign keys:\n"
                for fk_column, fk_table, fk_referenced_column in table_foreign_keys:
                    schema_description += f"    - '{fk_column}' referencing column '{fk_referenced_column}' in table '{fk_table}'\n"

        # Add schema overview if requested
        if include_schema_overview:
            schema_description += "\nSchema Overview:\n"
            for table in tables:
                table_columns = table.columns
                if columns_filter and table.table_name in columns_filter:
                    table_columns = [
                        col
                        for col in table.columns
                        if col.column_name in columns_filter[table.table_name]
                    ]

                columns_str = ", ".join(col.column_name for col in table_columns)
                if columns_str:
                    schema_description += f"{table.table_name} ({columns_str})\n"

            schema_description += "\n"

            # Add foreign key relationships
            table_names = {t.table_name for t in tables}
            for fk in self.foreign_keys:
                if (
                    fk.referencing_table in table_names
                    and fk.referenced_table in table_names
                ):
                    if not columns_filter or (
                        fk.column.column_name
                        in columns_filter.get(fk.referencing_table, [])
                        and fk.referenced_column.column_name
                        in columns_filter.get(fk.referenced_table, [])
                    ):
                        schema_description += (
                            f"{fk.referencing_table}.{fk.column.column_name} = "
                            f"{fk.referenced_table}.{fk.referenced_column.column_name}\n"
                        )

        return schema_description

    @staticmethod
    def _clean_json_response(response: str) -> str:
        """
        Remove markdown code block markers from LLM JSON responses.
        """
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        return response.strip()

    def get_full_schema_representation(self) -> tuple[str, dict[str, list[str]]]:
        all_columns = {}
        for table in self.tables:
            all_columns[table.table_name] = [col.column_name for col in table.columns]

        schema_description = self._format_schema_description(
            tables=self.tables, include_schema_overview=True
        )

        return schema_description, all_columns

    def get_gold_filtered_schema_representation(
        self, gold_tables: set, gold_columns: set
    ) -> tuple[str, dict[str, list[str]]]:
        """
        Returns a filtered schema representation including only tables and columns present in gold_tables and gold_columns of training data.
        """
        filtered_columns = {}
        filtered_tables = []

        for table in self.tables:
            if table.table_name in gold_tables:
                filtered_tables.append(table)
                filtered_columns[table.table_name] = [
                    col.column_name
                    for col in table.columns
                    if col.column_name in gold_columns
                ]

        schema_description = self._format_schema_description(
            tables=filtered_tables,
            columns_filter=filtered_columns,
            include_schema_overview=True,
        )

        return schema_description, filtered_columns

    def get_number_of_tables(self) -> int:
        return len(self.tables)

    def get_number_of_columns(self) -> int:
        return sum(len(table.columns) for table in self.tables)

    def extract_relevant_tables(
        self,
        user_question: str,
        evidence: str | None,
        llm_provider: ModelProvider,
        model_name: OpenAIModel | OllamaModel,
    ) -> (list[Table], str):
        """
        Extract relevant tables.
        """
        prompt_renderer = PromptRenderer(
            templates_dir_path="./src/experiments/augmentation/decrease_naturalness/schema_linking/prompt_templates/"
        )

        full_schema, _ = self.get_full_schema_representation()
        context = {
            "user_question": user_question,
            "hint": evidence,
            "full_schema": full_schema,
        }
        prompt = prompt_renderer.render("TCSL_table_linking", context)
        messages = compose_chat_messages(user_messages=[prompt])
        llm = ModelManager.create_model(
            model_provider=llm_provider,
            model_type=ModelType.COMPLETION,
            model_name=model_name,
        )

        llm_response = llm.get_chat_completion(messages=messages, temperature=0)[
            "completion_content"
        ][0]
        llm_response = self._clean_json_response(llm_response)
        tables = json.loads(llm_response)["tables"]
        filtered_tables = [table for table in self.tables if table.table_name in tables]

        schema_description = self._format_schema_description(
            tables=filtered_tables, include_schema_overview=True
        )

        return filtered_tables, schema_description

    def extract_relevant_columns(
        self,
        filtered_tables_schema: str,
        filtered_tables: List[Table],
        user_question: str,
        evidence: str | None,
        llm_provider: ModelProvider,
        model_name: OpenAIModel | OllamaModel,
    ) -> tuple[list[Table], str, dict[str, list[str]]]:
        """
        Extract relevant columns.
        """
        prompt_renderer = PromptRenderer(
            templates_dir_path="./src/experiments/augmentation/decrease_naturalness/schema_linking/prompt_templates/"
        )
        context = {
            "user_question": user_question,
            "hint": evidence,
            "filtered_tables_schema": filtered_tables_schema,
        }
        prompt = prompt_renderer.render("TCSL_column_linking", context)
        messages = compose_chat_messages(user_messages=[prompt])

        llm = ModelManager.create_model(
            model_provider=llm_provider,
            model_type=ModelType.COMPLETION,
            model_name=model_name,
        )

        llm_response = llm.get_chat_completion(messages=messages, temperature=0)[
            "completion_content"
        ][0]
        llm_response = self._clean_json_response(llm_response)
        data = json.loads(llm_response)

        # Filter columns based on LLM response
        all_filtered_columns = {}
        for table in filtered_tables:
            if table.table_name in data.keys():
                all_filtered_columns[table.table_name] = [
                    col
                    for col in data[table.table_name]
                    if col in [c.column_name for c in table.columns]
                ]
                table.columns = [
                    column
                    for column in table.columns
                    if column.column_name in data[table.table_name]
                ]

        filtered_tables_columns = [table for table in filtered_tables if table.columns]

        schema_description = self._format_schema_description(
            tables=filtered_tables_columns,
            columns_filter=all_filtered_columns,
            include_schema_overview=True,
        )

        return filtered_tables_columns, schema_description, all_filtered_columns

    def get_TCSL_filtered_schema_representation(
        self,
        user_question: str,
        evidence: str,
        model_provider: ModelProvider,
        model_name: OpenAIModel | OllamaModel,
    ) -> tuple[str, dict[str, list[str]]]:
        """
        Returns a filtered schema representation based on the user question and evidence.
        """
        # Prompt inspired from "The Death of Schema Linking?..." paper
        fitered_tables, filtered_tables_schema = self.extract_relevant_tables(
            user_question, evidence, model_provider, model_name
        )
        filtered_tables_columns, filtered_tables_columns_schema, all_columns = (
            self.extract_relevant_columns(
                filtered_tables_schema,
                fitered_tables,
                user_question,
                evidence,
                ModelProvider.OPENAI,
                OpenAIModel.GPT_4O_MINI,
            )
        )
        return filtered_tables_columns_schema, all_columns

    def get_SCSL_filtered_schema_representation(
        self,
        user_question: str,
        evidence: str | None,
        model_provider: ModelProvider,
        model_name: OpenAIModel | OllamaModel,
    ) -> tuple[str, dict[str, list[str]]]:
        """
        Returns a filtered schema representation using SCSL approach.
        Model-determined column-wise relevance where each column is assessed independently.
        """
        prompt_renderer = PromptRenderer(
            templates_dir_path="./src/experiments/augmentation/decrease_naturalness/schema_linking/prompt_templates/"
        )

        llm = ModelManager.create_model(
            model_provider=model_provider,
            model_type=ModelType.COMPLETION,
            model_name=model_name,
        )

        # Collect all relevant columns by evaluating each column independently
        relevant_columns = defaultdict(list)
        relevant_tables = set()

        for table in self.tables:
            for column in table.columns:
                context = {
                    "user_question": user_question,
                    "hint": evidence if evidence else "No additional hint provided.",
                    "candidate_column": f"{table.table_name}.{column.column_name}",
                }

                prompt = prompt_renderer.render("SCSL", context)
                messages = compose_chat_messages(user_messages=[prompt])
                llm_response = llm.get_chat_completion(
                    messages=messages, temperature=0
                )["completion_content"][0]
                llm_response = self._clean_json_response(llm_response)
                try:
                    response_data = json.loads(llm_response)
                    is_relevant = response_data.get("relevant", False)

                    if is_relevant:
                        relevant_columns[table.table_name].append(column.column_name)
                        relevant_tables.add(table.table_name)
                except json.JSONDecodeError:
                    logger.log(
                        "error",
                        "FAILED_TO_PARSE_SCSL_RESPONSE",
                        {
                            "column": f"{table.table_name}.{column.column_name}",
                            "response": llm_response,
                        },
                    )

        # Build the filtered schema representation
        filtered_tables = [t for t in self.tables if t.table_name in relevant_tables]
        schema_description = self._format_schema_description(
            tables=filtered_tables,
            columns_filter=dict(relevant_columns),
            include_schema_overview=True,
        )

        return schema_description, dict(relevant_columns)


if __name__ == "__main__":
    # # dev database
    # config = {
    #     "dbms": "sqlite",
    #     "tables_file_path": "data/benchmarks/Bird/dev_tables.json",
    #     "db_dir_path": "data/benchmarks/Bird/dev_databases/",
    # }

    # new dev database
    config = {
        "dbms": "sqlite",
        "tables_file_path": "data/augmentation/decrease_naturalness/new_dev_databases/new_dev_tables.json",
        "db_dir_path": "data/augmentation/decrease_naturalness/new_dev_databases/",
    }

    db_id = "california_schools"
    tables_file_path = config["tables_file_path"]
    dbms = config["dbms"]
    db_dir_path = config["db_dir_path"]

    question = "What is the surname of the driver with the best lap time in race number 19 in the second qualifying period?"
    evidence = "race number refers to raceId; second qualifying period refers to q2; best lap time refers to MIN(q2);"
    schema_reader = SchemaReader(db_id, tables_file_path, db_dir_path)

    # Full schema
    schema, all_columns = schema_reader.get_full_schema_representation()
    print("--------")
    print("Full Schema")
    print(schema)
    print("Columns by table:")
    print(all_columns)
    print("--------")

    # TCSL approach
    schema, all_columns = schema_reader.get_TCSL_filtered_schema_representation(
        question, evidence, ModelProvider.OPENAI, OpenAIModel.GPT_4O
    )
    print("--------")
    print("TCSL")
    print(schema)
    print("Columns by table:")
    print(all_columns)
    print("--------")

    # SCSL approach
    schema, all_columns = schema_reader.get_SCSL_filtered_schema_representation(
        question, evidence, ModelProvider.OPENAI, OpenAIModel.GPT_4O_MINI
    )
    print("--------")
    print("SCSL")
    print(schema)
    print("Columns by table:")
    print(all_columns)
    print("--------")

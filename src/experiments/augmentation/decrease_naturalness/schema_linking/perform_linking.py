import pandas as pd
import json
from pathlib import Path
from schema_linker import SchemaReader
from src.core.model_manager.model_manager import ModelProvider
from src.core.model_manager import OpenAIModel
from src.core.logger.logger import Logger

logger = Logger(__name__)


def perform_schema_linking(
    input_csv_path: Path, output_csv_path: Path, original_config: dict, new_config: dict
):
    # Read input CSV
    df = pd.read_csv(input_csv_path)
    logger.log(
        "info", "LOADED_INPUT_CSV", {"rows": len(df), "path": str(input_csv_path)}
    )

    # Initialize new columns
    schema_linking_columns = [
        "OO_Full-Schema",
        "OO_TCSL",
        "OO_SCSL",
        "OL_Full-Schema",
        "OL_TCSL",
        "OL_SCSL",
        "LO_Full-Schema",
        "LO_TCSL",
        "LO_SCSL",
        "LL_Full-Schema",
        "LL_TCSL",
        "LL_SCSL",
    ]

    # Check if output file exists and resume from there
    if output_csv_path.exists():
        logger.log(
            "info", "RESUMING_FROM_EXISTING_FILE", {"path": str(output_csv_path)}
        )
        existing_df = pd.read_csv(output_csv_path)

        # Merge existing results with input data
        for col in schema_linking_columns:
            if col in existing_df.columns:
                df[col] = existing_df[col]
            else:
                df[col] = None

        logger.log("info", "LOADED_EXISTING_PROGRESS", {"path": str(output_csv_path)})
    else:
        # Initialize new columns if starting fresh
        for col in schema_linking_columns:
            df[col] = None

    # Process each row
    for idx, row in df.iterrows():
        # Check if all combinations are already processed for this row
        all_processed = all(
            pd.notna(df.at[idx, col]) and df.at[idx, col] not in [None, ""]
            for col in schema_linking_columns
        )

        if all_processed:
            logger.log(
                "info",
                "SKIPPING_PROCESSED_ROW",
                {"row": idx + 1, "total": len(df), "question_id": row["question_id"]},
            )
            continue

        logger.log(
            "info",
            "PROCESSING_ROW",
            {"row": idx + 1, "total": len(df), "question_id": row["question_id"]},
        )

        db_id = row["db_id"]
        original_question = row["original_question"]
        original_evidence = (
            row["original_evidence"] if pd.notna(row["original_evidence"]) else ""
        )
        new_question = row["new_question"]
        new_evidence = row["new_evidence"] if pd.notna(row["new_evidence"]) else ""

        try:
            # Create schema readers for original and new databases
            original_schema_reader = SchemaReader(
                db_id,
                original_config["tables_file_path"],
                original_config["db_dir_path"],
            )

            new_schema_reader = SchemaReader(
                db_id, new_config["tables_file_path"], new_config["db_dir_path"]
            )

            # O/O: Original NL/evidence + Original Schema/SQL (use original_config)
            if (
                pd.isna(df.at[idx, "OO_Full-Schema"])
                or df.at[idx, "OO_Full-Schema"] == ""
            ):
                logger.log("info", "PROCESSING_COMBINATION", {"combination": "OO"})
                _, oo_full = original_schema_reader.get_full_schema_representation()
                df.at[idx, "OO_Full-Schema"] = json.dumps(oo_full)

            if pd.isna(df.at[idx, "OO_TCSL"]) or df.at[idx, "OO_TCSL"] == "":
                _, oo_tcsl = (
                    original_schema_reader.get_TCSL_filtered_schema_representation(
                        original_question,
                        original_evidence,
                        ModelProvider.OPENAI,
                        OpenAIModel.GPT_4O,
                    )
                )
                df.at[idx, "OO_TCSL"] = json.dumps(oo_tcsl)

            if pd.isna(df.at[idx, "OO_SCSL"]) or df.at[idx, "OO_SCSL"] == "":
                _, oo_scsl = (
                    original_schema_reader.get_SCSL_filtered_schema_representation(
                        original_question,
                        original_evidence,
                        ModelProvider.OPENAI,
                        OpenAIModel.GPT_4O_MINI,
                    )
                )
                df.at[idx, "OO_SCSL"] = json.dumps(oo_scsl)

            # O/L: Original NL/evidence + Less-natural Schema/SQL (use new_config)
            if (
                pd.isna(df.at[idx, "OL_Full-Schema"])
                or df.at[idx, "OL_Full-Schema"] == ""
            ):
                logger.log("info", "PROCESSING_COMBINATION", {"combination": "OL"})
                _, ol_full = new_schema_reader.get_full_schema_representation()
                df.at[idx, "OL_Full-Schema"] = json.dumps(ol_full)

            if pd.isna(df.at[idx, "OL_TCSL"]) or df.at[idx, "OL_TCSL"] == "":
                _, ol_tcsl = new_schema_reader.get_TCSL_filtered_schema_representation(
                    original_question,
                    original_evidence,
                    ModelProvider.OPENAI,
                    OpenAIModel.GPT_4O,
                )
                df.at[idx, "OL_TCSL"] = json.dumps(ol_tcsl)

            if pd.isna(df.at[idx, "OL_SCSL"]) or df.at[idx, "OL_SCSL"] == "":
                _, ol_scsl = new_schema_reader.get_SCSL_filtered_schema_representation(
                    original_question,
                    original_evidence,
                    ModelProvider.OPENAI,
                    OpenAIModel.GPT_4O_MINI,
                )
                df.at[idx, "OL_SCSL"] = json.dumps(ol_scsl)

            # L/O: Less-natural NL/evidence + Original Schema/SQL (use original_config)
            if (
                pd.isna(df.at[idx, "LO_Full-Schema"])
                or df.at[idx, "LO_Full-Schema"] == ""
            ):
                logger.log("info", "PROCESSING_COMBINATION", {"combination": "LO"})
                _, lo_full = original_schema_reader.get_full_schema_representation()
                df.at[idx, "LO_Full-Schema"] = json.dumps(lo_full)

            if pd.isna(df.at[idx, "LO_TCSL"]) or df.at[idx, "LO_TCSL"] == "":
                _, lo_tcsl = (
                    original_schema_reader.get_TCSL_filtered_schema_representation(
                        new_question,
                        new_evidence,
                        ModelProvider.OPENAI,
                        OpenAIModel.GPT_4O,
                    )
                )
                df.at[idx, "LO_TCSL"] = json.dumps(lo_tcsl)

            if pd.isna(df.at[idx, "LO_SCSL"]) or df.at[idx, "LO_SCSL"] == "":
                _, lo_scsl = (
                    original_schema_reader.get_SCSL_filtered_schema_representation(
                        new_question,
                        new_evidence,
                        ModelProvider.OPENAI,
                        OpenAIModel.GPT_4O_MINI,
                    )
                )
                df.at[idx, "LO_SCSL"] = json.dumps(lo_scsl)

            # L/L: Less-natural NL/evidence + Less-natural Schema/SQL (use new_config)
            if (
                pd.isna(df.at[idx, "LL_Full-Schema"])
                or df.at[idx, "LL_Full-Schema"] == ""
            ):
                logger.log("info", "PROCESSING_COMBINATION", {"combination": "LL"})
                _, ll_full = new_schema_reader.get_full_schema_representation()
                df.at[idx, "LL_Full-Schema"] = json.dumps(ll_full)

            if pd.isna(df.at[idx, "LL_TCSL"]) or df.at[idx, "LL_TCSL"] == "":
                _, ll_tcsl = new_schema_reader.get_TCSL_filtered_schema_representation(
                    new_question, new_evidence, ModelProvider.OPENAI, OpenAIModel.GPT_4O
                )
                df.at[idx, "LL_TCSL"] = json.dumps(ll_tcsl)

            if pd.isna(df.at[idx, "LL_SCSL"]) or df.at[idx, "LL_SCSL"] == "":
                _, ll_scsl = new_schema_reader.get_SCSL_filtered_schema_representation(
                    new_question,
                    new_evidence,
                    ModelProvider.OPENAI,
                    OpenAIModel.GPT_4O_MINI,
                )
                df.at[idx, "LL_SCSL"] = json.dumps(ll_scsl)

            logger.log("info", "ROW_PROCESSED_SUCCESSFULLY", {"row": idx + 1})

        except Exception as e:
            logger.log(
                "error",
                "ROW_PROCESSING_ERROR",
                {"row": idx + 1, "question_id": row["question_id"], "error": str(e)},
            )
            # Fill with error marker only for columns that are still empty
            for col in schema_linking_columns:
                if pd.isna(df.at[idx, col]) or df.at[idx, col] == "":
                    df.at[idx, col] = "ERROR"

        # Write output after each row to preserve progress
        df.to_csv(output_csv_path, index=False)
        logger.log("debug", "SAVED_PROGRESS", {"path": str(output_csv_path)})

    logger.log(
        "info",
        "PROCESSING_COMPLETED",
        {"total_rows": len(df), "output_path": str(output_csv_path)},
    )


if __name__ == "__main__":
    # Define file paths
    input_csv_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking/new_sql_nl_schema_queries.csv"
    )
    output_csv_path = Path(
        "data/augmentation/decrease_naturalness/schema_linking/new_sql_nl_schema_queries_with_schema_linking.csv"
    )

    # Original and new database configurations
    original_config = {
        "dbms": "sqlite",
        "tables_file_path": "data/benchmarks/Bird/dev_tables.json",
        "db_dir_path": "data/benchmarks/Bird/dev_databases/",
    }

    new_config = {
        "dbms": "sqlite",
        "tables_file_path": "data/augmentation/decrease_naturalness/new_dev_databases/new_dev_tables.json",
        "db_dir_path": "data/augmentation/decrease_naturalness/new_dev_databases/",
    }

    perform_schema_linking(input_csv_path, output_csv_path, original_config, new_config)

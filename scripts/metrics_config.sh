#!/bin/bash

# Evaluation technique
# Options:
#   - execution_accuracy
#   - exact_column_and_exact_cell
#   - exact_column_and_partial_cell
#   - semantic_column_and_exact_cell
#   - semantic_column_and_partial_cell
#   - free_column_and_partial_cell
#   - unified_column_and_semantic_row
export EVAL_TECHNIQUE="exact_column_and_exact_cell"

# Database configuration
# Options: SQLITE, DUCKDB
export DBMS="SQLITE"
export DB_PATH="data/benchmarks/Bird/dev_databases/california_schools/california_schools.sqlite"

# Evaluation settings
# Options: true, false
export PENALIZE_EXTRA_COLUMNS="true"

# Embedding model for semantic evaluations
# Options:
#   - TEXT_EMBEDDING_3_SMALL
#   - TEXT_EMBEDDING_3_LARGE
#   - TEXT_EMBEDDING_ADA_002
export EMBEDDING_MODEL="TEXT_EMBEDDING_3_SMALL"

# Logging configuration
export LOGS_DIR_PATH="data/evaluation_logs/"
# Options: true, false
export ENABLE_LOG="false"

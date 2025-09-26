# Relaxed Evaluation Metrics for Text-to-SQL

This project provides tools for evaluating text-to-SQL systems beyond binary metrics like Exact Match (EM) and Execution Accuracy (EX). We compute:

* **Execution Accuracy (EX)** – do the predicted and ground-truth results match exactly?
* **Execution Precision (EXP)** – of what the system predicted, how much was correct?
* **Execution Recall (EXR)** – of what should have been predicted, how much was recovered?
* **F1 Score** – a balance between precision and recall.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Experiments](#experiments)
5. [Results](#results)

---

## Project Structure

```
.
├── data/                        
├── docs/
│   ├── METRICS.md               # documentation for relaxed evaluation metrics
│   └── ...                      
├── scripts/
│   ├── load_dotenv.sh           # helper to load environment variables
│   └── ...                     
├── src/
│   ├── core/                    # core utilities and shared components
│   ├── analysis/                
│   │   ├── metrics/             
│   │   └── ...
│   ├── experiments/            
│   │   ├── metrics/             
│   │   └── ...
│   ├── metrics/                 # evaluation framework
│   │   ├── evaluation.py        # main entry point for running evaluation
│   │   ├── __init__.py
│   │   └── metrics/             
│   │       ├── __init__.py
│   │       ├── execution_accuracy.py
│   │       ├── exact_column_and_exact_cell.py
│   │       ├── exact_column_and_partial_cell.py
│   │       ├── semantic_column_and_exact_cell.py
│   │       ├── semantic_column_and_partial_cell.py
│   │       ├── free_column_and_partial_cell.py
│   │       └── unified_column_and_semantic_row.py
│   └── ...                      
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Setup

### 1. Environment Variables

Copy `.env.example` → `.env` and edit values and load with:

```bash
source scripts/load_dotenv.sh
```

---

## Usage

### CLI Usage

```bash
# 1. Configure settings (update file and run)
source scripts/metrics_config.sh

# 2. Run evaluation
python src/metrics/evaluation.py \
  --predicted-sql "SELECT ...;" \
  --ground-truth-sql "SELECT ...;"
```

### Python Usage

```python
from src.metrics.evaluation import Evaluation, EvaluationTechnique
from src.core.database.database_handler import DBMS
from src.core.model_manager import OpenAIModel

config = {
    "evaluation_technique": EvaluationTechnique.SEMANTIC_COLUMN_AND_PARTIAL_CELL,
    "db_params": {"dbms": DBMS.SQLITE, "db_path": "path/to/database.sqlite"},
    "penalize_extra_columns": True,
    "embedding_model": OpenAIModel.TEXT_EMBEDDING_3_SMALL,
    "logs_dir_path": "data/evaluation_outputs/",
}

predicted_sql = "SELECT ...;"
ground_truth_sql = "SELECT ...;"

evaluator = Evaluation(config)
results = evaluator.run_evaluation(predicted_sql, ground_truth_sql, log=True)

```

---

## Experiments

We provide three experiments showing how relaxed metrics uncover insights hidden by EX.

### Experiment 1: Table Shape Sensitivity

```bash
source scripts/run_metrics_experiment1.sh
```

---

### Experiment 2: Controlled Error Sensitivity

1- Single Error Mutants 
```bash
bash scripts/source scripts/run_metrics_experiment2_1.sh
```
2- Multi Error Mutants 

```bash
bash scripts/source scripts/run_metrics_experiment2_2.sh
```
---

### Experiment 3: System-Level Comparison on Shared Failures

```bash
source scripts/run_metrics_experiment3.sh
```

---

## Results

> Add tables, figures, or summary observations here.


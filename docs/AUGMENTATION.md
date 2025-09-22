# Text-to-SQL Coverage

This repository contains tools for **expanding and augmenting SQL queries** to enhance Text-to-SQL evaluation. We provide two main augmentation strategies:

* **Join Query Expansion (JQE):** Increases SQL complexity by adding diverse, valid joins.
* **Textual Query Augmentation (TQA):** Modifies natural language queries to test handling of linker tables.

These methods create **targeted challenges** that reveal weaknesses in SQL generation systems. Experiments show that JQE substantially lowers execution accuracy, while TQA exposes subtle natural language vulnerabilities.

---

## Table of Contents

1. [Setup](#setup)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)

---

## Setup

### 1. OpenAI API Key

GPT-4o is used for generating NL queries. Create a `.env` file in the **root** of the repo:

```env
OPENAI_API_KEY='your_api_key_here'
```

### 2. Download Data

Download the dataset from [Google Drive](https://drive.google.com/drive/u/0/folders/1CS5YGn_pjBLiGHsNvUdUXP-1EhUpWdVY).
Rename the folder to `data` and place it in the **root** directory:

```
root/data/
```

### 3. Source Configuration

Before running the scripts, source the configuration file:

```bash
source scripts/augmentation_config.sh
```

---

## Usage

### 1. Join Query Expansion (JQE)

Run the main augmentation script:

```bash
python -m src.augmentation.jq_graph_augmentation.main <db_id> <num_tables>
```

* `db_id`: Choose one of BIRD’s dev databases:

  ```
  california_schools, card_games, codebase_community, debit_card_specializing, 
  european_football_2, financial, formula_1, student_club, superhero, 
  thrombosis_prediction, toxicology
  ```

* `num_tables`: Number of tables to include in the new queries.

  * Must be at least `2`.
  * Cannot exceed the maximum for the database (see `rule_inputs/jq_augmentation/<db_id>:jq_graphs_n_tables.pkl`).

**Example:**

```bash
python -m src.augmentation.jq_graph_augmentation.main european_football_2 4
```

* **Optional:** Use `-gf` to generate queries using graph statistics:

```bash
python -m src.augmentation.jq_graph_augmentation.main european_football_2 4 -gf
```

#### Output

The script produces several JSON files under `rule_outputs/jq_augmentation/`:

1. **Filtered Queries:**

   ```
   filtered/query_first/db_id_(num_tables)t_qf_filtered_aug.json
   filtered/query_first/db_id_(num_tables)t_qf_filtered_ori.json
   ```

   * `*_aug.json`: New augmented queries.
   * `*_ori.json`: Original queries.

2. **Discarded Queries:**

   ```
   discarded/query_first/db_id_(num_tables)_qf_discarded_aug.json
   discarded/query_first/db_id_(num_tables)_qf_discarded_ori.json
   ```

   * Queries that were **skipped** because they were duplicates or too similar to existing queries.

### 2. Textual Query Augmentation (TQA)

> *Section to describe TQA workflow, scripts, and commands*

* **Generating augmented NL queries:**
* **Filtering / validation:**
* **Output format:**

---

## Experiments

> *Section to describe experiments conducted using JQE and TQA*

* **Experiment setup:**
* **Databases used:**
* **Metrics:**

  * Execution accuracy
  * Exact match
  * Other evaluation metrics
* **Scripts to reproduce experiments:**

---

## Results

> *Section to present results of JQE and TQA experiments*

* **Summary tables:**
* **Analysis and observations:**
* **Visualizations (if any):**

---

## Contributing

> *Instructions for contributing to the repository*

* Fork the repo
* Create a feature branch
* Submit a pull request

---

## License

MIT License
Copyright (c) 2025 DSG @ PolyMtl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

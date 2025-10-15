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
python -m src.augmentation.join_query_expansion.main <db_id> <num_tables>
```

* **`db_id`**: Select one of BIRD’s dev databases:

  ```
  california_schools, card_games, codebase_community, debit_card_specializing, 
  european_football_2, financial, formula_1, student_club, superhero, 
  thrombosis_prediction, toxicology
  ```

* **`num_tables`**: Number of tables to include in the generated queries.
  * Must be at least `2`.
  * Cannot exceed the maximum allowed for the database (see `rule_inputs/jq_augmentation/<db_id>:jq_graphs_n_tables.pkl`).

**Example:**

```bash
python -m src.augmentation.join_query_expansion.main european_football_2 4
```

* **Optional:** Add `-gf` to generate queries based on graph statistics:

```bash
python -m src.augmentation.join_query_expansion.main european_football_2 4 -gf
```

#### Output

The script produces several JSON and SQL files under `data/rule_outputs/jq_augmentation/`:

1. **Filtered Queries (Unique Expansions):**

   ```
   filtered/query_first/db_id_(num_tables)t_qf_filtered_aug.json
   filtered/query_first/db_id_(num_tables)t_qf_filtered_ori.json
   # Same naming for .sql files (including the queries for db_id).
   ```

   * `*_aug.json`: Augmented queries.
   * `*_ori.json`: Original queries.

2. **Discarded Queries (Duplicate Isomorphics):**

   ```
   discarded/query_first/db_id_(num_tables)t_qf_discarded_aug.json
   discarded/query_first/db_id_(num_tables)t_qf_discarded_ori.json
   # Same naming for .sql files (including the queries for db_id).
   ```

   * Queries **skipped** because they were duplicates or structurally too similar to existing queries.

---

### 2. Textual Query Augmentation (TQA)

The script splits the data into train, dev, and test sets using K-Means clustering over databases with respect to node distribution.  

Run the script:

```bash
python src/augmentation/main.py
```

#### Output

The script saves JSON and SQL files under `data/lt_elimination/split/`, where `split` is one of `train`, `dev`, or `test`.  

Each split contains files named as follows:

```
split_<technique>_queries.json
split_<technique>_queries.sql
```

where `<technique>` can be:
* `syn_rep` → Synonym Replacement  
* `backtrans` → Backtranslation  
* `context_aug` → Contextual Augmentation  


---

## Experiments

### JQE

#### Experiments 1 & 2: Connectivity and Cyclicity

The full set of expansion queries is stored in:

```
data/rule_outputs/jq_augmentation/aug_log/augmentation_log.pickle
```

To compute the connectivity and cyclicity values for the expansion set (derived from BIRD’s dev set), run the following command:

```bash
python src/experiments/augmentation/join_stats.py
```

This will generate two CSV files under `experiments/augmentation/`:

* `augmented_join_details.csv` — details for the augmented queries, including the average degree for each query and a boolean indicating whether the query contains cycles.
* `original_join_details.csv` — details for the original queries, including the average degree for each query and a boolean indicating whether the query contains cycles.

#### Experiment 3: Systems Performance - Unique Expansions

Three SOTA systems were evaluated on the 58 unique expansion queries derived from the BIRD dev set: CHESS, DIN-SQL, and MAC-SQL. Their outputs for this experiment are stored under:

```
data/experiments/system/
```

where `system` is one of `CHESS`, `DIN-SQL`, or `MAC-SQL`.

To calculate the scores of the systems on the original pre-expansion queries, the unique expansions, and the variation of Exact Match (Delta EX = EX_exp - EX_ori), run:

```bash
python src/experiments/augmentation/delta_ex.py
```

This produces two types of CSV files under `data/experiments/augmentation/`:

1. **`system_mode_results.csv`**  
   * `mode` = `aug` (expanded) or `dev` (original) queries  
   * `system` = CHESS, DIN-SQL, MAC-SQL  
   * Stores the system results on both the original dev set and the unique expansions.

2. **`system_delta_ex_results.csv`**  
   * Contains the Delta EX values for each unique expansion query.

#### Experiment 3: Systems Performance - Sampled Expansions

The same three SOTA systems—`CHESS`, `DIN-SQL`, and `MAC-SQL`—were evaluated on the 408 sampled queries from the full expansion set. Their outputs for this experiment are stored under:

```
data/experiments/system/
````

where `system` is one of `CHESS`, `DIN-SQL`, or `MAC-SQL`.

To generate the results in the file `data/experiments/join_sampling_results.csv`, run:

```bash
python src/experiments/augmentation/join_sampling_results.py
````

---

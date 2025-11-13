# Join Query Expansion (JQE)

Increases SQL complexity by adding diverse, valid joins.

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
source scripts/jqe_config.sh
```

---

## Usage

### 1. Join Query Expansion (JQE)

Run the main augmentation script:

```bash
python -m src/join_query_expansion/main.py <db_id> <num_tables>
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
python -m src/join_query_expansion/main.py european_football_2 4
```

* **Optional:** Add `-gf` to generate queries based on graph statistics:

```bash
python -m src/join_query_expansion/main.py european_football_2 4 -gf
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

## Experiments

### Experiments 1 & 2: Connectivity and Cyclicity

The full set of expansion queries is stored in:

```
data/rule_outputs/jq_augmentation/aug_log/augmentation_log.pickle
```

To compute the connectivity and cyclicity values for the expansion set (derived from BIRD’s dev set), run the following command:

```bash
python experiments/join_query_expansion/join_stats.py
```

This will generate two CSV files under `experiments/augmentation/`:

* `augmented_join_details.csv` — details for the augmented queries, including the average degree for each query and a boolean indicating whether the query contains cycles.
* `original_join_details.csv` — details for the original queries, including the average degree for each query and a boolean indicating whether the query contains cycles.

### Experiment 3: Systems Performance - Unique Expansions

Three SOTA systems were evaluated on the 58 unique expansion queries derived from the BIRD dev set: CHESS, DIN-SQL, and MAC-SQL. Their outputs for this experiment are stored under:

```
data/experiments/system/
```

where `system` is one of `CHESS`, `DIN-SQL`, or `MAC-SQL`.

To calculate the scores of the systems on the original pre-expansion queries, the unique expansions, and the variation of Exact Match (Delta EX = EX_exp - EX_ori), run:

```bash
python experiments/join_query_expansion/delta_ex.py
```

This produces two types of CSV files under `data/experiments/augmentation/`:

1. **`system_mode_results.csv`**  
   * `mode` = `aug` (expanded) or `dev` (original) queries  
   * `system` = CHESS, DIN-SQL, MAC-SQL  
   * Stores the system results on both the original dev set and the unique expansions.

2. **`system_delta_ex_results.csv`**  
   * Contains the Delta EX values for each unique expansion query.

### Experiment 3: Systems Performance - Sampled Expansions

The same three SOTA systems—`CHESS`, `DIN-SQL`, and `MAC-SQL`—were evaluated on the 408 sampled queries from the full expansion set. Their outputs for this experiment are stored under:

```
data/experiments/system/
````

where `system` is one of `CHESS`, `DIN-SQL`, or `MAC-SQL`.

To generate the results in the file `data/experiments/join_sampling_results.csv`, run:

```bash
python experiments/join_query_expansion/join_sampling_results.py
````

---

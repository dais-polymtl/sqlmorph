# Textual Query Augmentation - TQA (Old Approach)

* **Linker Table:** Modifies natural language queries to test handling of linker tables.

---

### Usage
The script splits the data into train, dev, and test sets using K-Means clustering over databases with respect to node distribution.  

Run the script:

```bash
python src/textual_query_augmentation/linker_table/main.py
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

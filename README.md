# SQLMorph: Query Mutation and Fine-Grained Metrics for Text-to-SQL Evaluation

## Abstract

Text-to-SQL systems translate natural language queries into executable SQL, democratizing access to structured data. Despite recent advances driven by large language models (LLMs), evaluation remains a major bottleneck: public benchmarks fail to capture the complexity of enterprise schemas, while building private evaluation sets is costly and nondeterministic, making them difficult to reproduce. To alleviate this issue, we present SQLMorph, a framework for robust Text-to-SQL evaluation via query mutation. SQLMorph introduces two techniques for automatic evaluation set generation and expansion: Join Query Expansion (JQE), which systematically increases structural complexity through semantically valid join additions, and Textual Query Augmentation (TQA), which generates controlled natural language perturbations to assess robustness to linguistic variation. Both techniques create targeted choke points to challenge specific system components. Applied to state-of-the-art models, JQE increases query coverage by up to X% and exposes accuracy degradation in existing systems as the number of joins increases. Meanwhile, TQA reveals that moderate linguistic brittleness can cause degradation (e.g., forcing less natural abbreviations results in up to 17% drop in accuracy).

Beyond evaluation sets, SQLMorph introduces a family of relaxed, execution-level metrics that address the limitations of current binary measures such as Execution Accuracy. We define Execution Precision (EXP) and Execution Recall (EXR) to quantify the fraction of correct and recovered results, respectively, and combine them via F1 for unified scoring. Our experiments show that these metrics enable fine-grained analysis of over- and under-prediction, revealing differences across systems that binary metrics obscure. Together, SQLMorph's mutation-based evaluation and granular metrics support system debugging and align Text-to-SQL evaluation with real-world deployment needs.

## Setup Environment

**1. Clone the Repository**

Clone the project repository to your local machine using:

```bash
git clone https://github.com/dais-polymtl/sqlmorph.git
cd sqlmorph
```

 **2. Installation**

You will first need to install `uv` to manage dependencies. Follow the instructions at the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). Once `uv` is installed, run the following command to install all dependencies, including optional ones for development:

```bash
uv sync --all-extras
```

This command will also automatically generate and manage a virtual environment (`./venv`) in the `sqlmorph` directory. `uv` will also handle fetching the appropriate Python version as needed.

If you're contributing or developing code, install the `pre-commit` hooks for automatic linting (`ruff`) and formatting (`black`). Run the following command:

```bash
uv run pre-commit install
```

## Framework Components

* [Join Query Expansion (JQE)](docs/JQE.md)
* [Textual Query Augmentation (TQA)](docs/TQA.md)
* [Fine-Grained Metrics](docs/METRICS.md)

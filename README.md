# SQLMorph: Query Mutation for Robust Text-to-SQL Evaluation

## Setup Environment

### 1. Clone the Repository

Clone the project repository to your local machine using:

```bash
git clone https://github.com/dsg-polymtl/text-to-sql-coverage.git
cd text-to-sql-coverage
```

### 2. Installation

You will first need to install `uv` to manage dependencies. Follow the instructions at the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). Once `uv` is installed, run the following command to install all dependencies, including optional ones for development:

```bash
uv sync --all-extras
```

This command will also automatically generate and manage a virtual environment (`./venv`) in the `text-to-sql-coverage` directory. `uv` will also handle fetching the appropriate Python version as needed.

If you're contributing or developing code, install the `pre-commit` hooks for automatic linting (`ruff`) and formatting (`black`). Run the following command:

```bash
uv run pre-commit install
```

## Documentation

* [Augmentation](docs/AUGMENTATION.md)
* [Metrics](docs/METRICS.md)

## Citation

If you use this repository, please cite:

```
@misc{sqlmorph2025,
  title     = {SQLMorph: Query Mutation for Robust Text-to-SQL Evaluation},
  author    = {Your Name},
  year      = {2025},
  howpublished = {https://github.com/dais-polymtl/text-to-sql-coverage}
}
```

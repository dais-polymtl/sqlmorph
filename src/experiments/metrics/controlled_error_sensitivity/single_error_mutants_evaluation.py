# single_pattern_mutants_evaluation.py
"""
Single Operator Mutants Evaluation

Process
───────
1. Load mutants from mutants.json
2. Filter to only depth=1 mutants (single operator mutations)
3. For each operator:
   - Apply smart sampling by evaluating with EXECUTION_ACCURACY
   - Keep only mutants with EX==0 until reaching sample_size
4. For each evaluation technique (multiple techniques):
   - Evaluate the filtered mutants for each operator
   - Use func_timeout for timeout enforcement (≤ 120s per query)
   - Apply quality filtering to remove problematic questions
   - Generate technique-specific output file
5. Results are saved as separate JSON files per technique in the scores/{experiment_name} directory
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm

from src.core.database.database_handler import DBMS
from src.core.model_manager import OpenAIModel
from src.metrics import Evaluation, EvaluationTechnique


# 1.  Smart sampling and quality filtering functions
def group_mutants_by_question(mutants: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group mutants by question_id to ensure complete sets are processed together.

    Args:
        mutants: List of mutant dictionaries

    Returns:
        Dictionary mapping question_id to list of mutants for that question
    """
    groups = defaultdict(list)
    for mutant in mutants:
        question_id = mutant["question_id"]
        groups[question_id].append(mutant)

    return dict(groups)


def group_mutants_by_operator(mutants: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group mutants by operator to allow sampling by operator type.

    Args:
        mutants: List of mutant dictionaries

    Returns:
        Dictionary mapping operator to list of mutants with that operator
    """
    groups = defaultdict(list)
    for mutant in mutants:
        # Extract first operator from the operators list
        if "operators" in mutant and mutant["operators"]:
            operator = mutant["operators"][
                0
            ]  # Single operator mutants have one operator
            groups[operator].append(mutant)

    return dict(groups)


def filter_by_specific_operators(
    mutants: List[Dict], specific_operators: List[str] = None
) -> List[Dict]:
    """
    Filter mutants to only include those with specific operators.

    Args:
        mutants: List of mutant dictionaries
        specific_operators: List of specific operators to include

    Returns:
        List of mutants filtered by specific operators
    """
    if not specific_operators:
        return mutants

    filtered_mutants = []
    for m in mutants:
        if "operators" in m and m["operators"]:
            operator = m["operators"][0]  # Single operator mutants
            if operator in specific_operators:
                filtered_mutants.append(m)

    # Count how many mutants we found for each operator
    operator_counts = defaultdict(int)
    for mutant in filtered_mutants:
        if "operators" in mutant and mutant["operators"]:
            operator_counts[mutant["operators"][0]] += 1

    print(f"\nFiltered mutants by {len(specific_operators)} specific operators:")
    for operator in specific_operators:
        count = operator_counts.get(operator, 0)
        print(f"  → {operator}: {count} mutants")

    return filtered_mutants


def evaluate_with_execution_accuracy(
    mutant: dict, dev_db_root: str, per_query_timeout: int, log_enabled: bool
) -> tuple:
    """
    Evaluate a single mutant with EXECUTION_ACCURACY to check if EX==0.

    Args:
        mutant: Mutant dictionary containing SQL and metadata
        dev_db_root: Path to development databases
        per_query_timeout: Timeout in seconds for evaluation
        log_enabled: Whether logging is enabled

    Returns:
        (mutant_with_ex_score, is_suitable) where is_suitable is True if EX==0
    """
    # Create config for EXECUTION_ACCURACY evaluation
    cfg = {
        "evaluation_technique": EvaluationTechnique.EXECUTION_ACCURACY,
        "db_params": {
            "dbms": DBMS.SQLITE,
            "db_path": str(
                Path(dev_db_root) / mutant["db_id"] / f"{mutant['db_id']}.sqlite"
            ),
        },
        "embedding_model": None,  # Not needed for EXECUTION_ACCURACY
        "penalize_extra_pred_cols": True,
        "logs_dir_path": "logs",
    }

    # Use func_timeout for evaluation
    try:
        scored_mutant = score_one_mutant(
            copy.deepcopy(mutant),
            cfg,
            dev_db_root,
            per_query_timeout,
            log_enabled,
        )
        is_suitable = scored_mutant.get("EX") == 0
        return scored_mutant, is_suitable
    except Exception:
        return copy.deepcopy(mutant), False


def smart_sample_mutants(
    mutants: List[Dict],
    sample_size: int = None,
    specific_operators: List[str] = None,
    random_seed: int = 42,
    dev_db_root: str = None,
    per_query_timeout: int = 30,
    log_enabled: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Apply smart sampling that evaluates mutants with EXECUTION_ACCURACY
    and keeps only those with EX==0.

    Sampling strategy:
    1. Filter to only depth=1 mutants (single operator mutations)
    2. If specific_operators is set, filter to only include those specific operators
    3. Group mutants by operator
    4. For each operator group:
       - Shuffle mutants using the random seed
       - Evaluate each mutant with EXECUTION_ACCURACY until finding sample_size with EX==0
       - Keep tracked of the evaluated mutants with EX==0

    Args:
        mutants: List of all mutants
        sample_size: Target number of mutants per operator (None = all)
        specific_operators: List of operators to filter by
        random_seed: Base random seed for reproducible sampling
        dev_db_root: Path to development databases
        per_query_timeout: Timeout in seconds for evaluation
        log_enabled: Whether logging is enabled

    Returns:
        Dictionary mapping operator to list of sampled mutants with EX==0
    """
    print(f"Original: {len(mutants)} mutants")

    # Filter to only depth=1 mutants first
    depth_1_mutants = [m for m in mutants if m.get("depth") == 1]
    print(f"Filtered to depth=1: {len(depth_1_mutants)} mutants")

    if sample_size is None and not specific_operators:
        # Group by operator and return all depth=1 mutants
        return group_mutants_by_operator(depth_1_mutants)

    mutants = depth_1_mutants

    # If specific_operators is defined, filter to those operators
    if specific_operators:
        filtered_mutants = filter_by_specific_operators(mutants, specific_operators)
        print(
            f"Filtered to {len(filtered_mutants)} mutants with {len(specific_operators)} specific operators"
        )
        mutants = filtered_mutants

    # If we don't need to sample by size, just group by operator and return
    if sample_size is None:
        return group_mutants_by_operator(mutants)

    # Group by operator first
    operator_groups = group_mutants_by_operator(mutants)
    sampled_operator_groups = {}

    # For each operator, sample mutants with EX==0
    for operator, operator_mutants in operator_groups.items():
        print(f"  → Sampling from {len(operator_mutants)} '{operator}' mutants")

        # Shuffle mutants using the random seed for this operator
        random.seed(
            random_seed + hash(operator)
        )  # Different seed per operator for diversity
        shuffled_mutants = copy.deepcopy(operator_mutants)
        random.shuffle(shuffled_mutants)

        sampled_mutants = []
        evaluated_count = 0

        # Evaluate mutants until we find enough with EX==0
        for mutant in tqdm(shuffled_mutants, desc=f"  Sampling {operator}", ncols=80):
            evaluated_count += 1
            scored_mutant, is_suitable = evaluate_with_execution_accuracy(
                mutant, dev_db_root, per_query_timeout, log_enabled
            )

            if is_suitable:
                sampled_mutants.append(scored_mutant)
                if len(sampled_mutants) >= sample_size:
                    break

        # Store the sampled mutants for this operator
        if sampled_mutants:
            sampled_operator_groups[operator] = sampled_mutants
            print(
                f"  → Selected {len(sampled_mutants)}/{evaluated_count} mutants with EX==0 for '{operator}'"
            )
        else:
            print(f"  → WARNING: No suitable mutants found for '{operator}'")

    # Print summary of sampled mutants by operator
    total_sampled = sum(len(mutants) for mutants in sampled_operator_groups.values())
    print(
        f"Sampled: {total_sampled} mutants across {len(sampled_operator_groups)} operators (all depth=1)"
    )

    return sampled_operator_groups


def filter_problematic_questions(mutants: List[Dict]) -> List[Dict]:
    """
    Remove question groups where any mutant achieves EX=1.

    If a mutant has EX=1 (perfect execution match), it suggests there's a problem
    with the query (like generating null tables) where adding errors doesn't affect
    the result. We remove the entire question group in such cases.

    Args:
        mutants: List of evaluated mutants (must have EX field)

    Returns:
        Filtered list of mutants with problematic questions removed
    """
    # Group by question_id and check for EX=1
    question_groups = group_mutants_by_question(mutants)
    problematic_questions = set()

    for question_id, group_mutants in question_groups.items():
        for mutant in group_mutants:
            if mutant.get("EX") == 1:
                problematic_questions.add(question_id)
                break

    # Filter out problematic question groups
    filtered_mutants = []
    for mutant in mutants:
        if mutant["question_id"] not in problematic_questions:
            filtered_mutants.append(mutant)

    if problematic_questions:
        removed_count = len(mutants) - len(filtered_mutants)
        print(
            f"  → Quality filter: Removed {len(problematic_questions)} problematic questions "
            f"({removed_count} mutants) with EX=1"
        )

        # Show which operators remain after filtering
        remaining_operators = set()
        for m in filtered_mutants:
            if "operators" in m and m["operators"]:
                remaining_operators.add(m["operators"][0])
        print(f"  → Remaining operators: {sorted(remaining_operators)}")

        print(
            f"  → Remaining: {len(filtered_mutants)} mutants across "
            f"{len(set(m['question_id'] for m in filtered_mutants))} questions"
        )

    return filtered_mutants


# 2.  Direct evaluation with func_timeout
def _evaluate_with_timeout(
    cfg: dict, pred_sql: str, gold_sql: str, log_enabled: bool
) -> tuple:
    """
    Executes Evaluation.run_evaluation with func_timeout.

    Args:
        cfg: Evaluation configuration dictionary
        pred_sql: Mutated SQL to evaluate
        gold_sql: Ground truth SQL for comparison
        log_enabled: Whether logging is enabled

    Returns:
        (metrics_dict, latency) on success
        (None, -1) on error or timeout
    """
    start = time.time()
    try:
        ctx = Evaluation(cfg).run_evaluation(
            predicted_sql=pred_sql,
            ground_truth_sql=gold_sql,
            log=log_enabled,
        )
        latency = ctx.get("latency", time.time() - start)
        return (ctx["metrics"], latency)
    except Exception:
        return (None, -1)


def score_one_mutant(
    mutant: dict,
    template_cfg: dict,
    dev_db_root: str,
    per_query_timeout: int,
    log_enabled: bool,
) -> dict:
    """
    Evaluate a single mutant with timeout protection.

    Process:
    1. Creates evaluation config with correct database path
    2. Uses func_timeout for timeout enforcement
    3. Merges evaluation metrics back into mutant dictionary

    Args:
        mutant: Mutant dictionary containing SQL and metadata
        template_cfg: Base evaluation configuration to copy
        dev_db_root: Path to development databases
        per_query_timeout: Timeout in seconds for evaluation
        log_enabled: Whether logging is enabled

    Returns:
        Updated mutant dict with EX/EXP/EXR/F1/latency fields
    """
    # Create config with correct database path
    cfg = copy.deepcopy(template_cfg)
    db_file = Path(dev_db_root) / mutant["db_id"] / f"{mutant['db_id']}.sqlite"
    cfg["db_params"]["db_path"] = str(db_file)

    # Use func_timeout for evaluation
    try:
        metrics, latency = func_timeout(
            per_query_timeout,
            _evaluate_with_timeout,
            args=(cfg, mutant["mutated_sql"], mutant["gold_sql"], log_enabled),
        )
    except FunctionTimedOut:
        metrics, latency = None, -1
    except Exception:
        metrics, latency = None, -1

    # Create a result dictionary with all original fields preserved
    result = copy.deepcopy(mutant)

    # Merge evaluation results into result dictionary
    if not metrics:
        result.update(
            {"EX": None, "EXP": None, "EXR": None, "F1": None, "latency": latency}
        )
    else:
        result.update(
            {
                "EX": metrics.get("EX", 0),
                "EXP": metrics.get("EXP", 0.0),
                "EXR": metrics.get("EXR", 0.0),
                "F1": metrics.get("F1", 0.0),
                "latency": latency,
            }
        )
    return result


def get_technique_name(technique: EvaluationTechnique) -> str:
    """
    Convert evaluation technique enum to filename-safe string.

    Args:
        technique: EvaluationTechnique enum value

    Returns:
        Lowercase string suitable for filename
    """
    return technique.name.lower()


def evaluate_with_technique(
    mutants: List[Dict],
    technique: EvaluationTechnique,
    embedding_model,
    penalize_extra_pred_cols: bool,
    logs_dir: str,
    dev_db_root: str,
    per_query_timeout: int,
    log_enabled: bool,
) -> List[Dict]:
    """
    Evaluate all mutants using a specific evaluation technique with simple iteration.

    Args:
        mutants: List of mutant dictionaries to evaluate
        technique: Evaluation technique to use
        embedding_model: Embedding model for semantic techniques
        penalize_extra_pred_cols: Whether to penalize extra predicted columns
        logs_dir: Directory for logs
        dev_db_root: Path to development databases
        per_query_timeout: Timeout in seconds for evaluation
        log_enabled: Whether logging is enabled

    Returns:
        List of mutants with evaluation results added (filtered for quality)
    """
    technique_name = get_technique_name(technique)

    # Create evaluation configuration template for this technique
    eval_template = {
        "evaluation_technique": technique,
        "db_params": {"dbms": DBMS.SQLITE, "db_path": ""},  # db_path filled per query
        "embedding_model": embedding_model,
        "penalize_extra_pred_cols": penalize_extra_pred_cols,
        "logs_dir_path": logs_dir,
    }

    print(f"  → Evaluating {len(mutants):,} mutants with {technique.name}")
    if log_enabled:
        print(f"  → Logs: {logs_dir}")

    scored: List[dict] = []

    # Simple iteration through all mutants with progress tracking
    for mutant in tqdm(mutants, desc=f"  {technique_name}", ncols=80):
        result = score_one_mutant(
            copy.deepcopy(mutant),
            eval_template,
            dev_db_root,
            per_query_timeout,
            log_enabled,
        )
        scored.append(result)

    # Apply quality filtering after evaluation
    filtered_scored = filter_problematic_questions(scored)

    return filtered_scored


# 3.  Main evaluation routine
def main():
    """
    Main evaluation pipeline that processes single operator mutants with multiple techniques.

    Process:
    1. Load mutants from single_operator_mutants.json
    2. Apply smart sampling by filtering each operator's mutants with EXECUTION_ACCURACY
       and keeping only those with EX==0
    3. For each evaluation technique:
       - Evaluate the filtered mutants for each operator
       - Use func_timeout for timeout protection
       - Apply quality filtering (remove questions with EX=1)
       - Save technique-specific results
    4. Generate summary statistics
    """
    print("=" * 60)
    print("SINGLE OPERATOR MUTANT EVALUATION - Multiple Techniques")
    print("=" * 60)

    # Load mutants from single_operator_mutants.json
    print(f"Loading mutants from {MUTANTS_JSON}")
    with Path(MUTANTS_JSON).open() as f:
        all_mutants: List[Dict] = json.load(f)

    print(f"Loaded {len(all_mutants):,} total mutants")

    # Apply smart sampling by filtering with EXECUTION_ACCURACY (EX==0)
    sampled_operator_groups = smart_sample_mutants(
        all_mutants,
        SAMPLE_SIZE,
        SPECIFIC_OPERATORS,
        RANDOM_SEED,
        DEV_DB_ROOT,
        PER_QUERY_TIMEOUT,
        LOG,
    )

    # Flatten the operator groups for overall statistics
    sampled_mutants = []
    for operator_mutants in sampled_operator_groups.values():
        sampled_mutants.extend(operator_mutants)

    # Show operator distribution after sampling
    print("\nOperator distribution in sampled mutants with EX==0:")
    for operator, operator_mutants in sorted(sampled_operator_groups.items()):
        print(f"  → {operator}: {len(operator_mutants)} mutants")

    # Display configuration
    print("\nConfiguration:")
    print(f"  Embedding model: {EMBEDDING_MODEL.value}")
    print(f"  Penalize extra pred cols: {PENALIZE_EXTRA_PRED_COLS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Per-query timeout: {PER_QUERY_TIMEOUT}s")
    print(f"  Logging enabled: {LOG}")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Log directory: {LOGS_DIR}")
    print(f"  Sample size per operator: {SAMPLE_SIZE}")
    print(f"  Specific operators: {SPECIFIC_OPERATORS}")

    # Evaluate with each technique and save separate results
    print(f"\nEvaluating with {len(EVALUATION_TECHNIQUES)} different techniques:")

    # For each evaluation technique
    for i, technique in enumerate(EVALUATION_TECHNIQUES, 1):
        technique_name = get_technique_name(technique)
        print(f"\n[{i}/{len(EVALUATION_TECHNIQUES)}] {technique.name}")

        all_scored_mutants = []

        # For each operator, evaluate its filtered mutants with this technique
        for operator, operator_mutants in sampled_operator_groups.items():
            print(f"  → Evaluating {len(operator_mutants)} '{operator}' mutants")

            # Run evaluation for this technique and operator
            scored_mutants = evaluate_with_technique(
                operator_mutants,
                technique,
                EMBEDDING_MODEL,
                PENALIZE_EXTRA_PRED_COLS,
                LOGS_DIR,
                DEV_DB_ROOT,
                PER_QUERY_TIMEOUT,
                LOG,
            )

            # Add to all scored mutants
            all_scored_mutants.extend(scored_mutants)

            # Show operator-specific results
            successful_op_evals = [m for m in scored_mutants if m.get("EX") is not None]
            if successful_op_evals:
                avg_ex = sum(m["EX"] for m in successful_op_evals) / len(
                    successful_op_evals
                )
                avg_f1 = sum(m["F1"] for m in successful_op_evals) / len(
                    successful_op_evals
                )
                print(
                    f"    • Success rate: {len(successful_op_evals)}/{len(scored_mutants)} mutants"
                )
                print(f"    • Avg EX: {avg_ex:.3f}, Avg F1: {avg_f1:.3f}")

        # Show operator distribution after evaluation and filtering
        final_operator_counts = defaultdict(int)
        for mutant in all_scored_mutants:
            if "operators" in mutant and mutant["operators"]:
                final_operator_counts[mutant["operators"][0]] += 1

        print("  → Final operator distribution:")
        for operator, count in sorted(final_operator_counts.items()):
            print(f"    • {operator}: {count} mutants")

        # Save technique-specific results
        out_file = Path(OUT_DIR) / f"mutant_scores_{technique_name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with out_file.open("w") as f:
            json.dump(all_scored_mutants, f, indent=2)

        # Calculate summary statistics
        successful_evals = [m for m in all_scored_mutants if m.get("EX") is not None]
        failed_evals = len(all_scored_mutants) - len(successful_evals)

        if successful_evals:
            avg_ex = sum(m["EX"] for m in successful_evals) / len(successful_evals)
            avg_f1 = sum(m["F1"] for m in successful_evals) / len(successful_evals)
            avg_latency = sum(m["latency"] for m in successful_evals) / len(
                successful_evals
            )

            print(f"  → Results saved to {out_file.name}")
            print(
                f"  → Final dataset: {len(all_scored_mutants)} mutants across "
                f"{len(set(m['question_id'] for m in all_scored_mutants))} questions"
            )
            print(
                f"  → Success rate: {len(successful_evals)}/{len(all_scored_mutants)} "
                f"({100 * len(successful_evals) / len(all_scored_mutants):.1f}%)"
            )
            if failed_evals > 0:
                print(f"  → Failed evaluations: {failed_evals}")
            print(f"  → Average EX: {avg_ex:.3f}")
            print(f"  → Average F1: {avg_f1:.3f}")
            print(f"  → Average latency: {avg_latency:.2f}s")
        else:
            print(f"  → WARNING: All evaluations failed for {technique.name}")
            print(f"  → Results saved to {out_file.name}")

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"Results saved in: {OUT_DIR}")
    if LOG:
        print(f"Logs saved in: {LOGS_DIR}")
    print(f"{'=' * 60}")


# Entry-point
if __name__ == "__main__":
    # Paths and directories
    experiment_name = "2025-10-06_single_error_with_penalty"
    DEV_DB_ROOT = "data/benchmarks/Bird/dev_databases"
    MUTANTS_JSON = (
        "data/metrics/experiments/controlled_error_sensitivity/mutants_depth1.json"
    )
    OUT_DIR = f"data/metrics/experiments/controlled_error_sensitivity/scores/{experiment_name}"

    LOGS_DIR = Path(OUT_DIR) / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Multiple evaluation techniques to compare mutation impact across different metrics
    EVALUATION_TECHNIQUES = [
        EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.EXACT_COLUMN_AND_PARTIAL_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
        EvaluationTechnique.SEMANTIC_COLUMN_AND_PARTIAL_CELL,
        EvaluationTechnique.NO_COLUMN_AND_PARTIAL_CELL,
    ]

    EMBEDDING_MODEL = OpenAIModel.TEXT_EMBEDDING_3_SMALL
    PENALIZE_EXTRA_PRED_COLS = True  # Whether to penalize extra predicted columns
    RANDOM_SEED = 42  # Base random seed for reproducible sampling

    SAMPLE_SIZE = (
        25  # None → score all mutants; small int for quick test (applied per operator)
    )

    # List of specific operators to evaluate - focusing on outer query structure only
    SPECIFIC_OPERATORS = [
        "projection_drop",  # Remove columns from SELECT
        "where_predicate_delete",  # Remove WHERE conditions
        "where_remove",  # Remove entire WHERE clause
        "where_condition_flip",  # Flip WHERE conditions
        "where_strengthen",  # Strengthen WHERE conditions
        "where_weaken",  # Weaken WHERE conditions (outer query)
        "join_break",  # Break JOIN relationships (outer query)
        "aggregation_swap",  # Swap aggregation functions
        "add_star_wildcard",  # Add SELECT * (outer query)
        "having_condition_flip",  # Flip HAVING conditions (outer query)
        "having_remove",  # Remove entire HAVING clause (outer query)
        "join_type_to_left",  # Change JOIN types
        "limit_increase",  # Increase LIMIT clause
        "limit_decrease",  # Decrease LIMIT clause
        "distinct_toggle",  # Toggle DISTINCT in SELECT
    ]

    PER_QUERY_TIMEOUT = 30  # wall-clock seconds per individual query evaluation
    LOG = True  # Enable/disable detailed evaluation logging per technique

    main()

# evaluate_mutants.py
"""
Experiment 1 – Step 2 (parallel evaluation with multiple techniques)

Goal
────
• Loads mutants_error_patterns.json produced in Step 1
• Evaluates each mutant using multiple evaluation techniques in parallel
• Each technique produces its own separate results file with quality filtering
• Extracts metrics EX / EXP / EXR / F1 and latency from Evaluation's return value

Process
───────
1. Load mutants from mutants_error_patterns.json
2. Apply smart sampling by specific error patterns and complete question_id groups
3. For each evaluation technique:
   - Create evaluation run with adaptive worker count (1 for API-heavy techniques, more for others)
   - Use ThreadPoolExecutor with subprocess isolation (≤ 30s per query)
   - Apply quality filtering to remove problematic questions with EX=1
   - Generate technique-specific output file
   - Calculate and display summary statistics including pattern distribution
4. Each mutant is evaluated inside its own short-lived subprocess with SIGALRM timeout
5. Results are saved as separate JSON files per technique in the scores/ex4 directory

Sampling Strategy
─────────────────
• SPECIFIC_PATTERNS: Filter to only include mutants with specified error patterns
• SAMPLE_SIZE: When set, sample complete question_id groups (not individual mutants)
• This ensures all depth levels for each original query are included together
• From each pattern group, sample up to SAMPLE_SIZE question groups for diversity
• Quality filtering removes question_id groups where any mutant achieves EX=1
  (suggests problematic queries where errors don't affect results)

Worker Management
─────────────────
• API-heavy techniques (SEMANTIC_COLUMN_AND_EXACT_CELL, UNIFIED_COLUMN_AND_SEMANTIC_ROW)
  use 1 worker to avoid rate limits and API quota exhaustion
• Non-API techniques use min(4, max(1, cpu_count // 2)) workers for faster processing
• FORCE_SEQUENTIAL mode forces all techniques to use 1 worker

Logging
───────
• All evaluation techniques share a single log directory (logs/ under output directory)
• LOG flag controls whether detailed evaluation logging is enabled
• Logs are saved alongside the mutant scores files for easy organization

Performance
───────────
• Uses ThreadPoolExecutor with adaptive worker count per technique
• Each query evaluation is isolated in subprocess with SIGALRM timeout (30s default)
• Multiprocessing uses 'fork' method for macOS/Jupyter compatibility

Output Files
────────────
• mutant_scores_exact_column_and_exact_cell.json
• mutant_scores_semantic_column_and_exact_cell.json (if enabled)
• mutant_scores_unified_column_and_semantic_row.json (if enabled)
• logs/ (shared directory for all techniques if LOG=True)

Configuration
─────────────
• SAMPLE_SIZE: Integer for testing subset, None to score entire dataset
• SPECIFIC_PATTERNS: List of error patterns to evaluate (filters before sampling)
• LOG: True/False to control detailed evaluation logging
• FORCE_SEQUENTIAL: True to use 1 worker for all techniques
• PER_QUERY_TIMEOUT: Timeout in seconds for individual query evaluation (default 30s)
• EVALUATION_TECHNIQUES: List of EvaluationTechnique enums to compare
• EMBEDDING_MODEL: OpenAI embedding model for semantic evaluation techniques

Quality Filtering
─────────────────
After evaluation, removes entire question groups where any mutant achieves EX=1,
as this suggests the query generates results unaffected by introduced errors
(e.g., null tables or trivial queries).
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm

from src.core.database.database_handler import DBMS
from src.core.model_manager import OpenAIModel
from src.metrics import Evaluation, EvaluationTechnique

# ──────────────────────────────────────────────────────────────────────────
# 0.  Paths, constants, and evaluation techniques
# ──────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")
DEV_DB_ROOT = ROOT / "data/benchmarks/Bird/dev_databases"
MUTANTS_JSON = (
    ROOT
    / "data/evaluation/experiments/controlled_error_sensitivity/mutants_error_patterns.json"
)
OUT_DIR = ROOT / "data/evaluation/experiments/controlled_error_sensitivity/scores/ex4"

LOGS_DIR = OUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Multiple evaluation techniques to compare mutation impact across different metrics
EVALUATION_TECHNIQUES = [
    EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW,
    EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
    EvaluationTechnique.EXACT_COLUMN_AND_EXACT_CELL,
]

# Embedding model configuration for semantic evaluation techniques
EMBEDDING_MODEL = OpenAIModel.TEXT_EMBEDDING_3_SMALL

# Configuration options
SAMPLE_SIZE = 30  # None → score all mutants; small int for quick test (applied to question groups)

# List of specific error patterns to evaluate
SPECIFIC_PATTERNS = [
    ## p-schema
    "projection_drop → add_star_wildcard → distinct_toggle",
    ## p-filter
    "limit_increase → where_predicate_delete → where_remove",
    ### p-join
    "projection_drop → join_type_change → join_break",
    ### p-aggregation
    "aggregation_swap → having_remove → add_star_wildcard",
    ### p-size
    "limit_increase → distinct_toggle → where_predicate_delete",
]

PER_QUERY_TIMEOUT = 120  # wall-clock seconds per individual query evaluation
LOG = True  # Enable/disable detailed evaluation logging per technique
FORCE_SEQUENTIAL = True  # Set to True to use 1 worker for all techniques
# Techniques that use embedding API calls and should use fewer workers
API_HEAVY_TECHNIQUES = {
    EvaluationTechnique.SEMANTIC_COLUMN_AND_EXACT_CELL,
    EvaluationTechnique.UNIFIED_COLUMN_AND_SEMANTIC_ROW,
}


# ──────────────────────────────────────────────────────────────────────────
# 1.  Smart sampling and quality filtering functions
# ──────────────────────────────────────────────────────────────────────────
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


def group_mutants_by_error_pattern(mutants: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group mutants by error_pattern to allow sampling by pattern type.

    Args:
        mutants: List of mutant dictionaries

    Returns:
        Dictionary mapping error_pattern to list of mutants with that pattern
    """
    groups = defaultdict(list)
    for mutant in mutants:
        # Skip mutants without error_pattern
        if "error_pattern" not in mutant:
            continue
        error_pattern = mutant["error_pattern"]
        groups[error_pattern].append(mutant)

    return dict(groups)


def filter_by_specific_patterns(
    mutants: List[Dict], specific_patterns: List[str] = None
) -> List[Dict]:
    """
    Filter mutants to only include those with specific error patterns.

    Args:
        mutants: List of mutant dictionaries
        specific_patterns: List of specific error patterns to include

    Returns:
        List of mutants filtered by specific error patterns
    """
    if not specific_patterns:
        return mutants

    filtered_mutants = [
        m for m in mutants if m.get("error_pattern") in specific_patterns
    ]

    # Count how many mutants we found for each pattern
    pattern_counts = defaultdict(int)
    for mutant in filtered_mutants:
        pattern_counts[mutant.get("error_pattern")] += 1

    print(f"\nFiltered mutants by {len(specific_patterns)} specific patterns:")
    for pattern in specific_patterns:
        count = pattern_counts.get(pattern, 0)
        print(f"  → {pattern}: {count} mutants")

    return filtered_mutants


def smart_sample_mutants(mutants: List[Dict], sample_size: int = None) -> List[Dict]:
    """
    Apply smart sampling that maintains complete question_id groups and filters by specific patterns.

    Sampling strategy:
    1. If SPECIFIC_PATTERNS is set, filter to only include those specific error patterns
    2. From each pattern group, sample up to sample_size question groups
    3. This ensures a diverse set of mutants across different error patterns

    Args:
        mutants: List of all mutants
        sample_size: Target number of mutants per pattern (None = all)

    Returns:
        List of sampled mutants maintaining complete question groups
    """
    if sample_size is None and not SPECIFIC_PATTERNS:
        return mutants

    print(f"Original: {len(mutants)} mutants")

    # If SPECIFIC_PATTERNS is defined, filter to those patterns
    if SPECIFIC_PATTERNS:
        filtered_mutants = filter_by_specific_patterns(mutants, SPECIFIC_PATTERNS)
        print(
            f"Filtered to {len(filtered_mutants)} mutants with {len(SPECIFIC_PATTERNS)} specific patterns"
        )
        mutants = filtered_mutants

    # If we don't need to sample by size, return all filtered mutants
    if sample_size is None:
        return mutants

    # Group by error_pattern first, then by question_id
    pattern_groups = defaultdict(list)
    for mutant in mutants:
        pattern = mutant.get("error_pattern", "unknown")
        pattern_groups[pattern].append(mutant)

    sampled_mutants = []

    # For each pattern group, sample up to sample_size question groups
    for pattern, pattern_mutants in pattern_groups.items():
        # Group by question_id
        question_groups = group_mutants_by_question(pattern_mutants)

        # Calculate how many question groups to sample
        if len(question_groups) <= sample_size:
            # If we have fewer question groups than the sample size, take all of them
            sampled_question_ids = list(question_groups.keys())
        else:
            # Otherwise, randomly sample question groups
            random.seed(42 + hash(pattern))  # Different seed per pattern for diversity
            sampled_question_ids = random.sample(
                list(question_groups.keys()), sample_size
            )

        # Collect all mutants from sampled questions
        for question_id in sampled_question_ids:
            sampled_mutants.extend(question_groups[question_id])

        print(
            f"  → Pattern '{pattern}': {len([m for m in sampled_mutants if m.get('error_pattern') == pattern])} mutants from {len(sampled_question_ids)} questions"
        )

    print(
        f"Sampled: {len(sampled_mutants)} mutants across {len(set(m['question_id'] for m in sampled_mutants))} questions"
    )

    return sampled_mutants


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

        # Show which patterns remain after filtering
        remaining_patterns = set(
            m.get("error_pattern") for m in filtered_mutants if m.get("error_pattern")
        )
        print(f"  → Remaining patterns: {sorted(remaining_patterns)}")

        print(
            f"  → Remaining: {len(filtered_mutants)} mutants across "
            f"{len(set(m['question_id'] for m in filtered_mutants))} questions"
        )

    return filtered_mutants


# ──────────────────────────────────────────────────────────────────────────
# 2.  Worker count management for different technique types
# ──────────────────────────────────────────────────────────────────────────
def get_optimal_worker_count(technique: EvaluationTechnique) -> int:
    """
    Determine optimal worker count based on technique type.

    API-heavy techniques (with embedding calls) should use 1 worker to avoid:
    - Rate limiting from the API provider
    - Connection timeouts from concurrent requests
    - API quota exhaustion

    Non-API techniques can safely use multiple workers.

    Args:
        technique: The evaluation technique to analyze

    Returns:
        Optimal number of workers for this technique
    """
    if FORCE_SEQUENTIAL:
        return 1

    if technique in API_HEAVY_TECHNIQUES:
        return 1  # Sequential processing for API-heavy techniques
    else:
        # Non-API techniques can use multiple workers
        return min(4, max(1, os.cpu_count() // 2))


# ──────────────────────────────────────────────────────────────────────────
# 3.  Direct evaluation with func_timeout
# ──────────────────────────────────────────────────────────────────────────
def _evaluate_with_timeout(cfg: dict, pred_sql: str, gold_sql: str) -> tuple:
    """
    Executes Evaluation.run_evaluation with func_timeout.

    Args:
        cfg: Evaluation configuration dictionary
        pred_sql: Mutated SQL to evaluate
        gold_sql: Ground truth SQL for comparison

    Returns:
        (metrics_dict, latency) on success
        (None, -1) on error or timeout
    """
    start = time.time()
    try:
        ctx = Evaluation(cfg).run_evaluation(
            predicted_sql=pred_sql,
            ground_truth_sql=gold_sql,
            log=LOG,
        )
        latency = ctx.get("latency", time.time() - start)
        return (ctx["metrics"], latency)
    except Exception:
        return (None, -1)


def score_one_mutant(mutant: dict, template_cfg: dict) -> dict:
    """
    Wrapper executed by the thread pool for each mutant evaluation.

    Process:
    1. Creates evaluation config with correct database path
    2. Uses func_timeout for timeout enforcement
    3. Merges evaluation metrics back into mutant dictionary

    Args:
        mutant: Mutant dictionary containing SQL and metadata
        template_cfg: Base evaluation configuration to copy

    Returns:
        Updated mutant dict with EX/EXP/EXR/F1/latency fields
    """
    # Create config with correct database path
    cfg = copy.deepcopy(template_cfg)
    db_file = DEV_DB_ROOT / mutant["db_id"] / f"{mutant['db_id']}.sqlite"
    cfg["db_params"]["db_path"] = str(db_file)

    # Use func_timeout for evaluation
    try:
        metrics, latency = func_timeout(
            PER_QUERY_TIMEOUT,
            _evaluate_with_timeout,
            args=(cfg, mutant["mutated_sql"], mutant["gold_sql"]),
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
    mutants: List[Dict], technique: EvaluationTechnique
) -> List[Dict]:
    """
    Evaluate all mutants using a specific evaluation technique.

    Uses adaptive worker count: 1 for API-heavy techniques, more for others.

    Args:
        mutants: List of mutant dictionaries to evaluate
        technique: Evaluation technique to use

    Returns:
        List of mutants with evaluation results added (filtered for quality)
    """
    technique_name = get_technique_name(technique)

    # Use the single log directory for all techniques
    log_dir_path = str(LOGS_DIR)

    # Determine optimal worker count for this technique
    max_workers = get_optimal_worker_count(technique)
    is_api_heavy = technique in API_HEAVY_TECHNIQUES

    # Create evaluation configuration template for this technique
    eval_template = {
        "evaluation_technique": technique,
        "db_params": {"dbms": DBMS.SQLITE, "db_path": ""},  # db_path filled per query
        "embedding_model": EMBEDDING_MODEL,
        "logs_dir_path": log_dir_path,  # Single log directory for all techniques
    }

    print(f"  → Evaluating {len(mutants):,} mutants with {technique.name}")
    print(
        f"  → Workers: {max_workers} {'(sequential for API safety)' if is_api_heavy else '(parallel)'}"
    )
    if LOG:
        print(f"  → Logs: {log_dir_path}")

    scored: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit all evaluation tasks
        futures = [
            pool.submit(score_one_mutant, copy.deepcopy(m), eval_template)
            for m in mutants
        ]

        # Collect results with progress tracking
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {technique_name}",
            ncols=80,
        ):
            scored.append(future.result())

    # Apply quality filtering after evaluation
    filtered_scored = filter_problematic_questions(scored)

    return filtered_scored


# ──────────────────────────────────────────────────────────────────────────
# 4.  Main evaluation routine
# ──────────────────────────────────────────────────────────────────────────
def main():
    """
    Main evaluation pipeline that processes mutants with multiple techniques.

    Process:
    1. Load mutants from Step 1
    2. Apply smart sampling (by specific patterns and question groups) and quality filtering
    3. For each evaluation technique:
       - Determine optimal worker count (1 for API-heavy, more for others)
       - Create technique-specific log directory
       - Run complete evaluation suite with logging
       - Apply quality filtering (remove questions with EX=1)
       - Save technique-specific results
    4. Generate summary statistics
    """
    print("=" * 60)
    print("MUTANT EVALUATION - Multiple Techniques")
    print("=" * 60)

    # Load mutants from Step 1
    print(f"Loading mutants from {MUTANTS_JSON}")
    with MUTANTS_JSON.open() as f:
        all_mutants: List[Dict] = json.load(f)

    print(f"Loaded {len(all_mutants):,} total mutants")

    # Apply smart sampling by specific patterns and question groups
    sampled_mutants = smart_sample_mutants(all_mutants, SAMPLE_SIZE)

    # Show pattern distribution before evaluation
    pattern_counts = defaultdict(int)
    for mutant in sampled_mutants:
        pattern_counts[mutant.get("error_pattern", "unknown")] += 1

    print("\nPattern distribution in sampled mutants:")
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  → {pattern}: {count} mutants")

    # Display configuration
    print("\nConfiguration:")
    print(f"  Embedding model: {EMBEDDING_MODEL.value}")
    print(f"  Per-query timeout: {PER_QUERY_TIMEOUT}s")
    print(f"  Logging enabled: {LOG}")
    print(f"  Force sequential: {FORCE_SEQUENTIAL}")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Log directory: {LOGS_DIR}")
    print(f"  Sample size per pattern: {SAMPLE_SIZE}")
    print(f"  Specific patterns: {SPECIFIC_PATTERNS}")

    # Show worker strategy
    print("\nWorker Strategy:")
    for technique in EVALUATION_TECHNIQUES:
        workers = get_optimal_worker_count(technique)
        is_api = technique in API_HEAVY_TECHNIQUES
        print(
            f"  {technique.name}: {workers} worker{'s' if workers > 1 else ''} "
            f"{'(API-heavy)' if is_api else '(standard)'}"
        )

    # Evaluate with each technique and save separate results
    print(f"\nEvaluating with {len(EVALUATION_TECHNIQUES)} different techniques:")

    for i, technique in enumerate(EVALUATION_TECHNIQUES, 1):
        technique_name = get_technique_name(technique)
        print(f"\n[{i}/{len(EVALUATION_TECHNIQUES)}] {technique.name}")

        # Run evaluation for this technique (includes quality filtering)
        scored_mutants = evaluate_with_technique(sampled_mutants, technique)

        # Show pattern distribution after evaluation and filtering
        final_pattern_counts = defaultdict(int)
        for mutant in scored_mutants:
            final_pattern_counts[mutant.get("error_pattern", "unknown")] += 1

        print("  → Final pattern distribution:")
        for pattern, count in sorted(final_pattern_counts.items()):
            print(f"    • {pattern}: {count} mutants")

        # Save technique-specific results
        out_file = OUT_DIR / f"mutant_scores_{technique_name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with out_file.open("w") as f:
            json.dump(scored_mutants, f, indent=2)

        # Calculate summary statistics
        successful_evals = [m for m in scored_mutants if m.get("EX") is not None]
        failed_evals = len(scored_mutants) - len(successful_evals)

        if successful_evals:
            avg_ex = sum(m["EX"] for m in successful_evals) / len(successful_evals)
            avg_f1 = sum(m["F1"] for m in successful_evals) / len(successful_evals)
            avg_latency = sum(m["latency"] for m in successful_evals) / len(
                successful_evals
            )

            # Calculate depth distribution
            depth_counts = {}
            for m in successful_evals:
                depth = m.get("depth", "unknown")
                depth_counts[depth] = depth_counts.get(depth, 0) + 1

            print(f"  → Results saved to {out_file.name}")
            print(
                f"  → Final dataset: {len(scored_mutants)} mutants across "
                f"{len(set(m['question_id'] for m in scored_mutants))} questions"
            )
            print(
                f"  → Success rate: {len(successful_evals)}/{len(scored_mutants)} "
                f"({100 * len(successful_evals) / len(scored_mutants):.1f}%)"
            )
            if failed_evals > 0:
                print(f"  → Failed evaluations: {failed_evals}")
            print(f"  → Depth distribution: {dict(sorted(depth_counts.items()))}")
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
        print(f"Logs saved in: {LOGS_DIR}")  # Show the single log dir
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

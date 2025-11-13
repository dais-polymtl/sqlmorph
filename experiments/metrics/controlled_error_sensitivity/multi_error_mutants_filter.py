import json
from pathlib import Path

import pandas as pd
from collections import defaultdict

# ────────────────────────────────────────────────────────────────────────
# Configuration variables
# ────────────────────────────────────────────────────────────────────────
ROOT = Path("data")
MUTANTS_JSON = (
    ROOT / "metrics/experiments/controlled_error_sensitivity/mutants_depth3.json"
)
OUT_FILE = (
    ROOT
    / "metrics/experiments/controlled_error_sensitivity/mutants_error_patterns.json"
)
STATS_FILE = (
    ROOT / "metrics/experiments/controlled_error_sensitivity/filter_mutants_stats.txt"
)


def filter_sequential_mutations(
    input_path: Path, output_path: Path, stats_path: Path
) -> pd.DataFrame:
    """
    Filter mutation data to extract ALL valid cumulative error patterns.

    Args:
        input_path: Path to the input mutants.json file
        output_path: Path to save the filtered output JSON
        stats_path: Path to save the statistics text file

    Returns:
        DataFrame with filtered mutations that form valid sequential patterns
    """
    # Load mutants.json
    with open(input_path, "r") as f:
        mutants = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(mutants)

    # Initialize statistics output
    stats_output = []

    def log_stat(message):
        """Helper function to log to both console and stats file"""
        print(message)
        stats_output.append(message)

    log_stat(f"Loaded {len(df)} total mutants")
    log_stat(f"Unique question IDs: {df['question_id'].nunique()}")

    # Automatically detect the maximum depth in the dataset
    MAX_DEPTH = df["depth"].max()
    log_stat(f"Detected maximum depth: {MAX_DEPTH}")

    # Group by question_id and filter for valid progression patterns
    valid_groups = []
    pattern_groups = defaultdict(list)
    patterns_per_question = defaultdict(int)

    # First group by question_id
    question_groups = df.groupby("question_id")

    for question_id, group in question_groups:
        # Get all depths for this question
        depths = sorted(group["depth"].unique())

        # Create set of required depths (1 through MAX_DEPTH)
        required_depths = set(range(1, MAX_DEPTH + 1))

        # Skip if we don't have all required depths
        if set(depths) != required_depths:
            continue

        # Store all valid patterns found for this question
        # question_patterns_found = 0

        # Start with depth 1 mutants
        depth1_mutants = group[group["depth"] == 1]

        for _, d1_row in depth1_mutants.iterrows():
            # Find all possible valid chains starting with this depth-1 mutation
            find_all_chains(
                d1_row, group, 1, MAX_DEPTH, [], valid_groups, pattern_groups
            )

        # Count patterns found for this question
        patterns_per_question[question_id] = len(
            set(
                row["error_pattern"]
                for row in valid_groups
                if row["question_id"] == question_id
            )
        )

    # Create filtered DataFrame with pattern information
    filtered_df = pd.DataFrame(valid_groups) if valid_groups else pd.DataFrame()

    # Display stats
    # Write all analysis results and stats to both console and file
    log_stat("\nAnalysis Results:")
    if len(filtered_df) > 0:
        unique_questions = filtered_df["question_id"].nunique()
        log_stat(
            f"Total questions with at least one valid progression pattern: {unique_questions}"
        )
        log_stat(f"Total mutants in valid patterns: {len(filtered_df)}")
        log_stat(f"Number of unique error patterns: {len(pattern_groups)}")

        # Calculate average patterns per question
        total_patterns = sum(patterns_per_question.values())
        avg_patterns = total_patterns / unique_questions if unique_questions > 0 else 0
        max_patterns = (
            max(patterns_per_question.values()) if patterns_per_question else 0
        )

        log_stat(f"Average patterns per question: {avg_patterns:.2f}")
        log_stat(f"Maximum patterns for a single question: {max_patterns}")

        # Calculate average number of questions per pattern
        pattern_counts = (
            filtered_df.groupby("error_pattern")["question_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        avg_questions_per_pattern = (
            pattern_counts.mean() if len(pattern_counts) > 0 else 0
        )
        log_stat(
            f"Average number of questions per error pattern: {avg_questions_per_pattern:.2f}"
        )

        # Write all top error patterns to file (not to console)
        stats_output.append("\nTop most common error patterns (all):")
        for pattern, count in pattern_counts.items():
            stats_output.append(f"  {pattern}: {count} questions")

        # Write all top error patterns with completely distinct operators to file (not to console)
        stats_output.append(
            "\nTop error patterns with completely distinct operators (no shared operators):"
        )
        seen_operators = set()
        for pattern, pattern_count in pattern_counts.items():
            operators = set(pattern.split(" → "))
            if any(op in seen_operators for op in operators):
                continue
            stats_output.append(f"  {pattern}: {pattern_count} questions")
            seen_operators.update(operators)

        # Save the filtered DataFrame to JSON
        filtered_json = filtered_df.to_dict(orient="records")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(filtered_json, f, indent=2)
        log_stat(f"\nFiltered mutants saved to {output_path}")
    else:
        log_stat(f"No valid progression patterns through depth {MAX_DEPTH} found.")
        log_stat("Try generating mutants with the sequential pattern approach first.")

    # Write statistics to file
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_output))
    log_stat(f"Statistics saved to {stats_path}")

    return filtered_df


def find_all_chains(
    current_row,
    group_df,
    current_depth,
    max_depth,
    current_chain,
    valid_groups,
    pattern_groups,
):
    """
    Recursively find all valid chains starting from the current row.

    Args:
        current_row: The current mutation row we're building from
        group_df: DataFrame containing all mutations for this question
        current_depth: The current depth level we're processing
        max_depth: Maximum depth to build chains to
        current_chain: List of rows in the current chain so far
        valid_groups: List to collect all valid mutation rows
        pattern_groups: Dict to collect mutations by pattern
    """
    # Add the current row to our chain
    current_chain = current_chain + [current_row]

    # If we've reached max depth, we have a complete chain
    if current_depth == max_depth:
        # Create pattern string from operators
        error_sequence = " → ".join(current_row["operators"])

        # Add all rows in the chain to valid groups with pattern info
        for row in current_chain:
            row_dict = row.to_dict() if isinstance(row, pd.Series) else row
            row_dict["error_pattern"] = error_sequence
            row_dict["pattern_position"] = int(row_dict["depth"])
            valid_groups.append(row_dict)

        # Add to pattern groups
        pattern_groups[error_sequence].extend(
            [
                row.to_dict() if isinstance(row, pd.Series) else row
                for row in current_chain
            ]
        )
        return

    # Current operators sequence
    current_ops = current_row["operators"]

    # Find all candidates at next depth that extend the current operator sequence
    next_depth = current_depth + 1
    candidates = group_df[
        (group_df["depth"] == next_depth)
        & (
            group_df["operators"].apply(
                lambda ops: len(ops) == next_depth
                and all(ops[i] == current_ops[i] for i in range(len(current_ops)))
            )
        )
    ]

    # Recursively explore each candidate to find all possible chains
    for _, candidate_row in candidates.iterrows():
        find_all_chains(
            candidate_row,
            group_df,
            next_depth,
            max_depth,
            current_chain,
            valid_groups,
            pattern_groups,
        )


def main():
    """Main entry point."""
    # Create output directory if it doesn't exist
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Run the filtering process
    filter_sequential_mutations(
        input_path=MUTANTS_JSON, output_path=OUT_FILE, stats_path=STATS_FILE
    )


if __name__ == "__main__":
    main()

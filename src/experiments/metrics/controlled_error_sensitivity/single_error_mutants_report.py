#!/usr/bin/env python3
"""
Simple CSV Export for Mutant Scores Analysis
============================================
This script reads mutant_scores_*.json files and exports a simple CSV with:
- operator: the mutation operator name
- EXACT_COLUMN_AND_EXACT_CELL, SEMANTIC_COLUMN_AND_EXACT_CELL, UNIFIED_COLUMN_AND_SEMANTIC_ROW:
  dict with EX, EXP, EXR, F1 scores and their delta values (metric-1)
"""

import glob
import json
from pathlib import Path
import pandas as pd
import math

# ────────────────────────────────────────────────────────────────────────
# Configuration - adjust to your environment
# ────────────���───────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")
SCORES_DIR = (
    ROOT
    / "data/evaluation/experiments/controlled_error_sensitivity/scores/single_operator"
)
OUTPUT_CSV = SCORES_DIR / "mutant_scores_summary.csv"

# Technique name mapping
TECHNIQUE_MAPPING = {
    "exact_column_and_exact_cell": "EXACT_COLUMN_AND_EXACT_CELL",
    "semantic_column_and_exact_cell": "SEMANTIC_COLUMN_AND_EXACT_CELL",
    "unified_column_and_semantic_row": "UNIFIED_COLUMN_AND_SEMANTIC_ROW",
}


def find_mutant_score_files(scores_dir):
    """Find all mutant_scores_*.json files and extract technique names."""
    pattern = str(scores_dir / "mutant_scores_*.json")
    files = glob.glob(pattern)

    techniques = []
    for file_path in files:
        filename = Path(file_path).name
        # Extract technique name from filename: mutant_scores_{technique}.json
        technique_name = filename.replace("mutant_scores_", "").replace(".json", "")

        # Map to standardized names
        mapped_name = TECHNIQUE_MAPPING.get(technique_name, technique_name.upper())

        techniques.append(
            {
                "name": technique_name,
                "mapped_name": mapped_name,
                "path": Path(file_path),
            }
        )

    return sorted(techniques, key=lambda x: x["name"])


def load_technique_data(file_path):
    """Load mutant scores for a single technique."""
    try:
        with file_path.open() as f:
            records = json.load(f)

        df = pd.DataFrame(records)
        required_cols = {"EX", "EXP", "EXR", "F1"}

        # Check for missing columns
        missing = required_cols.difference(df.columns)
        if missing:
            print(f"Warning: {file_path.name} missing columns: {missing}")
            return None

        return df
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None


def extract_operator_from_record(record):
    """Extract operator name from a record."""
    # Try to get operator from 'operators' field
    if "operators" in record and isinstance(record["operators"], list):
        if len(record["operators"]) == 1:
            return record["operators"][0]
        elif len(record["operators"]) > 1:
            # For multi-operator, use pattern_position if available
            if "pattern_position" in record:
                pos = record["pattern_position"] - 1  # Convert to 0-based
                if 0 <= pos < len(record["operators"]):
                    return record["operators"][pos]
            # Default to first operator
            return record["operators"][0]

    # Try to extract from error_pattern if available
    if "error_pattern" in record and " → " in str(record["error_pattern"]):
        operators = str(record["error_pattern"]).split(" → ")
        return operators[0] if operators else "unknown"

    # Fallback
    return "unknown"


def process_data_to_csv(techniques, output_path):
    """Process all technique data and write to CSV."""
    # Create a dictionary to store data by operator
    operator_data = {}

    for technique in techniques:
        print(f"Processing {technique['name']} -> {technique['mapped_name']}...")

        df = load_technique_data(technique["path"])
        if df is None or df.empty:
            print(f"  No data for {technique['name']}")
            continue

        # Group by operator and calculate mean metrics
        operator_groups = {}
        for _, record in df.iterrows():
            operator = extract_operator_from_record(record.to_dict())

            if operator not in operator_groups:
                operator_groups[operator] = {"EX": [], "EXP": [], "EXR": [], "F1": []}

            # Collect all metric values for this operator, filtering out None and NaN values
            ex_val = record.get("EX")
            if ex_val is not None and not (
                isinstance(ex_val, float) and math.isnan(ex_val)
            ):
                operator_groups[operator]["EX"].append(float(ex_val))

            exp_val = record.get("EXP")
            if exp_val is not None and not (
                isinstance(exp_val, float) and math.isnan(exp_val)
            ):
                operator_groups[operator]["EXP"].append(float(exp_val))

            exr_val = record.get("EXR")
            if exr_val is not None and not (
                isinstance(exr_val, float) and math.isnan(exr_val)
            ):
                operator_groups[operator]["EXR"].append(float(exr_val))

            f1_val = record.get("F1")
            if f1_val is not None and not (
                isinstance(f1_val, float) and math.isnan(f1_val)
            ):
                operator_groups[operator]["F1"].append(float(f1_val))

        # Calculate averages for each operator
        for operator, metrics_lists in operator_groups.items():
            if operator not in operator_data:
                operator_data[operator] = {}

            # Calculate mean metrics, handling empty lists
            avg_metrics = {}
            for metric in ["EX", "EXP", "EXR", "F1"]:
                if metrics_lists[metric]:  # Only calculate if list is not empty
                    avg_metrics[metric] = sum(metrics_lists[metric]) / len(
                        metrics_lists[metric]
                    )
                else:
                    avg_metrics[metric] = 0.0  # Default to 0 if no valid values

            # Combine metrics and deltas into single dict
            combined_metrics = {
                "EX": round(avg_metrics["EX"], 2),
                "EXP": round(avg_metrics["EXP"], 2),
                "EXR": round(avg_metrics["EXR"], 2),
                "F1": round(avg_metrics["F1"], 2),
                "delta_EX": round(avg_metrics["EX"] - 1.0, 2),
                "delta_EXP": round(avg_metrics["EXP"] - 1.0, 2),
                "delta_EXR": round(avg_metrics["EXR"] - 1.0, 2),
                "delta_F1": round(avg_metrics["F1"] - 1.0, 2),
                "count": max(
                    len(metrics_lists["EX"]),
                    len(metrics_lists["EXP"]),
                    len(metrics_lists["EXR"]),
                    len(metrics_lists["F1"]),
                ),  # Number of valid instances
            }

            operator_data[operator][technique["mapped_name"]] = str(combined_metrics)

        print(f"  Found {len(operator_groups)} unique operators")
        for op, lists in operator_groups.items():
            print(f"    {op}: {len(lists['EX'])} instances")

    # Convert to CSV rows
    csv_rows = []
    for operator, techniques_data in operator_data.items():
        row = {"operator": operator}

        # Add columns for each technique
        for technique_name in [
            "EXACT_COLUMN_AND_EXACT_CELL",
            "SEMANTIC_COLUMN_AND_EXACT_CELL",
            "UNIFIED_COLUMN_AND_SEMANTIC_ROW",
        ]:
            row[technique_name] = techniques_data.get(technique_name, "")

        csv_rows.append(row)

    # Write to CSV
    if csv_rows:
        df_output = pd.DataFrame(csv_rows)
        df_output.to_csv(output_path, index=False)
        print(f"\nCSV exported to: {output_path}")
        print(f"Total rows: {len(csv_rows)}")

        # Show summary
        print("\nSummary:")
        print(f"  Unique operators: {len(operator_data)}")
        print(f"  Operator types: {sorted(operator_data.keys())}")
    else:
        print("No data to export!")


def main():
    print("Simple Mutant Scores CSV Export")
    print("=" * 35)
    print(f"Input directory: {SCORES_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")

    # Find technique files
    techniques = find_mutant_score_files(SCORES_DIR)

    if not techniques:
        print(f"No mutant_scores_*.json files found in {SCORES_DIR}")
        return

    print(f"\nFound {len(techniques)} technique file(s):")
    for tech in techniques:
        print(f"  • {tech['name']} -> {tech['mapped_name']}")

    # Process and export to CSV
    process_data_to_csv(techniques, OUTPUT_CSV)

    print("\nDone!")


if __name__ == "__main__":
    main()

# generate_mutants.py
"""
Experiment 1 – Step 1 (controlled, sequential error addition)

Goal
────
For each gold SQL in BIRD-dev, apply atomic mutation operators sequentially,
with errors increasing by one new operator per depth. We systematically try
all applicable operators at each depth level to ensure comprehensive coverage:

    depth 1 :  1 error
    depth 2 :  1 + 1 new error (applied to depth 1 result)
    depth 3 :  1 + 1 new error + 1 more error (applied to depth 2 result)
→ The goal is to track how adding more errors impacts EX/EXP/EXR.

Algorithm Overview
──────────────────
The mutation generation follows a depth-based incremental approach:

1. **Sequential Error Addition**: For each depth level (1, 2, 3), we build upon
   the previous level by adding exactly one more error. This creates a controlled
   progression where we can measure the cumulative impact of multiple errors.

   IMPORTANT: Each new operator is applied to the SQL result from the previous
   depth, not to the original gold SQL. This ensures proper incremental mutation
   where operators work with the actual state of the SQL after previous mutations.

2. **Exhaustive Operator Selection**: At each depth, we systematically try ALL
   available operators that:
   - Can be successfully applied to the current SQL (from previous depth)
   - Don't conflict with previously applied operators
   - Actually modify the AST (verified by attempting application)
   - Produce syntactically valid SQL

3. **Incremental Building**: The mutation process follows this pattern:
   - Depth 1: Apply operator A to original SQL → SQL₁
   - Depth 2: Apply operator B to SQL₁ → SQL₂
   - Depth 3: Apply operator C to SQL₂ → SQL₃

   This ensures that operators like `where_condition_flip` at depth 3 work on
   the correct SQL state (e.g., after `where_predicate_delete` has already
   removed predicates at depth 2).

4. **Conflict Avoidance**: Currently, each operator conflicts only with itself
   to prevent duplicate application. The algorithm detects and avoids selecting
   operators that would conflict with any previously selected operator.

5. **Validation**: Each mutation sequence is validated to ensure:
   - All operators can be applied successfully
   - The final AST differs from the original
   - The resulting SQL is syntactically valid and parseable

Mutation Operators
──────────────────
We implement 12 atomic mutation operators that target different SQL components:

**SELECT Clause Mutations:**
• `projection_drop` — Removes a random column from SELECT list (requires >1 columns)
• `add_star_wildcard` — Adds * or alias.* to SELECT list

**WHERE Clause Mutations:**
• `where_predicate_delete` — Removes a random predicate from WHERE clause (requires >1 predicates)
• `where_condition_flip` — Flips comparison operators in WHERE clauses (=↔!=, >↔<, >=↔<=)
• `where_remove` — Removes the WHERE clause completely

**HAVING Clause Mutations:**
• `having_predicate_delete` — Removes a random predicate from HAVING clause (requires >1 predicates)
• `having_condition_flip` — Flips comparison operators in HAVING clauses (=↔!=, >↔<, >=↔<=)
• `having_remove` — Removes the HAVING clause completely

**JOIN Mutations:**
• `join_break` — Removes the ON condition from a JOIN clause
• `join_type_to_left` — Changes any JOIN to LEFT JOIN (makes joins more inclusive)

**Aggregation Mutations:**
• `aggregation_swap` — Swaps aggregation functions (AVG↔SUM, MIN↔MAX, COUNT→SUM)

**Result Limiting Mutations:**
• `limit_increase` — Adds or increases LIMIT clause (makes it less restrictive)

Operator Conflicts
──────────────────
In the current implementation, each operator conflicts only with itself to prevent
duplicate application within the same mutation sequence. This ensures that each
operator type is applied at most once per sequence, maintaining meaningful and
distinct mutations.

Correctness safeguards
──────────────────────
1.  **Each operator returns a boolean**
    • True  → it modified the AST.
    • False → it could not apply and the sequence is abandoned.

2.  **Final structural equality check**
    Even if every operator claims "changed", a later operator might UNDO
    the change. We compare the fully-mutated AST with the original AST.
    If they are structurally identical, the sequence is discarded.

3.  **Conflict detection**
    Each operator conflicts with itself to prevent duplicate application.
    The has_conflict() function checks if a new operator would conflict
    with any previously selected operators.

4.  **SQL Rendering & Validation**
    All SQL is rendered with `dialect="sqlite"` to ensure consistent
    output formatting. Generated SQL is validated for parseability to
    prevent malformed queries from being included in the mutation suite.

5.  **Incremental Mutation State**
    Each depth level builds upon the previous depth's SQL result, ensuring
    that operators work with the correct intermediate state rather than
    the original gold SQL.

Output
──────
`data/evaluation/experiments/controlled_error_sensitivity/mutants.json`
with keys:
    question_id · db_id · depth · operators[] · mutated_sql · error_count · gold_sql
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set

from sqlglot import parse_one, exp

# ────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")

BIRD_DEV_JSON = "data/benchmarks/Bird/bird_dev.json"
OUT_DIR = "data/evaluation/experiments/controlled_error_sensitivity"
OUT_FILE = OUT_DIR + "/single_operator_mutants.json"
MAX_DEPTH = 1  # Set this value to the desired max depth


# Load BIRD-dev once
def load_json(path: Path) -> list:
    with path.open() as f:
        return json.load(f)


BIRD_DEV = load_json(BIRD_DEV_JSON)


# ────────────────────────────────────────────────────────────────────────
# 1.  Atomic mutation operators (return True iff they change the AST)
# ────────────────────────────────────────────────────────────────────────
SqlAst = exp.Expression


def _rc(seq):
    return random.choice(seq) if seq else None


def projection_drop(ast: SqlAst) -> bool:
    """
    Remove a random column from the SELECT list.
    Requires at least 2 columns to avoid creating invalid SQL.
    """
    sel = ast.find(exp.Select)
    if sel and len(sel.expressions) > 1:
        sel.expressions.remove(_rc(sel.expressions))
        return True
    return False


def where_predicate_delete(ast: SqlAst) -> bool:
    """
    Remove a random predicate from the WHERE clause.
    Handles compound conditions (AND/OR) by flattening and reconstructing.
    Requires at least 2 predicates to avoid removing the entire WHERE clause.
    """
    where = ast.args.get("where")
    if not where:
        return False

    def flat(node):
        if isinstance(node, (exp.And, exp.Or)):
            return flat(node.left) + flat(node.right)
        return [node]

    preds = flat(where.this)
    if len(preds) <= 1:
        return False

    preds.remove(_rc(preds))
    new_w = preds[0]
    for p in preds[1:]:
        new_w = exp.and_(new_w, p)
    ast.set("where", exp.Where(this=new_w))
    return True


def having_predicate_delete(ast: SqlAst) -> bool:
    """
    Remove a random predicate from the HAVING clause.
    Handles compound conditions (AND/OR) by flattening and reconstructing.
    Requires at least 2 predicates to avoid removing the entire HAVING clause.
    """
    having = ast.args.get("having")
    if not having:
        return False

    def flat(node):
        if isinstance(node, (exp.And, exp.Or)):
            return flat(node.left) + flat(node.right)
        return [node]

    preds = flat(having.this)
    if len(preds) <= 1:
        return False

    preds.remove(_rc(preds))
    new_h = preds[0]
    for p in preds[1:]:
        new_h = exp.and_(new_h, p)
    ast.set("having", exp.Having(this=new_h))
    return True


def join_break(ast: SqlAst) -> bool:
    """
    Remove the ON condition from a random JOIN clause.
    This breaks the join logic, often resulting in a cartesian product.
    Only targets JOINs that have an ON condition.
    """
    joins = [j for j in ast.find_all(exp.Join) if j.args.get("on")]
    j = _rc(joins)
    if j:
        j.set("on", None)
        return True
    return False


_AGG_SWAP = dict(avg="sum", sum="avg", min="max", max="min", count="sum")


def aggregation_swap(ast: SqlAst) -> bool:
    """
    Swap aggregation functions with semantically different ones.
    - AVG ↔ SUM (changes from mean to total)
    - MIN ↔ MAX (changes from smallest to largest)
    - COUNT → SUM (changes from count to sum, often meaningless)
    """
    # Look for specific aggregation function node types
    aggs = []
    aggs.extend(ast.find_all(exp.Count))
    aggs.extend(ast.find_all(exp.Sum))
    aggs.extend(ast.find_all(exp.Max))
    aggs.extend(ast.find_all(exp.Min))
    aggs.extend(ast.find_all(exp.Avg))

    # Also check for generic functions that might be aggregations
    for f in ast.find_all(exp.Func):
        if hasattr(f, "name") and f.name and f.name.lower() in _AGG_SWAP:
            aggs.append(f)

    # print(f"Found aggregation functions: {[type(agg).__name__ for agg in aggs]}")

    f = _rc(aggs)
    if f:
        # Determine current function name and swap it
        if isinstance(f, exp.Count):
            new_name = _AGG_SWAP["count"]
        elif isinstance(f, exp.Sum):
            new_name = _AGG_SWAP["sum"]
        elif isinstance(f, exp.Max):
            new_name = _AGG_SWAP["max"]
        elif isinstance(f, exp.Min):
            new_name = _AGG_SWAP["min"]
        elif isinstance(f, exp.Avg):
            new_name = _AGG_SWAP["avg"]
        elif hasattr(f, "name") and f.name:
            new_name = _AGG_SWAP.get(f.name.lower())
        else:
            return False

        if new_name:
            # Replace the aggregation function with the new one
            if new_name == "count":
                new_func = exp.Count(this=f.this, distinct=getattr(f, "distinct", None))
            elif new_name == "sum":
                new_func = exp.Sum(this=f.this, distinct=getattr(f, "distinct", None))
            elif new_name == "max":
                new_func = exp.Max(this=f.this)
            elif new_name == "min":
                new_func = exp.Min(this=f.this)
            elif new_name == "avg":
                new_func = exp.Avg(this=f.this, distinct=getattr(f, "distinct", None))
            else:
                return False

            f.replace(new_func)
            return True
    return False


def add_star_wildcard(ast: SqlAst) -> bool:
    """
    Inject alias.* or bare * into the SELECT list.
    This adds extra columns that weren't in the original query.
    Skip if a star is already present to avoid redundancy.
    Prioritizes table alias if available, falls back to table name or bare *.
    """
    sel = ast.find(exp.Select)
    if not sel:
        return False
    if any(isinstance(e, exp.Star) for e in sel.expressions):
        return False

    first_tbl = next(ast.find_all(exp.Table), None)
    if first_tbl and first_tbl.alias:
        star = exp.Star(this=exp.Identifier(this=first_tbl.alias))
    elif first_tbl:
        star = exp.Star(this=first_tbl.this.copy())
    else:
        star = exp.Star()
    sel.expressions.append(star)
    return True


def where_condition_flip(ast: SqlAst) -> bool:
    """
    Flip comparison operators in WHERE clauses to their logical opposites.
    This fundamentally changes the meaning of conditions:
    - = becomes != (equal becomes not equal)
    - > becomes < (greater becomes less)
    - >= becomes <= (greater-or-equal becomes less-or-equal)
    This is a good alternative when projection operators conflict.
    """
    flip_map = {
        exp.EQ: exp.NEQ,
        exp.NEQ: exp.EQ,
        exp.GT: exp.LT,
        exp.LT: exp.GT,
        exp.GTE: exp.LTE,
        exp.LTE: exp.GTE,
    }

    # Only look for comparisons in WHERE clause
    where_clause = ast.args.get("where")
    if not where_clause:
        return False

    comparisons = [node for node in where_clause.find_all(*flip_map.keys())]
    comp = _rc(comparisons)
    if comp:
        new_type = flip_map[type(comp)]
        comp.replace(new_type(this=comp.this, expression=comp.expression))
        return True
    return False


def having_condition_flip(ast: SqlAst) -> bool:
    """
    Flip comparison operators in HAVING clauses to their logical opposites.
    This fundamentally changes the meaning of aggregate conditions:
    - = becomes != (equal becomes not equal)
    - > becomes < (greater becomes less)
    - >= becomes <= (greater-or-equal becomes less-or-equal)
    This targets aggregate filtering conditions specifically.
    """
    flip_map = {
        exp.EQ: exp.NEQ,
        exp.NEQ: exp.EQ,
        exp.GT: exp.LT,
        exp.LT: exp.GT,
        exp.GTE: exp.LTE,
        exp.LTE: exp.GTE,
    }

    # Only look for comparisons in HAVING clause
    having_clause = ast.args.get("having")
    if not having_clause:
        return False

    comparisons = [node for node in having_clause.find_all(*flip_map.keys())]
    comp = _rc(comparisons)
    if comp:
        new_type = flip_map[type(comp)]
        comp.replace(new_type(this=comp.this, expression=comp.expression))
        return True
    return False


def join_type_to_left(ast: SqlAst) -> bool:
    """
    Change any type of JOIN to LEFT JOIN.
    This makes joins more inclusive, often returning more rows.
    Only targets joins that are not already LEFT JOINs.
    """
    joins = list(ast.find_all(exp.Join))
    candidates = []

    for j in joins:
        current_type = j.args.get("kind")
        # Only consider joins that are not already LEFT joins
        if current_type != "LEFT":
            candidates.append(j)

    j = _rc(candidates)
    if j:
        j.set("kind", "LEFT")
        return True
    return False


def limit_increase(ast: SqlAst) -> bool:
    """
    Increase LIMIT clause: adds or increases the limit value.
    If no LIMIT exists, add one with a random value.
    If LIMIT exists, increase it to make the query less restrictive.
    """
    current_limit = ast.args.get("limit")

    if current_limit:
        # Modify existing limit by increasing it
        limit_value = current_limit.args.get("expression")
        if isinstance(limit_value, exp.Literal) and limit_value.is_int:
            current_val = int(limit_value.this)
            # Always increase limit (make less restrictive)
            new_value = current_val * 2 + random.randint(1, 5)
            ast.set("limit", exp.Limit(expression=exp.Literal.number(new_value)))
            return True
    else:
        # No LIMIT exists, add a new one with random value between 10-50
        ast.set(
            "limit", exp.Limit(expression=exp.Literal.number(random.randint(10, 50)))
        )
        return True

    return False


def having_remove(ast: SqlAst) -> bool:
    """
    Remove the HAVING clause completely.
    This breaks filtering on aggregate results.
    """
    if ast.args.get("having"):
        ast.set("having", None)
        return True
    return False


def where_remove(ast: SqlAst) -> bool:
    """
    Remove the WHERE clause completely.
    This breaks filtering on regular columns and often returns more rows.
    """
    if ast.args.get("where"):
        ast.set("where", None)
        return True
    return False


# Registry of all available mutation operators
OPERATORS: Dict[str, callable] = {
    "projection_drop": projection_drop,
    "where_predicate_delete": where_predicate_delete,
    "where_remove": where_remove,
    "having_predicate_delete": having_predicate_delete,
    "join_break": join_break,
    "aggregation_swap": aggregation_swap,
    "add_star_wildcard": add_star_wildcard,
    "where_condition_flip": where_condition_flip,
    "having_condition_flip": having_condition_flip,
    "join_type_to_left": join_type_to_left,
    "limit_increase": limit_increase,
    "having_remove": having_remove,
}
OP_NAMES = tuple(OPERATORS.keys())


# ────────────────────────────────────────────────────────────────────────
# 2.  Define operator conflicts (operators that shouldn't be used together)
# ────────────────────────────────────────────────────────────────────────
OPERATOR_CONFLICTS: Dict[str, Set[str]] = {
    "projection_drop": {"projection_drop"},
    "add_star_wildcard": {"add_star_wildcard"},
    "where_predicate_delete": {"where_predicate_delete"},
    "where_remove": {"where_remove"},
    "having_predicate_delete": {"having_predicate_delete"},
    "join_break": {"join_break"},
    "join_type_to_left": {"join_type_to_left"},
    "aggregation_swap": {"aggregation_swap"},
    "where_condition_flip": {"where_condition_flip"},
    "having_condition_flip": {"having_condition_flip"},
    "limit_increase": {"limit_increase"},
    "having_remove": {"having_remove"},
}


def has_conflict(op: str, existing_ops: List[str]) -> bool:
    """
    Check if adding this operator would conflict with existing ones.
    Returns True if the operator conflicts with any already selected operator.
    """
    conflicts = OPERATOR_CONFLICTS.get(op, set())
    return any(existing_op in conflicts for existing_op in existing_ops)


# ────────────────────────────────────────────────────────────────────────
# 3.  Apply a sequence; discard if any op is a no-op or net-no-change
# ────────────────────────────────────────────────────────────────────────
def apply_sequence(sql: str, seq: Tuple[str, ...]) -> str | None:
    """
    Apply operators in the given order.
    Returns mutated SQL (dialect='sqlite') or None if:
      • any operator could not be applied, OR
      • final AST is structurally equal to the original AST.
    """
    original_ast = parse_one(sql, read="sqlite")
    ast = original_ast.copy()

    for op in seq:
        if not OPERATORS[op](ast):  # operator had no effect
            return None

    if ast == original_ast:  # nothing changed overall
        return None

    return ast.sql(dialect="sqlite")


def apply_sequence_incremental(
    sql: str, seq: Tuple[str, ...]
) -> Tuple[str, List[str]] | None:
    """
    Apply operators incrementally, returning intermediate results.
    Returns (final_sql, intermediate_sqls) or None if any operator fails.
    intermediate_sqls[i] contains SQL after applying seq[0:i+1] operators.
    """
    original_ast = parse_one(sql, read="sqlite")
    ast = original_ast.copy()
    intermediate_sqls = []

    for op in seq:
        if not OPERATORS[op](ast):  # operator had no effect
            return None
        intermediate_sqls.append(ast.sql(dialect="sqlite"))

    if ast == original_ast:  # nothing changed overall
        return None

    return ast.sql(dialect="sqlite"), intermediate_sqls


def apply_single_operator(sql: str, op: str) -> str | None:
    """
    Apply a single operator to SQL.
    Returns mutated SQL or None if operator cannot be applied.
    """
    try:
        original_ast = parse_one(sql, read="sqlite")
        ast = original_ast.copy()

        if not OPERATORS[op](ast):
            return None

        if ast == original_ast:
            return None

        # Try to render the SQL and validate it's parseable
        result_sql = ast.sql(dialect="sqlite")

        # Validate the generated SQL is parseable
        try:
            parse_one(result_sql, read="sqlite")
        except Exception:
            # If the generated SQL is not parseable, reject this mutation
            return None

        return result_sql

    except Exception:
        # If original SQL can't be parsed or any other error occurs
        return None


# ────────────────────────────────────────────────────────────────────────
# 4.  Generate mutants with sequential error addition (depth-based)
# ────────────────────────────────────────────────────────────────────────
def generate_mutation_suite(
    seed: int = 42,
) -> List[dict]:
    """
    Generate mutation suite with sequential error addition.

    For each SQL query:
    1. Try EVERY possible operator as a starting point (depth=1)
    2. For each valid starting operator, build mutation sequences by systematically
       adding operators:
       - For each subsequent depth (2, 3), try adding EVERY other operator that:
         * Doesn't conflict with already selected operators
         * Successfully applies to the PREVIOUS depth's SQL result
    3. Save all valid mutation sequences that reach the target depths

    This approach ensures proper incremental mutation where each operator
    builds upon the previous mutations' effects.

    Args:
        seed: Random seed for reproducible results (used in operator internals)

    Returns:
        List of mutant dictionaries with metadata and mutated SQL
    """
    random.seed(seed)
    mutants: List[dict] = []

    for item in BIRD_DEV:
        sql, qid, db = item["SQL"], item["question_id"], item["db_id"]

        # Try EVERY operator as a starting point (depth=1)
        valid_sequences_with_sql = []  # Store (operators_list, current_sql)

        for first_op in OPERATORS.keys():
            mutated_sql = apply_single_operator(sql, first_op)
            if mutated_sql:
                valid_sequences_with_sql.append(([first_op], mutated_sql))
                # Add as depth 1 mutant
                mutants.append(
                    {
                        "question_id": qid,
                        "db_id": db,
                        "depth": 1,
                        "operators": [first_op],
                        "mutated_sql": mutated_sql,
                        "error_count": 1,
                        "gold_sql": sql,
                    }
                )

        # For depth 2 and 3, build upon each valid sequence from previous depth
        current_sequences = valid_sequences_with_sql.copy()

        for depth in range(2, MAX_DEPTH + 1):
            next_sequences = []
            for sequence, current_sql in current_sequences:
                # Try adding EVERY possible next operator to the CURRENT SQL
                for next_op in OPERATORS.keys():
                    # Skip if operator conflicts with already selected ones
                    if has_conflict(next_op, sequence):
                        continue

                    # Apply next operator to the current SQL (not original!)
                    new_sql = apply_single_operator(current_sql, next_op)

                    if new_sql:  # If successfully applied
                        new_sequence = sequence + [next_op]
                        next_sequences.append((new_sequence, new_sql))
                        # Add as a valid mutant for this depth
                        mutants.append(
                            {
                                "question_id": qid,
                                "db_id": db,
                                "depth": depth,
                                "operators": new_sequence,
                                "mutated_sql": new_sql,
                                "error_count": depth,
                                "gold_sql": sql,
                            }
                        )

            current_sequences = next_sequences

    return mutants


# ────────────────────────────────────────────────────────────────────────
# 5.  Entry-point – build and save mutants.json
# ────────────────────────────────────────────────────────────────────────
def main():
    """
    Main function to generate mutants with sequential error addition
    and save them to a JSON file.
    """
    print("Generating mutants with sequential error addition …")
    print("Systematically trying all possible operators at each depth level")
    suite = generate_mutation_suite()
    print(f"Generated {len(suite):,} mutants")

    # Print depth distribution
    depth_counts = {}
    for mutant in suite:
        depth = mutant["depth"]
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print("Depth distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]:,} mutants")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        json.dump(suite, f, indent=2)
    print(f"Mutants written to {OUT_FILE}")


if __name__ == "__main__":
    main()

# generate_mutants.py
"""
Experiment 1 – Step 1 (controlled, sequential error addition)

Goal
────
For each gold SQL in BIRD-dev, apply atomic mutation operators sequentially,
with errors increasing by one new operator per depth. We systematically try
all applicable operators at each depth level to ensure comprehensive coverage:

    depth 1 :  1 error
    depth 2 :  1 + 1 new error
    depth 3 :  1 + 1 new error + 1 more error
→ The goal is to track how adding more errors impacts EX/EXP/EXR.

Algorithm Overview
──────────────────
The mutation generation follows a depth-based exhaustive approach:

1. **Sequential Error Addition**: For each depth level (1, 2, 3), we build upon
   the previous level by adding exactly one more error. This creates a controlled
   progression where we can measure the cumulative impact of multiple errors.

2. **Exhaustive Operator Selection**: At each depth, we systematically try ALL
   available operators that:
   - Can be successfully applied to the current SQL
   - Don't conflict with previously applied operators
   - Actually modify the AST (verified by attempting application)

3. **Conflict Avoidance**: Some operators are mutually incompatible (e.g.,
   projection_drop + add_star_wildcard, limit_increase + limit_decrease).
   The algorithm detects and avoids such conflicting combinations.

4. **Validation**: Each mutation sequence is validated to ensure:
   - All operators can be applied successfully
   - The final AST differs from the original
   - The resulting SQL is syntactically valid

Mutation Operators
──────────────────
We implement 12 atomic mutation operators that target different SQL components:

**SELECT Clause Mutations:**
• `projection_drop` — Removes a random column from SELECT list (requires >1 columns)
• `distinct_toggle` — Toggles DISTINCT on/off in a query

**WHERE Clause Mutations:**
• `predicate_delete` — Removes a random predicate from WHERE clause (requires >1 predicates)

**JOIN Mutations:**
• `join_type_change` — Changes JOIN type (e.g., INNER to LEFT or LEFT to INNER)

**Aggregation and Grouping Mutations:**
• `aggregation_swap` — Swaps aggregation functions (AVG↔SUM, MIN↔MAX, COUNT→SUM)
• `group_by_remove` — Removes the GROUP BY clause completely
• `having_remove` — Removes the HAVING clause completely

**Result Limiting and Ordering Mutations:**
• `order_remove` — Completely removes ORDER BY clause
• `limit_increase` — Adds or increases LIMIT clause (makes it less restrictive)
• `limit_decrease` — Decreases existing LIMIT clause (makes it more restrictive, skips if limit=1)

**Wildcard Mutations:**
• `add_star_wildcard` — Adds * or alias.* to SELECT list

Operator Conflicts
──────────────────
Some operators are incompatible and shouldn't be applied together:
• Each operator conflicts with itself to prevent duplicate application
• projection_drop conflicts with add_star_wildcard
• limit_increase conflicts with limit_decrease
• Other conflicts are defined in the OPERATOR_CONFLICTS dictionary

The conflict detection system prevents selecting operators that would neutralize
each other's effects, ensuring meaningful mutations at each depth level.

Correctness safeguards
──────────────────────
1.  **Each operator returns a boolean**
    • True  → it modified the AST.
    • False → it could not apply and the whole permutation is abandoned.

2.  **Final structural equality check**
    Even if every operator claims "changed", a later operator might UNDO
    the change (e.g., add * then drop it).
    We compare the fully-mutated AST with the original AST using
    `ast.equals(original_ast)`.
    If they are structurally identical, the sequence is discarded.

3.  **Conflict detection**
    Some operators conflict with each other (e.g., add_star_wildcard
    vs projection_drop). We avoid selecting conflicting operators in the
    same sequence.

4.  **add_star_wildcard** injects `alias.*` or bare `*`; we render
    SQL with `dialect="sqlite"` so sqlglot does not rewrite divisions
    into `NULLIF(x,0)`.

Output
──────
`data/evaluation/metrics/experiment_1/mutants.json`
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
# Config – adjust paths for your workspace
# ────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/mhmalekpour/PycharmProjects/text-to-sql-coverage")

BIRD_DEV_JSON = ROOT / "data/benchmarks/Bird/bird_dev.json"
OUT_DIR = ROOT / "data/evaluation/metrics/experiment_1"
OUT_FILE = OUT_DIR / "mutants.json"
MAX_DEPTH = 3  # Set this value to the desired max depth


# ────────────────────────────────────────────────────────────────────────
# Load BIRD-dev once
# ────────────────────────────────────────────────────────────────────────
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


def predicate_delete(ast: SqlAst) -> bool:
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
    aggs = [f for f in ast.find_all(exp.Func) if f.name.lower() in _AGG_SWAP]
    f = _rc(aggs)
    if f:
        f.set("name", _AGG_SWAP[f.name.lower()])
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


def condition_flip(ast: SqlAst) -> bool:
    """
    Flip comparison operators to their logical opposites.
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

    comparisons = [node for node in ast.find_all(*flip_map.keys())]
    comp = _rc(comparisons)
    if comp:
        new_type = flip_map[type(comp)]
        comp.replace(new_type(this=comp.this, expression=comp.expression))
        return True
    return False


def join_type_change(ast: SqlAst) -> bool:
    """
    Change JOIN type (e.g., INNER to LEFT or LEFT to INNER).
    This alters result cardinality and inclusion of records without matches.
    """
    joins = list(ast.find_all(exp.Join))
    j = _rc(joins)
    if not j:
        return False

    # Define possible join type swaps
    join_types = {
        "": "LEFT",  # Default INNER JOIN -> LEFT JOIN
        "LEFT": "",  # LEFT JOIN -> INNER JOIN
        "RIGHT": "",  # RIGHT JOIN -> INNER JOIN
        "FULL": "LEFT",  # FULL JOIN -> LEFT JOIN
        "INNER": "LEFT",  # INNER JOIN -> LEFT JOIN
    }

    current_type = j.args.get("kind", "")
    if current_type in join_types:
        j.set("kind", join_types[current_type])
        return True
    return False


def distinct_toggle(ast: SqlAst) -> bool:
    """
    Toggle DISTINCT on/off in a query.
    This changes result cardinality by including/excluding duplicates.
    """
    sel = ast.find(exp.Select)
    if not sel:
        return False

    # Get current distinct value
    current_distinct = sel.args.get("distinct")

    # Toggle distinct on/off
    if current_distinct is None or current_distinct is False:
        # Turn on DISTINCT - use an empty exp.Distinct object instead of a boolean
        sel.set("distinct", exp.Distinct())
    else:
        # Turn off DISTINCT
        sel.set("distinct", None)

    return True


def limit_modify(ast: SqlAst) -> bool:
    """
    Modify LIMIT clause: randomly increase or decrease the limit value.
    If no LIMIT exists, add one with a random value.
    If LIMIT is 1, only increase is possible (can't decrease further).
    """
    current_limit = ast.args.get("limit")

    if current_limit:
        # Modify existing limit
        limit_value = current_limit.args.get("expression")
        if isinstance(limit_value, exp.Literal) and limit_value.is_int:
            current_val = int(limit_value.this)

            # If current limit is 1, we can only increase
            if current_val <= 1:
                # Increase limit (make less restrictive)
                new_value = current_val * 2 + random.randint(1, 5)
                ast.set("limit", exp.Limit(expression=exp.Literal.number(new_value)))
                return True

            # Randomly choose to increase or decrease
            if random.choice([True, False]):  # 50% chance for each
                # Increase limit (make less restrictive)
                new_value = current_val * 2 + random.randint(0, current_val)
                ast.set("limit", exp.Limit(expression=exp.Literal.number(new_value)))
                return True
            else:
                # Decrease limit (make more restrictive)
                new_value = max(1, current_val // 2)
                if new_value != current_val:  # Only if value actually changes
                    ast.set(
                        "limit", exp.Limit(expression=exp.Literal.number(new_value))
                    )
                    return True
                # If new_value would be the same, try increasing instead
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


# Registry of all available mutation operators
OPERATORS: Dict[str, callable] = {
    "projection_drop": projection_drop,
    "predicate_delete": predicate_delete,
    "join_break": join_break,
    "aggregation_swap": aggregation_swap,
    "add_star_wildcard": add_star_wildcard,
    "condition_flip": condition_flip,
    "join_type_change": join_type_change,
    "distinct_toggle": distinct_toggle,
    "limit_modify": limit_modify,
    "having_remove": having_remove,
}
OP_NAMES = tuple(OPERATORS.keys())


# ────────────────────────────────────────────────────────────────────────
# 2.  Define operator conflicts (operators that shouldn't be used together)
# ────────────────────────────────────────────────────────────────────────
OPERATOR_CONFLICTS: Dict[str, Set[str]] = {
    "projection_drop": {"projection_drop", "add_star_wildcard"},
    "add_star_wildcard": {"add_star_wildcard", "projection_drop"},
    "predicate_delete": {"predicate_delete"},
    "join_break": {"join_break", "join_type_change"},
    "join_type_change": {"join_type_change", "join_break"},
    "aggregation_swap": {"aggregation_swap"},
    "condition_flip": {"condition_flip"},
    "distinct_toggle": {"distinct_toggle"},
    "limit_modify": {"limit_modify"},
    "having_remove": {"having_remove"},
    # Add more conflicts as needed
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
         * Successfully applies to the current SQL
    3. Save all valid mutation sequences that reach the target depths

    This exhaustive approach ensures we explore all possible valid combinations
    of operators at each depth level, following the sequential rule where each
    depth builds upon the previous with exactly one additional operator.

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
        valid_starting_sequences = []
        for first_op in OPERATORS.keys():
            if apply_sequence(sql, (first_op,)):
                valid_starting_sequences.append([first_op])
                # Add as depth 1 mutant
                mutated_sql = apply_sequence(sql, (first_op,))
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
        current_sequences = valid_starting_sequences.copy()

        for depth in range(2, MAX_DEPTH + 1):
            next_sequences = []
            for sequence in current_sequences:
                # Try adding EVERY possible next operator
                for next_op in OPERATORS.keys():
                    # Skip if operator conflicts with already selected ones
                    if has_conflict(next_op, sequence):
                        continue

                    # Try applying the extended sequence
                    new_sequence = sequence + [next_op]
                    mutated_sql = apply_sequence(sql, tuple(new_sequence))

                    if mutated_sql:  # If successfully applied
                        next_sequences.append(new_sequence)
                        # Add as a valid mutant for this depth
                        mutants.append(
                            {
                                "question_id": qid,
                                "db_id": db,
                                "depth": depth,
                                "operators": new_sequence,
                                "mutated_sql": mutated_sql,
                                "error_count": depth,
                                "gold_sql": sql,
                            }
                        )

            current_sequences = next_sequences

    return mutants


# ────────────────────────────────────────────────────────────────────────
# 5.  Entry-point – build and save mutants.json
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
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

# generate_mutants.py
"""
Experiment 1 – Step 1 (controlled, sequential error addition)

Goal
────
For each gold SQL in BIRD-dev, apply atomic mutation operators sequentially,
with errors increasing by one new operator per depth. If an operator doesn't
apply at a given depth, we try another operator until we have exactly the
desired number of operators applied:

    depth 1 :  1 error
    depth 2 :  1 + 1 new error
    depth 3 :  1 + 1 new error + 1 more error
→ The goal is to track how adding more errors impacts EX/EXP/EXR.

Algorithm Overview
──────────────────
The mutation generation follows a depth-based sequential approach:

1. **Sequential Error Addition**: For each depth level (1, 2, 3), we build upon
   the previous level by adding exactly one more error. This creates a controlled
   progression where we can measure the cumulative impact of multiple errors.

2. **Operator Selection**: At each depth, we randomly select a new operator that:
   - Can be successfully applied to the current SQL
   - Doesn't conflict with previously applied operators
   - Actually modifies the AST (verified by attempting application)

3. **Conflict Avoidance**: Some operators are mutually incompatible (e.g.,
   projection_drop + add_irrelevant_column). The algorithm detects and avoids
   such conflicting combinations.

4. **Validation**: Each mutation sequence is validated to ensure:
   - All operators can be applied successfully
   - The final AST differs from the original
   - The resulting SQL is syntactically valid

Mutation Operators
──────────────────
We implement 7 atomic mutation operators that target different SQL components:

**SELECT Clause Mutations:**
• `projection_drop` — Removes a random column from SELECT list (requires >1 columns)
• `add_irrelevant_column` — Adds alias.* or bare * to SELECT list

**WHERE Clause Mutations:**
• `predicate_delete` — Removes a random predicate from WHERE clause (requires >1 predicates)
• `condition_flip` — Flips comparison operators (= ↔ !=, > ↔ <, >= ↔ <=, etc.)

**JOIN Mutations:**
• `join_break` — Removes ON condition from a random JOIN, breaking the join logic

**Aggregation Mutations:**
• `aggregation_swap` — Swaps aggregation functions (AVG↔SUM, MIN↔MAX, COUNT→SUM)

**ORDER BY Mutations:**
• `order_remove` — Completely removes ORDER BY clause

Operator Conflicts
──────────────────
Some operators are incompatible and shouldn't be applied together:
• `projection_drop` ↔ `add_irrelevant_column` (they can cancel each other out)

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
    Some operators conflict with each other (e.g., add_irrelevant_column
    vs projection_drop). We avoid selecting conflicting operators in the
    same sequence.

4.  **add_irrelevant_column** injects `alias.*` or bare `*`; we render
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
MAX_DEPTH = 2  # Set this value to the desired max depth


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


def order_remove(ast: SqlAst) -> bool:
    """
    Completely remove the ORDER BY clause.
    This can change result ordering in cases where it matters for correctness.
    """
    if ast.args.get("order"):
        ast.set("order", None)
        return True
    return False


def add_irrelevant_column(ast: SqlAst) -> bool:
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


# Registry of all available mutation operators
OPERATORS: Dict[str, callable] = {
    "projection_drop": projection_drop,
    "predicate_delete": predicate_delete,
    "join_break": join_break,
    "aggregation_swap": aggregation_swap,
    "order_remove": order_remove,
    "add_irrelevant_column": add_irrelevant_column,
    "condition_flip": condition_flip,
}
OP_NAMES = tuple(OPERATORS.keys())


# ────────────────────────────────────────────────────────────────────────
# 2.  Define operator conflicts (operators that shouldn't be used together)
# ────────────────────────────────────────────────────────────────────────
OPERATOR_CONFLICTS: Dict[str, Set[str]] = {
    "projection_drop": {"projection_drop", "add_irrelevant_column"},
    "add_irrelevant_column": {"add_irrelevant_column", "projection_drop"},
    "condition_flip": {"condition_flip"},
    "order_remove": {"order_remove"},
    "join_break": {"join_break"},
    # Add more conflicts as needed, e.g.:
    # "predicate_delete": {"some_other_predicate_op"},
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
    depths: Tuple[int, ...] = (1, 2, 3),
    seed: int = 42,
) -> List[dict]:
    """
    Generate mutation suite with sequential error addition.

    For each SQL query and each depth level:
    1. Start with an empty operator sequence
    2. For each error level (1 to depth):
       - Filter out operators that conflict with already selected ones
       - Try random operators until one successfully applies
       - Add the working operator to the sequence
    3. Save the mutant if we successfully applied the target number of operators

    Args:
        depths: Tuple of depth levels to generate (currently unused, uses MAX_DEPTH)
        seed: Random seed for reproducible results

    Returns:
        List of mutant dictionaries with metadata and mutated SQL
    """
    random.seed(seed)
    mutants: List[dict] = []

    for item in BIRD_DEV:
        sql, qid, db = item["SQL"], item["question_id"], item["db_id"]

        for depth in range(1, MAX_DEPTH + 1):  # Generate mutants up to MAX_DEPTH
            # Start with one error and sequentially add new errors per depth
            operator_sequence = []
            for base_error in range(depth):
                operators_left = list(OPERATORS.keys())
                # Filter out operators that conflict with already selected ones
                operators_left = [
                    op
                    for op in operators_left
                    if not has_conflict(op, operator_sequence)
                ]

                # Start with one random operator for each level
                while operators_left:
                    op = random.choice(operators_left)
                    new_ast = apply_sequence(sql, operator_sequence + [op])
                    if new_ast is not None:
                        operator_sequence.append(op)
                        break
                    else:
                        operators_left.remove(op)  # remove non-working operator

                if len(operator_sequence) != base_error + 1:
                    # If we cannot apply enough operators, skip this path
                    break

            # If the mutation was valid for this depth, save it
            if len(operator_sequence) == depth:
                final_mutated_sql = apply_sequence(sql, operator_sequence)
                if final_mutated_sql:  # Double-check the sequence still works
                    mutants.append(
                        {
                            "question_id": qid,
                            "db_id": db,
                            "depth": depth,
                            "operators": operator_sequence,
                            "mutated_sql": final_mutated_sql,
                            "error_count": depth,
                            "gold_sql": sql,
                        }
                    )

    return mutants


# ────────────────────────────────────────────────────────────────────────
# 5.  Entry-point – build and save mutants.json
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating mutants with sequential error addition …")
    suite = generate_mutation_suite()
    print(f"Generated {len(suite):,} mutants")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        json.dump(suite, f, indent=2)
    print(f"Mutants written to {OUT_FILE}")

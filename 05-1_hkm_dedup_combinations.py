from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union


"""
Paper-style HKM template generator with semantic deduplication.

This script follows the HKM paper constraints as closely as possible for the
pure-Boolean setting, with one intentional change requested by the user:
thresholded Boolean SUM is removed from the operator set.

Included constraints:
1. Exact variable counts in {2, 3, 4}
2. Each variable is used at most once
3. At most 4 Boolean operators total
4. Allowed operators: AND, OR, NOT

Output formulas are deduplicated:
- semantically, by truth table
- structurally across variable renaming, so x1 AND x2 and x2 AND x1 collapse

The generated formulas are templates. You can later map x1, x2, ... onto your
filtered atomic subsequence indicators.
"""


# HKM variable counts to enumerate. The paper allows up to 4 variables; here we
# keep the exact lengths requested for downstream template matching.
LENGTHS = (2, 3, 4)

# Paper-style operator set, with SUM intentionally removed.
ALLOW_NOT = True
BINARY_OPERATORS = ("AND", "OR")
USE_THRESHOLD_SUM = False

# The paper constrains HKMs to at most four Boolean operators total.
# We count NOT together with AND/OR under this cap.
MAX_TOTAL_OPERATORS = 4

OUTPUT_DIR = Path(__file__).resolve().parent / "hkm_dedup_output"


Literal = Tuple[str, int, bool]  # ("lit", variable_index, is_negated)
Expr = Union[Literal, Tuple[str, "Expr", "Expr"]]  # ("AND"/"OR", left, right)


@dataclass(frozen=True)
class FormulaRecord:
    length: int
    formula: str
    truth_table: str
    canonical_signature: str
    negation_count: int
    binary_operator_count: int
    total_operator_count: int


def literal(var_index: int, is_negated: bool) -> Literal:
    return ("lit", var_index, is_negated)


def format_expr(expr: Expr) -> str:
    if expr[0] == "lit":
        _, var_index, is_negated = expr
        base = f"x{var_index + 1}"
        return f"NOT {base}" if is_negated else base
    op, left, right = expr
    return f"({format_expr(left)} {op} {format_expr(right)})"


def eval_expr(expr: Expr, assignment: Sequence[int]) -> int:
    if expr[0] == "lit":
        _, var_index, is_negated = expr
        value = int(assignment[var_index])
        return 1 - value if is_negated else value

    op, left, right = expr
    left_value = eval_expr(left, assignment)
    right_value = eval_expr(right, assignment)
    if op == "AND":
        return left_value & right_value
    if op == "OR":
        return left_value | right_value
    raise ValueError(f"Unknown operator: {op}")


def all_assignments(length: int) -> List[Tuple[int, ...]]:
    return [tuple(bits) for bits in product((0, 1), repeat=length)]


def truth_table_bits(expr: Expr, length: int) -> str:
    return "".join(str(eval_expr(expr, assignment)) for assignment in all_assignments(length))


def canonical_signature(bits: str, length: int) -> str:
    assignments = all_assignments(length)
    table = {assignment: bit for assignment, bit in zip(assignments, bits)}
    signatures = []
    for perm in permutations(range(length)):
        permuted_bits = "".join(
            table[tuple(assignment[index] for index in perm)]
            for assignment in assignments
        )
        signatures.append(permuted_bits)
    return min(signatures)


def build_all_binary_exprs(literals_in_order: Sequence[Literal]) -> Iterable[Expr]:
    if len(literals_in_order) == 1:
        yield literals_in_order[0]
        return

    for split in range(1, len(literals_in_order)):
        left_items = literals_in_order[:split]
        right_items = literals_in_order[split:]
        for left_expr in build_all_binary_exprs(left_items):
            for right_expr in build_all_binary_exprs(right_items):
                for op in BINARY_OPERATORS:
                    yield (op, left_expr, right_expr)


def negation_patterns(length: int, max_negations: int | None) -> Iterable[Tuple[int, ...]]:
    all_indices = range(length)
    if not ALLOW_NOT:
        yield ()
        return

    upper = length if max_negations is None else min(length, max_negations)
    for count in range(0, upper + 1):
        for combo in combinations(all_indices, count):
            yield combo


def enumerate_formulas_for_length(length: int) -> List[FormulaRecord]:
    binary_operator_count = length - 1
    if MAX_TOTAL_OPERATORS is None:
        max_negations = None
    else:
        max_negations = MAX_TOTAL_OPERATORS - binary_operator_count
        if max_negations < 0:
            return []

    best_by_signature: Dict[str, FormulaRecord] = {}

    for variable_order in permutations(range(length)):
        for negated_indices in negation_patterns(length, max_negations):
            negated_index_set = set(negated_indices)
            literals_in_order = [
                literal(var_index, position in negated_index_set)
                for position, var_index in enumerate(variable_order)
            ]
            negation_count = len(negated_index_set)
            total_operator_count = binary_operator_count + negation_count

            for expr in build_all_binary_exprs(literals_in_order):
                formula = format_expr(expr)
                bits = truth_table_bits(expr, length)
                signature = canonical_signature(bits, length)
                record = FormulaRecord(
                    length=length,
                    formula=formula,
                    truth_table=bits,
                    canonical_signature=signature,
                    negation_count=negation_count,
                    binary_operator_count=binary_operator_count,
                    total_operator_count=total_operator_count,
                )
                current = best_by_signature.get(signature)
                if current is None:
                    best_by_signature[signature] = record
                    continue
                if (len(record.formula), record.formula) < (len(current.formula), current.formula):
                    best_by_signature[signature] = record

    return sorted(
        best_by_signature.values(),
        key=lambda item: (
            item.total_operator_count,
            item.negation_count,
            len(item.formula),
            item.formula,
        ),
    )


def write_outputs(records_by_length: Dict[int, List[FormulaRecord]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "summary.json"
    summary = {
        str(length): {
            "count": len(records),
            "paper_style_constraints": True,
            "allow_not": ALLOW_NOT,
            "binary_operators": list(BINARY_OPERATORS),
            "use_threshold_sum": USE_THRESHOLD_SUM,
            "max_total_operators": MAX_TOTAL_OPERATORS,
            "deduplication": [
                "same truth table",
                "same function up to variable renaming",
            ],
        }
        for length, records in records_by_length.items()
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for length, records in records_by_length.items():
        jsonl_path = OUTPUT_DIR / f"hkm_len{length}_dedup.jsonl"
        txt_path = OUTPUT_DIR / f"hkm_len{length}_dedup.txt"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for index, record in enumerate(records, start=1):
                payload = {
                    "formula_id": f"len{length}_{index:04d}",
                    "length": record.length,
                    "formula": record.formula,
                    "truth_table": record.truth_table,
                    "canonical_signature": record.canonical_signature,
                    "negation_count": record.negation_count,
                    "binary_operator_count": record.binary_operator_count,
                    "total_operator_count": record.total_operator_count,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with txt_path.open("w", encoding="utf-8") as f:
            f.write(
                f"# length={length} count={len(records)} "
                f"paper_style_constraints=True "
                f"allow_not={ALLOW_NOT} "
                f"use_threshold_sum={USE_THRESHOLD_SUM} "
                f"max_total_operators={MAX_TOTAL_OPERATORS}\n"
            )
            for index, record in enumerate(records, start=1):
                f.write(
                    f"{index:04d}\t{record.formula}\t"
                    f"neg={record.negation_count}\t"
                    f"ops={record.total_operator_count}\t"
                    f"sig={record.canonical_signature}\n"
                )


def main() -> None:
    print("HKM template generation mode: paper-style without thresholded SUM")
    print(
        f"constraints: lengths={LENGTHS}, "
        f"operators={BINARY_OPERATORS} + NOT={ALLOW_NOT}, "
        f"max_total_operators={MAX_TOTAL_OPERATORS}"
    )
    records_by_length: Dict[int, List[FormulaRecord]] = {}
    for length in LENGTHS:
        records = enumerate_formulas_for_length(length)
        records_by_length[length] = records
        print(f"length={length} dedup_count={len(records)}")

    write_outputs(records_by_length)
    print(f"saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

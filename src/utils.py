"""
src/utils.py
Core data structures, types, puzzle loading, and arithmetic helpers.
Shared by all solver modules and the verifier.
"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Instruction(NamedTuple):
    """One register-machine instruction: result = reg[src1] OP reg[src2] → reg[dst]"""
    op: str    # one of '+', '-', '*', '/'
    src1: int  # register index [0, 47]
    src2: int  # register index [0, 47]
    dst: int   # register index [0, 47]

    def to_list(self) -> list:
        return [self.op, self.src1, self.src2, self.dst]


class ValueOp(NamedTuple):
    """Value-level operation (register-index-free), used internally during search."""
    op: str
    a: int   # operand value (must be in pool at time of execution)
    b: int   # operand value (must be in pool at time of execution)
    result: int  # = apply_op(op, a, b)


@dataclass
class Puzzle:
    id: str
    seeds: list[int]
    targets: list[int]
    value_limit: int
    registers: int = 48
    # practice-only metadata
    family: Optional[str] = None
    design_tier: Optional[str] = None
    lower_bound: Optional[int] = None

    @property
    def seed_set(self) -> frozenset[int]:
        return frozenset(self.seeds)

    @property
    def target_set(self) -> frozenset[int]:
        return frozenset(self.targets)


# ---------------------------------------------------------------------------
# Puzzle loading
# ---------------------------------------------------------------------------

def load_puzzles(path: str) -> list[Puzzle]:
    """Load puzzles.csv → list[Puzzle]."""
    puzzles = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            puzzles.append(Puzzle(
                id=row["id"],
                seeds=json.loads(row["starting_values"]),
                targets=json.loads(row["targets"]),
                value_limit=int(row["value_limit"]),
                registers=int(row.get("registers", 48)),
            ))
    return puzzles


def load_practice_puzzles(path: str) -> list[Puzzle]:
    """Load practice_puzzles.csv → list[Puzzle] with family/tier/lb metadata."""
    puzzles = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lb = row.get("lower_bound", "")
            puzzles.append(Puzzle(
                id=row["id"],
                seeds=json.loads(row["starting_values"]),
                targets=json.loads(row["targets"]),
                value_limit=int(row["value_limit"]),
                registers=int(row.get("registers", 48)),
                family=row.get("family"),
                design_tier=row.get("design_tier"),
                lower_bound=int(lb) if lb else None,
            ))
    return puzzles


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

VALID_OPS = ("+", "-", "*", "/")


def apply_op(op: str, a: int, b: int, value_limit: int = 10**19) -> Optional[int]:
    """
    Apply op to (a, b). Returns the integer result if valid under competition
    rules, else None.
    Rules:
      - Result must be a positive integer (> 0).
      - Subtraction: a > b required (strict).
      - Division: exact (a % b == 0), b != 0.
      - Result <= value_limit.
    """
    if op == "+":
        r = a + b
    elif op == "-":
        if a <= b:
            return None
        r = a - b
    elif op == "*":
        r = a * b
    elif op == "/":
        if b == 0 or a % b != 0:
            return None
        r = a // b
    else:
        return None

    if r <= 0 or r > value_limit:
        return None
    return r


def build_one_op_map(pool_vals: set[int], value_limit: int) -> dict[int, ValueOp]:
    """
    Enumerate all values reachable in exactly one operation from pool × pool.
    Returns {result_value: ValueOp} (first path found per result value).
    Does NOT include values already in pool_vals.
    """
    F: dict[int, ValueOp] = {}
    pool_list = sorted(pool_vals)
    for a in pool_list:
        for b in pool_list:
            for op in VALID_OPS:
                r = apply_op(op, a, b, value_limit)
                if r is not None and r not in pool_vals and r not in F:
                    F[r] = ValueOp(op=op, a=a, b=b, result=r)
    return F


# ---------------------------------------------------------------------------
# Program serialisation helpers
# ---------------------------------------------------------------------------

def program_to_json(program: list[Instruction]) -> str:
    """Convert list[Instruction] to the competition's JSON string format."""
    return json.dumps([inst.to_list() for inst in program])


def program_from_json(s: str) -> list[Instruction]:
    """Parse a JSON program string back to list[Instruction]."""
    return [Instruction(op=row[0], src1=row[1], src2=row[2], dst=row[3])
            for row in json.loads(s)]


def write_submission(results: dict[str, list[Instruction]], out_path: str) -> None:
    """Write a results dict {puzzle_id: program} to a valid submission CSV."""
    import csv as _csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["id", "program"])
        for pid in sorted(results.keys()):
            w.writerow([pid, program_to_json(results[pid])])

"""
src/verifier.py
Competition-spec simulator and verifier. This is the oracle — every candidate
program must pass through here before being accepted or submitted.

Usage:
    from src.verifier import verify_program, replay_submission
"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from typing import Optional
from src.utils import Puzzle, Instruction, apply_op, program_from_json


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    valid: bool
    instruction_count: int
    targets_hit: set[int]
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Single-program verifier
# ---------------------------------------------------------------------------

def verify_program(puzzle: Puzzle, program: list[Instruction]) -> VerifyResult:
    """
    Replay `program` on `puzzle` and check all competition rules.

    Rules checked (each produces a distinct error string):
      1. Register index out of range [0, N_REGS-1].
      2. Reading an uninitialized register.
      3. Subtraction result ≤ 0 (a must be strictly > b).
      4. Division: b == 0, or a % b != 0.
      5. Any result > value_limit.
      6. Any result ≤ 0 (catches add/mul edge cases; shouldn't happen with valid inputs
         but we guard anyway).
      7. After all instructions, at least one target was never produced.

    Returns a VerifyResult. `.valid` is True iff all rules pass and all targets hit.
    """
    N_REGS = puzzle.registers
    regs: list[Optional[int]] = [None] * N_REGS
    for i, v in enumerate(puzzle.seeds):
        regs[i] = v

    targets_remaining = set(puzzle.targets)
    # Seed values that happen to be targets are already hit
    for s in puzzle.seeds:
        targets_remaining.discard(s)

    for step, inst in enumerate(program):
        op, s1, s2, dst = inst.op, inst.src1, inst.src2, inst.dst

        # --- Rule 1: register indices in range ---
        for name, idx in (("src1", s1), ("src2", s2), ("dst", dst)):
            if not (0 <= idx < N_REGS):
                return VerifyResult(
                    valid=False, instruction_count=step,
                    targets_hit=set(puzzle.targets) - targets_remaining,
                    error=f"Step {step}: {name} register {idx} out of range [0,{N_REGS-1}]"
                )

        # --- Rule 2: uninitialized reads ---
        if regs[s1] is None:
            return VerifyResult(
                valid=False, instruction_count=step,
                targets_hit=set(puzzle.targets) - targets_remaining,
                error=f"Step {step}: src1 register r{s1} is uninitialized"
            )
        if regs[s2] is None:
            return VerifyResult(
                valid=False, instruction_count=step,
                targets_hit=set(puzzle.targets) - targets_remaining,
                error=f"Step {step}: src2 register r{s2} is uninitialized"
            )

        a, b = regs[s1], regs[s2]

        # --- Rules 3-6: arithmetic validity ---
        result = apply_op(op, a, b, puzzle.value_limit)
        if result is None:
            # Distinguish reason
            if op == "-" and a <= b:
                reason = f"subtraction not strictly positive: {a} - {b} = {a - b}"
            elif op == "/" and b == 0:
                reason = "division by zero"
            elif op == "/" and a % b != 0:
                reason = f"inexact division: {a} % {b} = {a % b}"
            else:
                r_raw = a + b if op == "+" else a * b
                if r_raw > puzzle.value_limit:
                    reason = f"result {r_raw} exceeds value_limit {puzzle.value_limit}"
                else:
                    reason = f"result ≤ 0"
            return VerifyResult(
                valid=False, instruction_count=step,
                targets_hit=set(puzzle.targets) - targets_remaining,
                error=f"Step {step} [{op} r{s1}={a} r{s2}={b}]: {reason}"
            )

        # Write result (read-before-write semantics already handled above)
        regs[dst] = result
        targets_remaining.discard(result)

    # --- Rule 7: all targets produced ---
    if targets_remaining:
        missed = sorted(targets_remaining)[:5]
        return VerifyResult(
            valid=False, instruction_count=len(program),
            targets_hit=set(puzzle.targets) - targets_remaining,
            error=f"Missing {len(targets_remaining)} targets: {missed}{'...' if len(targets_remaining)>5 else ''}"
        )

    return VerifyResult(
        valid=True, instruction_count=len(program),
        targets_hit=set(puzzle.targets)
    )


# ---------------------------------------------------------------------------
# Bulk verifier (full submission)
# ---------------------------------------------------------------------------

def replay_submission(
    puzzles_path: str,
    submission_path: str,
    *,
    verbose: bool = False,
) -> int:
    """
    Verify every row in submission_path against puzzles_path.
    Raises RuntimeError on the first invalid program.
    Returns total instruction count.
    """
    from src.utils import load_puzzles
    puzzles = {p.id: p for p in load_puzzles(puzzles_path)}
    total = 0
    seen = set()

    with open(submission_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            if pid not in puzzles:
                raise RuntimeError(f"Unknown puzzle id '{pid}' in submission")
            if pid in seen:
                raise RuntimeError(f"Duplicate puzzle id '{pid}' in submission")
            seen.add(pid)

            program = program_from_json(row["program"])
            result = verify_program(puzzles[pid], program)
            if not result.valid:
                raise RuntimeError(f"Puzzle '{pid}' INVALID: {result.error}")
            total += result.instruction_count
            if verbose:
                print(f"  {pid}: {result.instruction_count} instructions — OK")

    missing = set(puzzles.keys()) - seen
    if missing:
        raise RuntimeError(f"Submission missing {len(missing)} puzzles: {sorted(missing)[:5]}")

    return total


# ---------------------------------------------------------------------------
# Convenience: verify one puzzle quickly (returns bool + count)
# ---------------------------------------------------------------------------

def quick_verify(puzzle: Puzzle, program: list[Instruction]) -> tuple[bool, int, str]:
    """Returns (valid, instruction_count, error_msg)."""
    r = verify_program(puzzle, program)
    return r.valid, r.instruction_count, r.error or ""

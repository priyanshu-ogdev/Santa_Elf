"""
tests/test_verifier.py
Tests for the verifier (competition-spec simulator).
Run: python -m pytest tests/test_verifier.py -v
"""
import pytest
from src.utils import Puzzle, Instruction, load_puzzles
from src.verifier import verify_program, replay_submission, quick_verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_puzzle(seeds, targets, value_limit=10**18):
    return Puzzle(id="test", seeds=seeds, targets=targets, value_limit=value_limit)


# ---------------------------------------------------------------------------
# Basic validity tests
# ---------------------------------------------------------------------------

def test_empty_program_all_seeds_are_targets():
    """If all targets are seeds, an empty program is valid."""
    p = make_puzzle(seeds=[1, 2, 3], targets=[1, 2, 3])
    r = verify_program(p, [])
    assert r.valid
    assert r.instruction_count == 0


def test_seed_plus_seed():
    """1 + 1 = 2 satisfies target 2."""
    p = make_puzzle(seeds=[1], targets=[2])
    prog = [Instruction("+", 0, 0, 1)]
    r = verify_program(p, prog)
    assert r.valid
    assert r.instruction_count == 1


def test_subtraction_requires_a_gt_b():
    """a - b is invalid when a <= b."""
    p = make_puzzle(seeds=[5, 10], targets=[5])
    # 5 - 10 = invalid (a < b)
    prog = [Instruction("-", 0, 1, 2)]  # r0=5, r1=10 → 5 - 10 invalid
    r = verify_program(p, prog)
    assert not r.valid
    assert "subtraction" in r.error.lower()


def test_division_must_be_exact():
    """7 / 2 is invalid (not exact)."""
    p = make_puzzle(seeds=[7, 2], targets=[3])
    prog = [Instruction("/", 0, 1, 2)]
    r = verify_program(p, prog)
    assert not r.valid
    assert "inexact" in r.error.lower()


def test_value_limit_exceeded():
    """Result exceeding value_limit is invalid."""
    p = make_puzzle(seeds=[5, 5], targets=[10], value_limit=9)
    prog = [Instruction("+", 0, 1, 2)]
    r = verify_program(p, prog)
    assert not r.valid
    assert "value_limit" in r.error.lower() or "exceeds" in r.error.lower()


def test_missing_target():
    """Program that doesn't hit a target is invalid."""
    p = make_puzzle(seeds=[1], targets=[5])
    prog = [Instruction("+", 0, 0, 1)]  # produces 2, not 5
    r = verify_program(p, prog)
    assert not r.valid
    assert "Missing" in r.error


def test_uninitialized_register():
    """Reading an uninitialized register is invalid."""
    p = make_puzzle(seeds=[1], targets=[2])
    prog = [Instruction("+", 0, 5, 1)]  # r5 never initialized
    r = verify_program(p, prog)
    assert not r.valid
    assert "uninitialized" in r.error.lower()


def test_register_out_of_range():
    """Register index ≥ 48 is invalid."""
    p = make_puzzle(seeds=[1], targets=[2])
    prog = [Instruction("+", 0, 48, 1)]  # r48 out of range
    r = verify_program(p, prog)
    assert not r.valid
    assert "out of range" in r.error.lower()


def test_target_hit_then_overwritten():
    """A target value only needs to appear once at any point; later overwriting is fine."""
    # seeds=[1], target=2; produce 2 in r1, then overwrite r1 with 3
    p = make_puzzle(seeds=[1], targets=[2])
    prog = [
        Instruction("+", 0, 0, 1),  # r1 = 2 ← target hit
        Instruction("+", 0, 1, 1),  # r1 = 3 ← overwrite
    ]
    r = verify_program(p, prog)
    assert r.valid
    assert r.instruction_count == 2


def test_seed_equal_target_is_free():
    """When a seed == a target, that target costs 0 instructions."""
    p = make_puzzle(seeds=[7, 42], targets=[7, 42])
    r = verify_program(p, [])
    assert r.valid


def test_self_division_gives_one():
    """seed / seed = 1 (valid exact division)."""
    p = make_puzzle(seeds=[5], targets=[1])
    prog = [Instruction("/", 0, 0, 1)]
    r = verify_program(p, prog)
    assert r.valid


def test_multiplication():
    """3 * 5 = 15."""
    p = make_puzzle(seeds=[3, 5], targets=[15])
    prog = [Instruction("*", 0, 1, 2)]
    r = verify_program(p, prog)
    assert r.valid


# ---------------------------------------------------------------------------
# Bulk replay
# ---------------------------------------------------------------------------

def test_replay_submission_baseline():
    """Baseline submission replays with the documented score (1,490,379)."""
    total = replay_submission("data/puzzles.csv", "data/sample_submission.csv")
    assert total == 1_490_379, f"Expected 1490379, got {total}"


# ---------------------------------------------------------------------------
# quick_verify interface
# ---------------------------------------------------------------------------

def test_quick_verify_valid():
    p = make_puzzle(seeds=[1], targets=[2])
    prog = [Instruction("+", 0, 0, 1)]
    ok, count, err = quick_verify(p, prog)
    assert ok
    assert count == 1
    assert err == ""


def test_quick_verify_invalid():
    p = make_puzzle(seeds=[1], targets=[5])
    prog = [Instruction("+", 0, 0, 1)]  # produces 2, not 5
    ok, count, err = quick_verify(p, prog)
    assert not ok
    assert err != ""

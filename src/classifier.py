"""
src/classifier.py
Lightweight puzzle classifier. Labels each puzzle with a strategy hint so the
orchestrator can route it to the cheapest solver module.

The classifier runs in <1ms per puzzle. All thresholds were chosen from the
measured data in docs/06_exploratory_data_analysis.md.
"""
from __future__ import annotations
import math
from enum import Enum, auto
from src.utils import Puzzle


class PuzzleClass(Enum):
    TINY        = auto()   # value_limit < 100; solvable by exact BFS
    SMOOTH      = auto()   # ≥35% of targets are factor-smooth (lpf ≤ 2000)
    CLUSTERED   = auto()   # ≥5% of adjacent sorted target pairs within 0.5%
    GENERIC     = auto()   # everything else → meet-in-the-middle


def _smoothness_fraction(targets: list[int], lpf_threshold: int = 2000) -> float:
    """Fraction of targets whose largest prime factor is ≤ lpf_threshold."""
    if not targets:
        return 0.0
    smooth = sum(1 for t in targets if _largest_prime_factor(t) <= lpf_threshold)
    return smooth / len(targets)


def _largest_prime_factor(n: int) -> int:
    """Return the largest prime factor of n (n >= 1). Returns 1 for n==1."""
    if n <= 1:
        return 1
    lpf = 1
    for p in range(2, min(2001, int(math.isqrt(n)) + 1)):
        if n % p == 0:
            lpf = p
            while n % p == 0:
                n //= p
    if n > 1:
        lpf = n  # n itself is a prime > 2000
    return lpf


def _cluster_fraction(targets: list[int], rel_tol: float = 0.005) -> float:
    """Fraction of adjacent sorted target pairs within rel_tol of each other."""
    if len(targets) < 2:
        return 0.0
    t = sorted(targets)
    close = sum(
        1 for i in range(len(t) - 1)
        if t[i] > 0 and abs(t[i + 1] - t[i]) / t[i] <= rel_tol
    )
    return close / (len(t) - 1)


def classify(puzzle: Puzzle) -> PuzzleClass:
    """Return the PuzzleClass label for a puzzle."""
    # TINY: tiny value_limit — small state space, BFS is optimal
    if puzzle.value_limit < 100:
        return PuzzleClass.TINY

    # SMOOTH: dominant factor structure
    if _smoothness_fraction(puzzle.targets) >= 0.35:
        return PuzzleClass.SMOOTH

    # CLUSTERED: many nearby target pairs
    if _cluster_fraction(puzzle.targets) >= 0.05:
        return PuzzleClass.CLUSTERED

    return PuzzleClass.GENERIC


def classify_all(puzzles: list[Puzzle]) -> dict[str, PuzzleClass]:
    """Classify a list of puzzles. Returns {puzzle_id: PuzzleClass}."""
    return {p.id: classify(p) for p in puzzles}

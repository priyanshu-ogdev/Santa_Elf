"""
src/solvers/factor_smooth.py
Factor-decomposition solver for puzzles where many targets have small prime factors.

Strategy:
  1. Factor each target using trial division up to 2000 (covering smooth targets).
  2. Build a "factor basis": prime factors ≤ 2000 that appear across the puzzle's
     smooth targets. Build each basis prime from the seeds using M-i-M.
  3. For each smooth target, express it as a product/combination of basis elements:
       t = a * b (if both live in pool) → 1 instruction
       t = a * b ± c                  → 2–3 instructions
  4. Non-smooth targets fall back to M-i-M core.
"""
from __future__ import annotations
from src.utils import Puzzle, Instruction, apply_op
from src.solvers.meet_in_middle import LivePool, find_target


def _trial_factor(n: int, limit: int = 2000) -> tuple[dict[int, int], int]:
    """
    Trial division up to `limit`. Returns (prime_exponents_dict, remainder).
    If remainder == 1, n is smooth (all factors ≤ limit).
    """
    factors: dict[int, int] = {}
    for p in range(2, min(limit + 1, int(n**0.5) + 1)):
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    return factors, n  # remainder n (if > 1, it's a prime > limit)


def _collect_basis_primes(targets: list[int]) -> set[int]:
    """Collect all small prime factors (≤ 2000) across all targets."""
    primes: set[int] = set()
    for t in targets:
        factors, _ = _trial_factor(t)
        primes.update(factors.keys())
    return primes


def solve(puzzle: Puzzle) -> list[Instruction]:
    """
    Solve using factor decomposition for smooth targets; M-i-M for the rest.
    """
    pool = LivePool(puzzle.seeds)
    instructions: list[Instruction] = []
    remaining = set(puzzle.targets)

    # Mark seeds that equal targets as free
    for s in puzzle.seeds:
        remaining.discard(s)

    # --- Phase 1: build the factor basis from seeds ---
    basis_primes = _collect_basis_primes(list(remaining))
    # Remove primes already in pool (seeds)
    basis_to_build = [p for p in sorted(basis_primes) if not pool.has(p)]

    for prime in basis_to_build:
        if prime > puzzle.value_limit:
            continue  # can't build this prime; skip
        path = find_target(prime, pool, puzzle.value_limit, max_depth=3)
        if path is not None:
            instructions.extend(pool.emit_path(path, remaining))

    # --- Phase 2: process each target ---
    for target in sorted(remaining):  # ascending order
        if pool.has(target):
            remaining.discard(target)
            continue

        factors, remainder = _trial_factor(target)

        if remainder == 1:
            # Fully smooth: express as product of prime powers in pool
            derived = _build_from_factors(target, factors, pool, instructions, remaining, puzzle.value_limit)
        else:
            derived = False

        if not derived:
            # Fall back to M-i-M (which may itself fall back to binary)
            path = find_target(target, pool, puzzle.value_limit, max_depth=3)
            if path is not None:
                instructions.extend(pool.emit_path(path, remaining))
            else:
                _fs_fallback_target(target, pool, instructions, remaining, puzzle)

        remaining.discard(target)

    return instructions


def _build_from_factors(
    target: int,
    factors: dict[int, int],
    pool: LivePool,
    instructions: list[Instruction],
    remaining: set[int],
    value_limit: int,
) -> bool:
    """
    Try to build `target` by multiplying together its prime powers.
    Uses values currently in pool. Returns True if successful.
    """
    # Build prime powers that aren't yet in pool
    for p, exp in factors.items():
        power = p ** exp
        if power > value_limit:
            return False
        if not pool.has(power):
            # Try to build p^exp from p (if p is in pool)
            if pool.has(p):
                current = p
                for _ in range(exp - 1):
                    new_val = current * p
                    if new_val > value_limit:
                        return False
                    r = apply_op("*", current, p, value_limit)
                    if r is None:
                        return False
                    instructions.append(pool.emit("*", current, p, new_val, remaining))
                    current = new_val
            else:
                return False  # prime not in pool, can't build

    # Now multiply all prime powers together
    # Collect all prime powers
    prime_powers = []
    for p, exp in sorted(factors.items()):
        prime_powers.append(p ** exp)

    if not all(pool.has(pw) for pw in prime_powers):
        return False

    if len(prime_powers) == 1:
        # Only one prime power — it must equal target
        return prime_powers[0] == target

    # Multiply them pairwise
    accum = prime_powers[0]
    for pw in prime_powers[1:]:
        new_val = accum * pw
        if new_val > value_limit:
            return False
        r = apply_op("*", accum, pw, value_limit)
        if r is None:
            return False
        instructions.append(pool.emit("*", accum, pw, new_val, remaining))
        accum = new_val

    return accum == target


def _fs_fallback_target(
    target: int,
    pool: LivePool,
    instructions: list[Instruction],
    remaining: set[int],
    puzzle: Puzzle,
) -> None:
    from src.solvers.meet_in_middle import _emit_binary_target
    if not pool.has(1):
        s0 = puzzle.seeds[0]
        instructions.append(pool.emit("/", s0, s0, 1, remaining))
    _emit_binary_target(target, pool, instructions, remaining, puzzle.value_limit)

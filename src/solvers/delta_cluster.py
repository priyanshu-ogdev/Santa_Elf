"""
src/solvers/delta_cluster.py
Cluster-aware solver for puzzles where many targets are close together.

Strategy:
  1. Sort targets and find clusters: runs where adjacent targets are within
     rel_tol of each other.
  2. For each cluster, pick the median value as an anchor and solve it using
     the M-i-M core.
  3. Each other member is derived as: anchor ± delta, where delta is computed
     cheaply (usually 1 instruction if delta is reachable from pool).
  4. Non-clustered targets are solved by the M-i-M core directly.

Expected saving: 1–2 instructions per additional cluster member (vs. M-i-M alone).
"""
from __future__ import annotations
from src.utils import Puzzle, Instruction, apply_op
from src.solvers.meet_in_middle import LivePool, find_target


REL_TOL = 0.005  # 0.5% — targets within this ratio are considered clustered


def _find_clusters(targets: list[int]) -> list[list[int]]:
    """
    Group sorted targets into clusters where adjacent members are within REL_TOL.
    Returns a list of cluster groups (each group is a sorted list of targets).
    Single-element groups are still returned (non-clustered).
    """
    if not targets:
        return []
    t = sorted(targets)
    clusters: list[list[int]] = []
    current = [t[0]]
    for i in range(1, len(t)):
        ref = current[0]
        if ref > 0 and abs(t[i] - ref) / ref <= REL_TOL:
            current.append(t[i])
        else:
            clusters.append(current)
            current = [t[i]]
    clusters.append(current)
    return clusters


def _solve_via_mim(
    target: int,
    pool: LivePool,
    instructions: list[Instruction],
    remaining: set[int],
    puzzle: "Puzzle",
) -> None:
    """Solve one target via M-i-M or fallback binary construction."""
    if pool.has(target):
        return
    path = find_target(target, pool, puzzle.value_limit, max_depth=3)
    if path is not None:
        instructions.extend(pool.emit_path(path, remaining))
    else:
        _dc_fallback_target(target, pool, instructions, remaining, puzzle)


def solve(puzzle: Puzzle) -> list[Instruction]:
    """
    Solve using cluster detection + M-i-M for anchors + delta derivation.
    """
    pool = LivePool(puzzle.seeds)
    instructions: list[Instruction] = []
    remaining = set(puzzle.targets)

    # Mark seeds that equal targets as free
    for s in puzzle.seeds:
        remaining.discard(s)

    clusters = _find_clusters(list(puzzle.targets))

    for cluster in clusters:
        if len(cluster) == 1:
            t = cluster[0]
            _solve_via_mim(t, pool, instructions, remaining, puzzle)
            remaining.discard(t)
        else:
            # Multi-target cluster: build anchor then derive others by delta
            anchor = cluster[len(cluster) // 2]  # median

            # --- Solve anchor ---
            _solve_via_mim(anchor, pool, instructions, remaining, puzzle)
            remaining.discard(anchor)

            # --- Derive cluster members from anchor ---
            for t in cluster:
                if t == anchor:
                    continue
                if pool.has(t):
                    remaining.discard(t)
                    continue

                delta = abs(t - anchor)
                if delta == 0:
                    remaining.discard(t)
                    continue

                derived = False

                # Strategy 1: delta already in pool → 1 instruction
                if pool.has(delta):
                    op = "+" if t > anchor else "-"
                    r = apply_op(op, anchor, delta, puzzle.value_limit)
                    if r == t:
                        instructions.append(pool.emit(op, anchor, delta, t, remaining))
                        derived = True

                # Strategy 2: build delta via M-i-M (depth ≤ 2), then combine
                if not derived and delta <= puzzle.value_limit:
                    path_delta = find_target(delta, pool, puzzle.value_limit, max_depth=2)
                    if path_delta is not None:
                        # Extend path_delta with the combine step
                        op = "+" if t > anchor else "-"
                        r = apply_op(op, anchor, delta, puzzle.value_limit)
                        if r == t:
                            from src.utils import ValueOp
                            combine_vop = ValueOp(op=op, a=anchor, b=delta, result=t)
                            full_path = path_delta + [combine_vop]
                            instructions.extend(pool.emit_path(full_path, remaining))
                            derived = True

                # Strategy 3: solve t directly with M-i-M or fallback
                if not derived:
                    _solve_via_mim(t, pool, instructions, remaining, puzzle)

                remaining.discard(t)

    return instructions


def _dc_fallback_target(
    target: int,
    pool: LivePool,
    instructions: list[Instruction],
    remaining: set[int],
    puzzle: Puzzle,
) -> None:
    """Emit binary construction for a single target using the pool."""
    from src.solvers.meet_in_middle import _emit_binary_target
    if not pool.has(1):
        s0 = puzzle.seeds[0]
        inst = pool.emit("/", s0, s0, 1, remaining)
        instructions.append(inst)
    _emit_binary_target(target, pool, instructions, remaining, puzzle.value_limit)

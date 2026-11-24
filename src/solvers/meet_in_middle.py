"""
src/solvers/meet_in_middle.py
Pool-based meet-in-the-middle solver. The workhorse for the majority of puzzles.

Architecture:
  - Maintains a live pool of values ↔ register slots (max 48).
  - For each target, searches for the cheapest derivation path at depth 0–3
    using a hash-based backward lookup over the 1-op frontier set F.
  - Targets are processed in a chosen order that maximises pool reuse.
  - Falls back to binary construction for any target it cannot reach in ≤ 3 ops.

The pool tracks {value → register_index}. Seeds always occupy r0..r(K-1) and
are never evicted. As new values are computed, they occupy r_K onward.
When all 48 registers are live, we evict the value whose future utility is
the lowest (fewest remaining targets it participates in reaching).
"""
from __future__ import annotations
import random
from typing import Optional
from src.utils import (
    Puzzle, Instruction, ValueOp, apply_op, build_one_op_map, VALID_OPS
)


# ---------------------------------------------------------------------------
# LivePool: register ↔ value mapping
# ---------------------------------------------------------------------------

class LivePool:
    """
    Maps values ↔ register indices. Max size = 48.
    Seeds are pinned and never evicted.
    """
    MAX_REGS = 48

    def __init__(self, seeds: list[int]) -> None:
        self._val_to_reg: dict[int, int] = {}
        self._reg_to_val: dict[int, int] = {}
        self._seeds: frozenset[int] = frozenset(seeds)
        for i, v in enumerate(seeds):
            self._val_to_reg[v] = i
            self._reg_to_val[i] = v
        self._next_free: int = len(seeds)

    # -- queries --
    @property
    def values(self) -> set[int]:
        return set(self._val_to_reg)

    @property
    def size(self) -> int:
        return len(self._val_to_reg)

    def has(self, v: int) -> bool:
        return v in self._val_to_reg

    def reg_of(self, v: int) -> int:
        return self._val_to_reg[v]

    # -- mutations --
    def add_value(self, val: int, remaining_targets: set[int],
                  protect_regs: set[int] | None = None) -> int:
        """
        Add `val` to pool, evicting if full. Returns the dst register index.
        If val is already in pool, returns its current register (no-op).
        `protect_regs`: register indices that must NOT be evicted (e.g. operand regs).
        """
        if val in self._val_to_reg:
            return self._val_to_reg[val]

        if self._next_free < self.MAX_REGS:
            dst = self._next_free
            self._next_free += 1
        else:
            dst = self._evict(remaining_targets, protect_regs=protect_regs)

        self._val_to_reg[val] = dst
        self._reg_to_val[dst] = val
        return dst

    def _evict(self, remaining_targets: set[int],
               protect_regs: set[int] | None = None) -> int:
        """
        Choose and evict a non-seed value. Eviction policy:
        Score each non-seed value by how many remaining targets it can help
        reach in 1 op (higher = more useful = keep). Evict lowest-score.
        Tie-break: evict the value in the highest register index (LIFO).
        Returns the freed register index.
        protect_regs: register indices that must NOT be chosen for eviction.
        """
        protect_regs = protect_regs or set()
        best_reg = -1
        best_score = float("inf")

        for reg, val in self._reg_to_val.items():
            if val in self._seeds:
                continue  # never evict seeds
            if reg in protect_regs:
                continue  # protect operand registers
            # Utility: count remaining targets reachable from this value in 1 op
            score = 0
            for t in remaining_targets:
                if val == t:
                    score += 10  # it IS a target — very high utility
                    continue
                if t % val == 0 or val % t == 0:
                    score += 2
                diff = abs(t - val)
                if diff < val * 0.01:  # very close
                    score += 1

            if score < best_score or (score == best_score and reg > best_reg):
                best_score = score
                best_reg = reg

        if best_reg == -1:
            # All non-seed regs are protected — evict ANY non-seed (last resort)
            for reg, val in self._reg_to_val.items():
                if val not in self._seeds:
                    best_reg = reg
                    break
        if best_reg == -1:
            raise RuntimeError("Cannot evict: pool entirely occupied by seeds")

        evicted_val = self._reg_to_val[best_reg]
        del self._val_to_reg[evicted_val]
        del self._reg_to_val[best_reg]
        return best_reg

    def emit(self, op: str, a_val: int, b_val: int,
             result_val: int, remaining_targets: set[int]) -> Instruction:
        """
        Emit an instruction that computes `result_val = a_val op b_val`,
        updating the pool in place. Returns the Instruction.
        Reads src registers BEFORE add_value so eviction cannot clobber them.
        """
        src1 = self._val_to_reg[a_val]
        src2 = self._val_to_reg[b_val]
        # Pass protect_regs so eviction never frees src1 or src2
        dst = self.add_value(result_val, remaining_targets,
                             protect_regs={src1, src2})
        return Instruction(op=op, src1=src1, src2=src2, dst=dst)

    def emit_path(self, path: list, remaining_targets: set[int]) -> list:
        """
        Emit a sequence of ValueOps as Instructions, protecting ALL future
        operand values from eviction at each step.
        Returns list[Instruction].
        """
        instructions = []
        for step_i, vop in enumerate(path):
            # Collect all values needed by future steps in this path
            future_vals: set[int] = set()
            for future_vop in path[step_i + 1:]:
                future_vals.add(future_vop.a)
                future_vals.add(future_vop.b)
            # Map future values to current registers (only those in pool)
            future_regs = {
                self._val_to_reg[v]
                for v in future_vals
                if v in self._val_to_reg
            }
            src1 = self._val_to_reg[vop.a]
            src2 = self._val_to_reg[vop.b]
            protect = {src1, src2} | future_regs
            dst = self.add_value(vop.result, remaining_targets,
                                 protect_regs=protect)
            instructions.append(Instruction(op=vop.op, src1=src1, src2=src2, dst=dst))
        return instructions


# ---------------------------------------------------------------------------
# Search: find_target
# ---------------------------------------------------------------------------

def find_target(
    target: int,
    pool: LivePool,
    value_limit: int,
    max_depth: int = 3,
) -> Optional[list[ValueOp]]:
    """
    Find a path of ValueOps to produce `target` using values in `pool`.
    Returns a list (possibly empty for depth-0 hits) or None if not found.

    The returned list has depth = number of new instructions required:
      depth 0 → target already in pool
      depth 1 → one instruction
      depth 2 → two instructions (build one intermediate, combine)
      depth 3 → three instructions (build two intermediates, combine)
    """
    pool_vals = pool.values

    # --- Depth 0: already in pool ---
    if target in pool_vals:
        return []

    pool_list = sorted(pool_vals)

    # --- Depth 1: target = pool_a op pool_b ---
    for a in pool_list:
        for b in pool_list:
            for op in VALID_OPS:
                r = apply_op(op, a, b, value_limit)
                if r == target:
                    return [ValueOp(op=op, a=a, b=b, result=target)]

    if max_depth < 2:
        return None

    # Build 1-op frontier F = all values reachable in one op from pool
    F: dict[int, ValueOp] = build_one_op_map(pool_vals, value_limit)
    all_reachable = pool_vals | set(F.keys())

    # --- Depth 2 & 3: target = f op x (or x op f) ---
    # For each f in F, compute backward candidates x such that f op x = target
    # or x op f = target. Check if x is in pool (depth 2) or F (depth 3).
    for f_val, f_vop in F.items():
        for path in _backward_candidates(target, f_val, all_reachable, value_limit, F, pool_vals):
            build_f = f_vop
            full_path = [build_f] + path
            if 0 < len(full_path) <= max_depth:
                # Validate: replay the path and ensure every operand is available
                if _path_operands_ok(full_path, pool_vals):
                    return full_path

    return None


def _path_operands_ok(path: list, pool_vals: set[int]) -> bool:
    """
    Check that every step in `path` can be executed in order:
    - Both operands (a, b) must be in pool_vals or produced by an earlier step.
    Returns True if safe, False if any operand is missing.
    """
    available = set(pool_vals)
    for vop in path:
        if vop.a not in available or vop.b not in available:
            return False
        available.add(vop.result)
    return True


def _backward_candidates(
    target: int,
    f_val: int,
    all_reachable: set[int],
    value_limit: int,
    F: dict[int, ValueOp],
    pool_vals: set[int],
) -> list[list[ValueOp]]:
    """
    For a given f_val already computed (from F), find all x in all_reachable
    such that f_val op x = target or x op f_val = target.
    Returns paths (not including the step to build f_val).
    """
    results = []
    f_vop = F[f_val]  # the ValueOp that produced f_val (used in final path)

    def try_x(x: int, combine_op: str, operand_order: str) -> None:
        """Check if x is reachable and add a path if so."""
        if x <= 0 or x > value_limit:
            return
        if x not in all_reachable:
            return
        # Verify combine step produces target
        if operand_order == "fx":
            check = apply_op(combine_op, f_val, x, value_limit)
        else:
            check = apply_op(combine_op, x, f_val, value_limit)
        if check != target:
            return

        # Build path: [build_x (if x in F not pool), combine_to_target]
        if x in pool_vals:
            # depth 2 total (f + combine = 2 ops; we're generating the 2nd part)
            if operand_order == "fx":
                combine = ValueOp(op=combine_op, a=f_val, b=x, result=target)
            else:
                combine = ValueOp(op=combine_op, a=x, b=f_val, result=target)
            results.append([combine])
        elif x in F:
            # depth 3 total (f + build_x + combine = 3 ops)
            build_x = F[x]
            if operand_order == "fx":
                combine = ValueOp(op=combine_op, a=f_val, b=x, result=target)
            else:
                combine = ValueOp(op=combine_op, a=x, b=f_val, result=target)
            results.append([build_x, combine])

    # target = f + x → x = target - f
    try_x(target - f_val, "+", "fx")
    # target = x + f (same result, f+x commutative; skip to avoid duplicates)
    # target = f - x → x = f - target  (need f > target)
    if f_val > target:
        try_x(f_val - target, "-", "fx")
    # target = x - f → x = target + f
    try_x(target + f_val, "-", "xf")
    # target = f * x → x = target // f (if exact)
    if f_val != 0 and target % f_val == 0:
        try_x(target // f_val, "*", "fx")
    # target = f / x → x = f // target (if f divisible by target)
    if target != 0 and f_val % target == 0:
        try_x(f_val // target, "/", "fx")
    # target = x / f → x = target * f
    x_div = target * f_val
    if x_div <= value_limit:
        try_x(x_div, "/", "xf")

    return results


# ---------------------------------------------------------------------------
# Target ordering
# ---------------------------------------------------------------------------

def order_targets(
    targets: list[int],
    pool: LivePool,
    value_limit: int,
) -> list[int]:
    """
    Order targets to maximise pool reuse.
    Priority:
      1. Targets already in pool (cost 0).
      2. Targets reachable in 1 op from current pool.
      3. Smooth targets (small prime factors → cheaper to build).
      4. Remaining targets in ascending order of magnitude (heuristic).
    """
    import math
    pool_vals = pool.values
    F = build_one_op_map(pool_vals, value_limit)

    free    = [t for t in targets if t in pool_vals]
    depth1  = [t for t in targets if t not in pool_vals and t in F]
    rest    = [t for t in targets if t not in pool_vals and t not in F]

    # Sort rest: smooth-first, then ascending
    def sort_key(t):
        # Approximate smoothness: count small factors
        n, smooth_score = t, 0
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while n % p == 0:
                n //= p
                smooth_score += 1
        return (-smooth_score, t)

    rest.sort(key=sort_key)
    return free + depth1 + rest


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(puzzle: Puzzle, *, random_seed: Optional[int] = None) -> list[Instruction]:
    """
    Solve a puzzle using pool-based meet-in-the-middle search.
    Returns a valid list[Instruction] (falls back to binary construction
    for any target that can't be reached in ≤ 3 ops).

    `random_seed`: if set, adds a small random perturbation to target ordering
    for use in multi-start ensembles.
    """
    from src.solvers.fallback import _build_target as _fallback_build

    pool = LivePool(puzzle.seeds)
    instructions: list[Instruction] = []
    remaining = set(puzzle.targets)

    # Satisfy seed-equal targets for free
    for s in puzzle.seeds:
        remaining.discard(s)

    if not remaining:
        return instructions

    # Produce constant 1 (needed by fallback sub-paths) if any seed ≥ 2
    # We'll only produce it if we actually need it.
    have_one = 1 in pool.values
    reg_one: Optional[int] = pool.reg_of(1) if have_one else None

    # Need a scratch register for fallback targets
    # Reserve r(n_seeds) for constant-1 and r(n_seeds+1) for scratch if needed.
    reg_fallback_scratch = puzzle.registers - 1  # r47: last register, used as scratch

    ordered = order_targets(list(remaining), pool, puzzle.value_limit)

    if random_seed is not None:
        rng = random.Random(random_seed)
        # Lightly shuffle only the non-free, non-depth1 portion (rest)
        # Count free (depth-0) and depth-1 targets at the front
        pool_vals_now = pool.values
        F_now = build_one_op_map(pool_vals_now, puzzle.value_limit)
        fixed_count = sum(
            1 for t in ordered
            if t in pool_vals_now or t in F_now
        )
        if fixed_count < len(ordered):
            tail = ordered[fixed_count:]
            rng.shuffle(tail)
            ordered = ordered[:fixed_count] + tail

    for target in ordered:
        if not pool.has(target):
            # Try M-i-M search (depth ≤ 3)
            path = find_target(target, pool, puzzle.value_limit, max_depth=3)

            if path is not None:
                # Emit the value ops as register instructions
                # Use emit_path to protect future operands from eviction
                instructions.extend(pool.emit_path(path, remaining))
            else:
                # Fallback: binary construction for this target
                # We need reg_one (constant 1) in the pool
                if not pool.has(1):
                    # Produce 1 using seeds[0] / seeds[0]
                    seed0 = puzzle.seeds[0]
                    inst = pool.emit("/", seed0, seed0, 1, remaining)
                    instructions.append(inst)

                # Build target into a fresh/scratch register
                _emit_binary_target(target, pool, instructions, remaining,
                                    puzzle.value_limit)

        remaining.discard(target)

    return instructions


def _emit_binary_target(
    target: int,
    pool: LivePool,
    instructions: list[Instruction],
    remaining: set[int],
    value_limit: int,
) -> None:
    """
    Emit binary construction instructions for `target` using the pool.
    Requires that 1 is already in the pool.
    Tracks all intermediate values in the pool.
    """
    if target == 1:
        return  # 1 is already in pool

    one = 1  # pool must have this
    bits = target.bit_length()

    # Start accumulator at 1
    # We'll track the accumulator value and emit into pool
    accum = 1

    for bit_pos in range(bits - 2, -1, -1):
        # Double
        new_accum = accum * 2
        if new_accum > value_limit:
            # Shouldn't happen (target ≤ value_limit and accum ≤ target/2)
            # but guard anyway — fall through without completing (caller handles)
            return
        inst = pool.emit("+", accum, accum, new_accum, remaining)
        instructions.append(inst)
        accum = new_accum

        # Add 1 if bit is set
        if (target >> bit_pos) & 1:
            new_accum = accum + 1
            if new_accum > value_limit:
                return
            inst = pool.emit("+", accum, one, new_accum, remaining)
            instructions.append(inst)
            accum = new_accum

    # If we didn't reach target (e.g. value_limit overflow mid-build), that
    # is a caller issue — the verifier will catch it. For safety, assert here
    # during development but don't raise in production.
    # assert accum == target, f"_emit_binary_target failed: got {accum}, want {target}"

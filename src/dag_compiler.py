"""
src/dag_compiler.py
Register-minimal topological scheduler and allocator for arbitrary value DAGs.

Given a value-level acyclic DAG `defs: {val: (op, a, b)}` rooted at `puzzle.seeds`
covering all `puzzle.targets`, compiles it to valid 48-register `Instruction`s:
  1. Verifies strict acyclicity and reachability of all targets.
  2. Evaluates multiple Sethi-Ullman / subtree-affinity / shared-hub topological
     schedules to minimize peak register liveness.
  3. Allocates registers r0..r(N-1) with:
     - Safe seed reclamation once every downstream scheduled node needing a seed
       has completed (`seed_last_needed`).
     - In-place Read-Before-Write destination reuse when an operand dies on step `i`.
     - Cost-aware Belady eviction + recursive re-materialization only if live > N_REGS,
       never evicting a value whose remat path depends on an already-overwritten seed.
"""
from __future__ import annotations
from collections import defaultdict
from src.utils import Puzzle, Instruction


def _extract_needed_subgraph(
    puzzle: Puzzle, defs: dict[int, tuple[str, int, int]]
) -> set[int]:
    """Return the exact set of non-seed nodes reachable from puzzle.targets."""
    seeds = set(puzzle.seeds)
    needed: set[int] = set()
    stk = [t for t in puzzle.targets if t not in seeds]
    vis = set(seeds)
    while stk:
        v = stk.pop()
        if v in vis:
            continue
        if v not in defs:
            raise KeyError(f"Value {v} missing from DAG definitions")
        vis.add(v)
        needed.add(v)
        _, a, b = defs[v]
        if a not in vis:
            stk.append(a)
        if b not in vis:
            stk.append(b)
    return needed


def _compute_heights(
    needed: set[int], seeds: set[int], defs: dict[int, tuple[str, int, int]]
) -> dict[int, int]:
    """Compute topological height of every node; raises ValueError on cycle."""
    height: dict[int, int] = {s: 0 for s in seeds}
    state: dict[int, int] = {}  # 1 = visiting, 2 = done

    for root in needed:
        if root in height:
            continue
        stk: list[tuple[int, int]] = [(root, 0)]
        while stk:
            v, phase = stk.pop()
            if v in seeds:
                continue
            if phase == 0:
                st = state.get(v, 0)
                if st == 2:
                    continue
                if st == 1:
                    raise ValueError(f"Cycle detected involving value {v}")
                state[v] = 1
                stk.append((v, 1))
                _, a, b = defs[v]
                if a not in seeds and state.get(a, 0) != 2:
                    stk.append((a, 0))
                if b not in seeds and b != a and state.get(b, 0) != 2:
                    stk.append((b, 0))
            else:
                _, a, b = defs[v]
                height[v] = max(height.get(a, 0), height.get(b, 0)) + 1
                state[v] = 2
    return height


def _schedule_dfs(
    target_order: list[int],
    seeds: set[int],
    defs: dict[int, tuple[str, int, int]],
    height: dict[int, int],
) -> list[int]:
    """Sethi-Ullman post-order DFS emission (heavier child evaluated first)."""
    emitted: set[int] = set(seeds)
    schedule: list[int] = []

    def emit(root: int) -> None:
        if root in emitted:
            return
        stk: list[tuple[int, int]] = [(root, 0)]
        while stk:
            v, phase = stk.pop()
            if v in emitted:
                continue
            if phase == 0:
                stk.append((v, 1))
                _, a, b = defs[v]
                # Push shallower child FIRST so deeper child is popped and evaluated FIRST
                if height.get(a, 0) >= height.get(b, 0):
                    if b not in emitted:
                        stk.append((b, 0))
                    if a not in emitted:
                        stk.append((a, 0))
                else:
                    if a not in emitted:
                        stk.append((a, 0))
                    if b not in emitted:
                        stk.append((b, 0))
            else:
                if v not in emitted:
                    emitted.add(v)
                    schedule.append(v)

    for t in target_order:
        emit(t)
    return schedule


def _target_ancestors(
    t: int, seeds: set[int], defs: dict[int, tuple[str, int, int]]
) -> set[int]:
    anc: set[int] = set()
    stk = [t]
    while stk:
        u = stk.pop()
        if u in seeds or u in anc:
            continue
        anc.add(u)
        _, a, b = defs[u]
        stk.append(a)
        stk.append(b)
    return anc


def _affinity_target_order(
    targets: list[int],
    seeds: set[int],
    defs: dict[int, tuple[str, int, int]],
    height: dict[int, int],
) -> list[int]:
    """
    Order targets greedily so consecutive targets share maximum common ancestors,
    reducing long-range liveness across unrelated targets.
    """
    if len(targets) <= 2:
        return sorted(targets, key=lambda x: (height.get(x, 0), x))

    anc_map = {t: _target_ancestors(t, seeds, defs) for t in targets}
    # Start with smallest subtree
    unscheduled = set(targets)
    covered: set[int] = set()
    order: list[int] = []

    first = min(unscheduled, key=lambda t: (len(anc_map[t]), height.get(t, 0), t))
    order.append(first)
    unscheduled.remove(first)
    covered.update(anc_map[first])

    while unscheduled:
        # Pick target minimizing number of NEW nodes needed, tie-broken by overlap size & small height
        nxt = min(
            unscheduled,
            key=lambda t: (len(anc_map[t] - covered), -len(anc_map[t] & covered), height.get(t, 0), t),
        )
        order.append(nxt)
        unscheduled.remove(nxt)
        covered.update(anc_map[nxt])

    return order


def _allocate_schedule(
    puzzle: Puzzle,
    schedule: list[int],
    defs: dict[int, tuple[str, int, int]],
) -> list[Instruction]:
    """
    Allocate `schedule` onto `puzzle.registers` registers cleanly.
    """
    seeds = set(puzzle.seeds)
    n_regs = puzzle.registers

    # Compute last direct consumer index for every value
    last_use: dict[int, int] = {}
    for idx, v in enumerate(schedule):
        _, a, b = defs[v]
        last_use[a] = idx
        last_use[b] = idx

    # Compute seed transitive liveness (seed_last_needed[s]):
    # A seed is safe to overwrite ONLY after every scheduled node that transitively
    # depends on it has finished (in case an intermediate ever rematerializes).
    seed_deps: dict[int, frozenset[int]] = {s: frozenset([s]) for s in seeds}
    for v in schedule:
        _, a, b = defs[v]
        seed_deps[v] = seed_deps.get(a, frozenset()) | seed_deps.get(b, frozenset())

    seed_last_needed: dict[int, int] = {}
    for idx, v in enumerate(schedule):
        for s in seed_deps[v]:
            seed_last_needed[s] = idx

    # Map register assignments (first handle duplicate seeds cleanly)
    val_to_reg: dict[int, int] = {}
    reg_to_val: dict[int, int] = {}
    free_regs: list[int] = []
    for i, s in enumerate(puzzle.seeds):
        if s not in val_to_reg:
            val_to_reg[s] = i
            reg_to_val[i] = s
        else:
            free_regs.append(i)
    for r in range(n_regs - 1, len(puzzle.seeds) - 1, -1):
        free_regs.append(r)

    instructions: list[Instruction] = []

    def is_dead(lv: int, step_idx: int, protect: set[int]) -> bool:
        if lv in protect:
            return False
        if lv in seeds:
            return seed_last_needed.get(lv, -1) < step_idx
        return last_use.get(lv, -1) < step_idx

    def can_remat(lv: int, live_vals: set[int], vis: set[int]) -> bool:
        if lv in live_vals:
            return True
        if lv in seeds or lv in vis:
            return False
        vis.add(lv)
        _, pa, pb = defs[lv]
        return can_remat(pa, live_vals, vis) and can_remat(pb, live_vals, vis)

    def remat_cost_live(v: int, live_set: set[int], vis: set[int]) -> int:
        if v in live_set or v in vis:
            return 0
        if v in seeds:
            return 999
        vis.add(v)
        _, pa, pb = defs[v]
        return 1 + remat_cost_live(pa, live_set, vis) + remat_cost_live(pb, live_set, vis)

    def alloc_reg(step_idx: int, protect: set[int]) -> int:
        for lv in list(val_to_reg.keys()):
            if is_dead(lv, step_idx, protect):
                r = val_to_reg.pop(lv)
                del reg_to_val[r]
                free_regs.append(r)
        if free_regs:
            return free_regs.pop()

        # Must evict one live rematerializable value
        live_vals = set(val_to_reg.keys())
        best_lv = None
        best_score = -1.0
        for lv in val_to_reg:
            if lv in protect:
                continue
            if lv in seeds and seed_last_needed.get(lv, -1) >= step_idx:
                continue
            # Ensure lv is actually rematerializable from remaining live values (excluding lv)
            if not can_remat(lv, live_vals - {lv}, set()):
                continue
            nxt = 10**9
            for j in range(step_idx + 1, len(schedule)):
                _, ua, ub = defs[schedule[j]]
                if ua == lv or ub == lv:
                    nxt = j
                    break
            rc = remat_cost_live(lv, live_vals - {lv}, set())
            sc = (nxt - step_idx) / max(1, rc)
            if sc > best_score:
                best_score = sc
                best_lv = lv

        if best_lv is None:
            raise RuntimeError("Register allocator exhausted: all 48 registers live & non-rematerializable")
        r = val_to_reg.pop(best_lv)
        del reg_to_val[r]
        return r

    def ensure_live(v: int, step_idx: int, protect: set[int]) -> int:
        if v in val_to_reg:
            return val_to_reg[v]
        if v in seeds:
            raise RuntimeError(f"Seed {v} missing when needed at step {step_idx}")
        op, a, b = defs[v]
        ensure_live(a, step_idx, protect | {b})
        ensure_live(b, step_idx, protect | {a})
        ra = val_to_reg[a]
        rb = val_to_reg[b]
        dst = alloc_reg(step_idx, protect | {a, b})
        instructions.append(Instruction(op=op, src1=ra, src2=rb, dst=dst))
        if dst in reg_to_val and reg_to_val[dst] != v:
            del val_to_reg[reg_to_val[dst]]
        val_to_reg[v] = dst
        reg_to_val[dst] = v
        return dst

    for idx, v in enumerate(schedule):
        op, a, b = defs[v]
        for lv in list(val_to_reg.keys()):
            if is_dead(lv, idx, {a, b}):
                r = val_to_reg.pop(lv)
                del reg_to_val[r]
                free_regs.append(r)

        ensure_live(a, idx, {b})
        ensure_live(b, idx, {a})
        ra = val_to_reg[a]
        rb = val_to_reg[b]

        # In-place overwrite if operand dies at this exact step (and isn't needed for seed deps)
        dst = None
        if a != v and (a not in seeds or seed_last_needed.get(a, -1) <= idx) and last_use.get(a, -1) <= idx:
            dst = ra
            del val_to_reg[a]
            del reg_to_val[ra]
        elif b != v and (b not in seeds or seed_last_needed.get(b, -1) <= idx) and last_use.get(b, -1) <= idx:
            dst = rb
            del val_to_reg[b]
            del reg_to_val[rb]
        else:
            dst = alloc_reg(idx, {a, b})

        val_to_reg[v] = dst
        reg_to_val[dst] = v
        instructions.append(Instruction(op=op, src1=ra, src2=rb, dst=dst))

    return instructions


def compile_dag(
    puzzle: Puzzle, defs: dict[int, tuple[str, int, int]]
) -> list[Instruction]:
    """
    Compile an acyclic value DAG `defs` covering all `puzzle.targets` into
    valid 48-register `Instruction`s, picking the minimum-spill topological schedule.
    """
    seeds = set(puzzle.seeds)
    targets = sorted(set(puzzle.targets) - seeds)
    if not targets:
        return []

    needed = _extract_needed_subgraph(puzzle, defs)
    if not needed:
        return []
    height = _compute_heights(needed, seeds, defs)

    target_orders = [
        sorted(targets, key=lambda x: (height.get(x, 0), x)),
        _affinity_target_order(targets, seeds, defs, height),
        sorted(targets, key=lambda x: (x.bit_length(), x)),
        sorted(targets, key=lambda x: (-height.get(x, 0), x)),
    ]

    best_prog: list[Instruction] | None = None
    seen_scheds: set[tuple[int, ...]] = set()
    for order in target_orders:
        sched = _schedule_dfs(order, seeds, defs, height)
        key = tuple(sched)
        if key in seen_scheds:
            continue
        seen_scheds.add(key)
        try:
            prog = _allocate_schedule(puzzle, sched, defs)
            if best_prog is None or len(prog) < len(best_prog):
                best_prog = prog
                # Zero-spill optimal reached! No need to try further schedules
                if len(best_prog) == len(needed):
                    break
        except RuntimeError:
            continue

    if best_prog is None:
        raise RuntimeError(f"Failed to compile DAG for puzzle {puzzle.id} in {puzzle.registers} regs")
    return best_prog

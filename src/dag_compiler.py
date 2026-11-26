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
import random
import time
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


def _compute_ershov(
    needed: set[int], seeds: set[int], defs: dict[int, tuple[str, int, int]]
) -> dict[int, int]:
    """Compute exact Sethi-Ullman Ershov register pressure numbers for DAG nodes."""
    er: dict[int, int] = {s: 1 for s in seeds}
    post: list[int] = []
    seen: set[int] = set(seeds)
    for root in needed:
        if root in seen:
            continue
        stk: list[tuple[int, int]] = [(root, 0)]
        while stk:
            v, ph = stk.pop()
            if v in seen:
                continue
            if ph == 0:
                stk.append((v, 1))
                _, a, b = defs[v]
                if a not in seen:
                    stk.append((a, 0))
                if b not in seen and b != a:
                    stk.append((b, 0))
            else:
                if v not in seen:
                    seen.add(v)
                    post.append(v)
    for v in post:
        _, a, b = defs[v]
        ea, eb = er.get(a, 1), er.get(b, 1)
        if a == b:
            er[v] = max(2, ea)
        elif ea == eb:
            er[v] = ea + 1
        else:
            er[v] = ea if ea > eb else eb
    return er


def _schedule_dfs(
    target_order: list[int],
    seeds: set[int],
    defs: dict[int, tuple[str, int, int]],
    metric: dict[int, int],
    *,
    lighter_first_on_tie: bool = False,
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
                ha, hb = metric.get(a, 0), metric.get(b, 0)
                eval_a_first = (ha > hb) if lighter_first_on_tie else (ha >= hb)
                # Push second child FIRST so first-to-eval child is popped and evaluated FIRST
                if eval_a_first:
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


def _mincut_target_order(
    targets: list[int],
    seeds: set[int],
    defs: dict[int, tuple[str, int, int]],
    height: dict[int, int],
) -> list[int]:
    """
    Order targets to minimize peak live register cut: greedily pick the next target
    maximizing (shared live nodes freed minus new unmaterialized nodes opened).
    """
    if len(targets) <= 2:
        return sorted(targets, key=lambda x: (height.get(x, 0), x))

    anc_map = {t: _target_ancestors(t, seeds, defs) for t in targets}
    rem_refs: dict[int, int] = defaultdict(int)
    for t in targets:
        for u in anc_map[t]:
            rem_refs[u] += 1

    unscheduled = set(targets)
    covered: set[int] = set()
    order: list[int] = []

    first = min(unscheduled, key=lambda t: (len(anc_map[t]), height.get(t, 0), t))
    order.append(first)
    unscheduled.remove(first)
    covered.update(anc_map[first])
    for u in anc_map[first]:
        rem_refs[u] -= 1

    while unscheduled:
        best_t = min(
            unscheduled,
            key=lambda t: (
                len(anc_map[t] - covered) - sum(1 for u in anc_map[t] if rem_refs[u] == 1),
                len(anc_map[t] - covered),
                height.get(t, 0),
                t,
            ),
        )
        order.append(best_t)
        unscheduled.remove(best_t)
        covered.update(anc_map[best_t])
        for u in anc_map[best_t]:
            rem_refs[u] -= 1

    return order


def _hub_cluster_target_order(
    targets: list[int],
    seeds: set[int],
    defs: dict[int, tuple[str, int, int]],
    ershov: dict[int, int],
) -> list[int]:
    """
    Order targets to evaluate target-to-target dependencies right before their
    consumers and keep heavy shared hubs tight in cache before eviction.
    """
    if len(targets) <= 2:
        return sorted(targets, key=lambda x: (ershov.get(x, 1), x))

    anc_map = {t: _target_ancestors(t, seeds, defs) for t in targets}
    rem_refs: dict[int, int] = defaultdict(int)
    for t in targets:
        for u in anc_map[t]:
            rem_refs[u] += 1

    # Weight nodes by Ershov/reuse so freeing a high-cost shared node ranks highest
    unscheduled = set(targets)
    covered: set[int] = set()
    order: list[int] = []

    first = min(unscheduled, key=lambda t: (ershov.get(t, 1), len(anc_map[t]), t))
    order.append(first)
    unscheduled.remove(first)
    covered.update(anc_map[first])
    for u in anc_map[first]:
        rem_refs[u] -= 1

    while unscheduled:
        best_t = min(
            unscheduled,
            key=lambda t: (
                len(anc_map[t] - covered) - sum(2 for u in (anc_map[t] & covered) if rem_refs[u] == 1),
                ershov.get(t, 1),
                len(anc_map[t] - covered),
                t,
            ),
        )
        order.append(best_t)
        unscheduled.remove(best_t)
        covered.update(anc_map[best_t])
        for u in anc_map[best_t]:
            rem_refs[u] -= 1
    return order


def _allocate_schedule(
    puzzle: Puzzle,
    schedule: list[int],
    defs: dict[int, tuple[str, int, int]],
    *,
    soft_remat: bool = False,
) -> list[Instruction]:
    """
    Allocate `schedule` onto `puzzle.registers` registers cleanly.
    """
    seeds = set(puzzle.seeds)
    n_regs = puzzle.registers

    # Compute last direct consumer index and all future consumer positions
    import bisect
    last_use: dict[int, int] = {}
    use_positions: dict[int, list[int]] = defaultdict(list)
    for idx, v in enumerate(schedule):
        _, a, b = defs[v]
        last_use[a] = idx
        if b != a:
            last_use[b] = idx
        use_positions[a].append(idx)
        if b != a:
            use_positions[b].append(idx)

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
            up = use_positions.get(lv, [])
            p0 = bisect.bisect_right(up, step_idx)
            rem_uses = max(1, len(up) - p0)
            if p0 < len(up):
                nxt = up[p0]
            rc = remat_cost_live(lv, live_vals - {lv}, set())
            denom = max(1.0, rc * (1.0 + 0.15 * (rem_uses - 1)) if soft_remat else rc * rem_uses)
            sc = (nxt - step_idx) / denom
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
        dst = None
        if a != v and is_dead(a, step_idx, protect | {b}):
            dst = ra
            del val_to_reg[a]
            del reg_to_val[ra]
        elif b != v and is_dead(b, step_idx, protect | {a}):
            dst = rb
            del val_to_reg[b]
            del reg_to_val[rb]
        else:
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


def _schedule_min_live(
    needed: set[int],
    defs: dict[int, tuple[str, int, int]],
    height: dict[int, int],
) -> list[int]:
    """
    Topological Kahn list scheduler minimizing live register pressure:
    picks the ready node producing the steepest reduction in live registers
    (freeing operands at their last consumer).
    """
    deps_rem: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    out_rem: dict[int, int] = defaultdict(int)

    for v in needed:
        _, a, b = defs[v]
        need_p = 0
        if a in needed:
            need_p += 1
            children[a].append(v)
            out_rem[a] += 1
        if b in needed and b != a:
            need_p += 1
            children[b].append(v)
            out_rem[b] += 1
        deps_rem[v] = need_p

    ready = {v for v, c in deps_rem.items() if c == 0}
    sched: list[int] = []
    while ready:
        best_v = min(
            ready,
            key=lambda v: (
                1
                - (1 if defs[v][1] in needed and out_rem[defs[v][1]] == 1 else 0)
                - (1 if defs[v][2] in needed and defs[v][2] != defs[v][1] and out_rem[defs[v][2]] == 1 else 0),
                out_rem.get(v, 0),
                height.get(v, 0),
                v,
            ),
        )
        ready.remove(best_v)
        sched.append(best_v)
        _, a, b = defs[best_v]
        if a in needed:
            out_rem[a] -= 1
        if b in needed and b != a:
            out_rem[b] -= 1
        for ch in children.get(best_v, ()):
            deps_rem[ch] -= 1
            if deps_rem[ch] == 0:
                ready.add(ch)
    return sched


def _emergency_compile(
    puzzle: Puzzle,
    defs: dict[int, tuple[str, int, int]],
    height: dict[int, int],
) -> list[Instruction]:
    """
    Crash-proof low-pressure Sethi-Ullman subtree compiler fallback.
    Builds targets one-by-one holding only `seeds` + an active evaluation stack
    so any valid acyclic DAG with <= (registers - 4) seeds always compiles cleanly.
    """
    seeds = set(puzzle.seeds)
    n_regs = puzzle.registers
    targets = sorted(set(puzzle.targets) - seeds, key=lambda x: (height.get(x, 0), x))

    val_to_reg: dict[int, int] = {}
    reg_to_val: dict[int, int] = {}
    seed_regs: set[int] = set()
    for i, s in enumerate(puzzle.seeds):
        if s not in val_to_reg:
            val_to_reg[s] = i
            reg_to_val[i] = s
            seed_regs.add(i)

    def _free_reg(protected_regs: set[int]) -> int:
        for r in range(n_regs - 1, -1, -1):
            if r not in reg_to_val and r not in protected_regs and r not in seed_regs:
                return r
        for r in range(n_regs - 1, -1, -1):
            if r not in protected_regs and r not in seed_regs:
                old_v = reg_to_val.pop(r)
                val_to_reg.pop(old_v, None)
                return r
        raise RuntimeError(f"Too many pinned seed registers ({len(seed_regs)}) for {n_regs} regs")

    instructions: list[Instruction] = []
    tree_sz: dict[int, int] = {}

    def tsize(u: int, vis: set[int]) -> int:
        if u in seeds or u in val_to_reg:
            return 0
        if u in tree_sz:
            return tree_sz[u]
        if u in vis:
            return 1
        vis.add(u)
        _, a, b = defs[u]
        sz = 1 + tsize(a, vis) + (tsize(b, vis) if b != a else 0)
        tree_sz[u] = sz
        return sz

    def eval_node(u: int, protected_regs: set[int]) -> int:
        if u in val_to_reg:
            return val_to_reg[u]
        op, a, b = defs[u]
        if a == b:
            ra = eval_node(a, protected_regs)
            dst = _free_reg(protected_regs | {ra})
            val_to_reg[u] = dst
            reg_to_val[dst] = u
            instructions.append(Instruction(op=op, src1=ra, src2=ra, dst=dst))
            return dst
        sa = tsize(a, set())
        sb = tsize(b, set())
        if sa >= sb:
            ra = eval_node(a, protected_regs)
            rb = eval_node(b, protected_regs | {ra})
            ra = val_to_reg[a] if a in val_to_reg else eval_node(a, protected_regs | {rb})
        else:
            rb = eval_node(b, protected_regs)
            ra = eval_node(a, protected_regs | {rb})
            rb = val_to_reg[b] if b in val_to_reg else eval_node(b, protected_regs | {ra})
        dst = _free_reg(protected_regs | {ra, rb})
        val_to_reg[u] = dst
        reg_to_val[dst] = u
        instructions.append(Instruction(op=op, src1=ra, src2=rb, dst=dst))
        return dst

    for t in targets:
        eval_node(t, set())
    return instructions


def compile_dag(
    puzzle: Puzzle, defs: dict[int, tuple[str, int, int]], *, n_trials: int = 48
) -> list[Instruction]:
    """
    Compile an acyclic value DAG `defs` covering all `puzzle.targets` into
    valid 48-register `Instruction`s, minimizing register spills via Min-Cut,
    Sethi-Ullman / Subtree-Affinity ordering, min-live Kahn scheduling, and
    local permutation hill-climbing.
    """
    seeds = set(puzzle.seeds)
    targets = sorted(set(puzzle.targets) - seeds)
    if not targets:
        return []

    needed = _extract_needed_subgraph(puzzle, defs)
    if not needed:
        return []
    height = _compute_heights(needed, seeds, defs)
    ershov = _compute_ershov(needed, seeds, defs)

    target_orders = [
        _mincut_target_order(targets, seeds, defs, height),
        _hub_cluster_target_order(targets, seeds, defs, ershov),
        _affinity_target_order(targets, seeds, defs, height),
        sorted(targets, key=lambda x: (ershov.get(x, 1), height.get(x, 0), x)),
        sorted(targets, key=lambda x: (height.get(x, 0), x)),
        sorted(targets, key=lambda x: (x.bit_length(), x)),
        sorted(targets, key=lambda x: (-height.get(x, 0), x)),
    ]

    best_prog: list[Instruction] | None = None
    best_order: list[int] = target_orders[0]
    best_metric: dict[int, int] = ershov
    best_sched: list[int] | None = None
    top_scheds: list[list[int]] = []
    seen_scheds: set[tuple[int, ...]] = set()
    optimal_len = len(needed)

    def try_schedule(
        sched: list[int], order_ref: list[int] | None = None, metric_ref: dict[int, int] | None = None
    ) -> bool:
        nonlocal best_prog, best_order, best_metric, best_sched
        key = tuple(sched)
        if key in seen_scheds:
            return False
        seen_scheds.add(key)
        try:
            prog = _allocate_schedule(puzzle, sched, defs, soft_remat=False)
            if best_prog is None or len(prog) < len(best_prog):
                best_prog = prog
                best_sched = sched
                top_scheds.append(sched)
                if order_ref is not None:
                    best_order = list(order_ref)
                if metric_ref is not None:
                    best_metric = metric_ref
                return True
        except RuntimeError:
            pass
        return False

    def try_order(
        order: list[int], flip_tie: bool = False, metric: dict[int, int] | None = None
    ) -> bool:
        m = metric if metric is not None else height
        sched = _schedule_dfs(order, seeds, defs, m, lighter_first_on_tie=flip_tie)
        return try_schedule(sched, order, m)

    try_schedule(_schedule_min_live(needed, defs, height))
    if best_prog is None or len(best_prog) > optimal_len:
        for order in target_orders:
            for m_eval in (ershov, height):
                try_order(order, flip_tie=False, metric=m_eval)
                if best_prog is not None and len(best_prog) == optimal_len:
                    break
                try_order(order, flip_tie=True, metric=m_eval)
                if best_prog is not None and len(best_prog) == optimal_len:
                    break
            if best_prog is not None and len(best_prog) == optimal_len:
                break

    # Permutation hill-climbing if any register spill remains
    if best_prog is not None and len(best_prog) > optimal_len and len(targets) > 2 and n_trials > 0:
        rng = random.Random(42)
        t_start = time.monotonic()
        cur_order = list(best_order)
        for trial in range(n_trials):
            if (time.monotonic() - t_start) > 1.10:
                break
            cand_order = list(cur_order)
            if trial % 4 == 0:
                i = rng.randrange(len(cand_order) - 1)
                cand_order[i], cand_order[i + 1] = cand_order[i + 1], cand_order[i]
            elif trial % 4 == 1:
                i, j = rng.sample(range(len(cand_order)), 2)
                cand_order[i], cand_order[j] = cand_order[j], cand_order[i]
            elif trial % 4 == 2:
                i = rng.randrange(len(cand_order))
                j = rng.randrange(len(cand_order))
                val = cand_order.pop(i)
                cand_order.insert(j, val)
            else:
                # Move a contiguous 2-3 target cluster together
                sz = min(3, max(2, len(cand_order) // 6))
                if len(cand_order) > sz + 1:
                    i = rng.randrange(len(cand_order) - sz + 1)
                    blk = cand_order[i : i + sz]
                    del cand_order[i : i + sz]
                    j = rng.randrange(len(cand_order) + 1)
                    cand_order[j:j] = blk

            if try_order(cand_order, flip_tie=(trial % 2 == 1), metric=best_metric):
                cur_order = list(cand_order)
                if len(best_prog) == optimal_len:
                    break

    if best_prog is not None and len(best_prog) > optimal_len:
        for sc_try in reversed(top_scheds[-2:]):
            try:
                p_sr = _allocate_schedule(puzzle, sc_try, defs, soft_remat=True)
                if len(p_sr) < len(best_prog):
                    best_prog = p_sr
            except RuntimeError:
                pass

    if best_prog is None:
        return _emergency_compile(puzzle, defs, height)
    return best_prog

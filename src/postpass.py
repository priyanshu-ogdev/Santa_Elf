"""
src/postpass.py
Post-processing passes applied after any solver produces a valid program.

Passes (applied in order):
  1. remove_dead_ops  — eliminates instructions whose output is never read
                        and doesn't hit a target. Free improvement, O(n).
  2. peephole        — 2–3 instruction window local rewrites. O(n).
  3. simulated_annealing — probabilistic local search over the instruction
                        sequence. The verifier is the oracle. Time-bounded.

Each pass takes a Puzzle + list[Instruction] → list[Instruction].
All passes produce programs that pass verify_program().
"""
from __future__ import annotations
import random
import time
from src.utils import Puzzle, Instruction, apply_op, VALID_OPS
from src.verifier import verify_program


# ---------------------------------------------------------------------------
# Pass 1: Dead Instruction Elimination
# ---------------------------------------------------------------------------

def remove_dead_ops(puzzle: Puzzle, program: list[Instruction]) -> list[Instruction]:
    """
    Remove instructions whose destination register is overwritten before it is
    ever read AND whose result value never satisfies a target.

    Algorithm (single backward scan):
      - Track which registers are "needed" at each point.
      - An instruction is live if its dst register is in the needed set at
        that point OR if its result value is in puzzle.targets.
      - After scanning backwards, re-filter to keep only live instructions.
    """
    if not program:
        return program

    n = len(program)
    targets = set(puzzle.targets)
    seeds = set(puzzle.seeds)

    # Forward simulation: track register values so we know what each
    # instruction produces (needed to check "result in targets").
    regs: list[int | None] = [None] * puzzle.registers
    for i, v in enumerate(puzzle.seeds):
        regs[i] = v

    result_values: list[int | None] = []
    for inst in program:
        a = regs[inst.src1]
        b = regs[inst.src2]
        r = apply_op(inst.op, a, b, puzzle.value_limit) if (a is not None and b is not None) else None
        result_values.append(r)
        if r is not None:
            regs[inst.dst] = r

    # Backward scan: which registers are needed AFTER step i?
    # A register is needed if it will be used as src1/src2 in a future instruction.
    needed: set[int] = set()

    # At the end, no register is needed (targets are recorded during simulation)
    live = [False] * n

    for i in range(n - 1, -1, -1):
        inst = program[i]
        result_val = result_values[i]

        # This instruction is live if:
        # (a) its dst is currently needed by a later instruction, OR
        # (b) its result is a target value (must have been produced)
        is_target_hit = result_val is not None and result_val in targets
        if inst.dst in needed or is_target_hit:
            live[i] = True
            # This instruction reads src1 and src2, so they become needed
            needed.discard(inst.dst)  # dst is now "satisfied" by this inst
            needed.add(inst.src1)
            needed.add(inst.src2)
        # If not live: dst is overwritten (or never read), skip

    return [program[i] for i in range(n) if live[i]]


# ---------------------------------------------------------------------------
# Pass 2: Peephole Rewrites
# ---------------------------------------------------------------------------

def _live_shortcut_pass(puzzle: Puzzle, program: list[Instruction]) -> list[Instruction]:
    """
    Deterministic peephole pass:
      1. If step `i` computes a value `res` ALREADY present in another live register
         `r_old != dst` (and `res` isn't a not-yet-hit target), or even if a target
         was ALREADY hit earlier, alias subsequent reads of `dst` (before `dst` is next
         overwritten) to `r_old` if safe, eliminating step `i` via `remove_dead_ops`.
      2. Always re-verifies with `verify_program` before accepting any edit.
    """
    if len(program) <= 1:
        return program

    cur = list(program)
    n = len(cur)
    regs: list[int | None] = [None] * puzzle.registers
    for i, s in enumerate(puzzle.seeds):
        regs[i] = s
    targets_hit = set(puzzle.seeds) & set(puzzle.targets)

    changed = False
    for i in range(n):
        inst = cur[i]
        a = regs[inst.src1]
        b = regs[inst.src2]
        if a is None or b is None:
            break
        res = apply_op(inst.op, a, b, puzzle.value_limit)
        if res is None:
            break

        # Is `res` ALREADY sitting live in another register right now?
        need_target = (res in puzzle.targets) and (res not in targets_hit)
        if not need_target:
            other_reg = None
            for r_idx, r_val in enumerate(regs):
                if r_idx != inst.dst and r_val == res:
                    other_reg = r_idx
                    break
            if other_reg is not None:
                # Rebind future reads of `inst.dst` to `other_reg` until `inst.dst` OR `other_reg` is overwritten
                cand = list(cur)
                for j in range(i + 1, n):
                    jn = cand[j]
                    ns1 = other_reg if jn.src1 == inst.dst else jn.src1
                    ns2 = other_reg if jn.src2 == inst.dst else jn.src2
                    if ns1 != jn.src1 or ns2 != jn.src2:
                        cand[j] = Instruction(jn.op, ns1, ns2, jn.dst)
                    if cand[j].dst == inst.dst or cand[j].dst == other_reg:
                        break
                cand_clean = remove_dead_ops(puzzle, cand)
                if len(cand_clean) < len(cur) and verify_program(puzzle, cand_clean).valid:
                    return _live_shortcut_pass(puzzle, cand_clean)

        regs[inst.dst] = res
        if res in puzzle.targets:
            targets_hit.add(res)

    return cur


def _peephole_algebra_pass(puzzle: Puzzle, program: list[Instruction]) -> list[Instruction]:
    """
    Collapse adjacent constant/algebraic adjustment chains across live registers,
    e.g., `(x + c1) + c2 -> x + c3` when `c3 = c1 + c2` (or inverse ops) already live.
    """
    if len(program) <= 2:
        return program

    cur = list(program)
    n = len(cur)
    targets = set(puzzle.targets)
    regs: list[int | None] = [None] * puzzle.registers
    for i, s in enumerate(puzzle.seeds):
        regs[i] = s
    targets_hit = set(puzzle.seeds) & targets

    for i in range(n - 1):
        ins_i = cur[i]
        a = regs[ins_i.src1]
        b = regs[ins_i.src2]
        if a is None or b is None:
            break
        res_i = apply_op(ins_i.op, a, b, puzzle.value_limit)
        if res_i is None:
            break

        need_target = (res_i in targets) and (res_i not in targets_hit)
        # Simulate regs right after instruction i
        regs_after = list(regs)
        regs_after[ins_i.dst] = res_i
        if not need_target and ins_i.src1 != ins_i.dst and ins_i.src2 != ins_i.dst:
            val_to_reg: dict[int, int] = {}
            for r_idx, rv in enumerate(regs_after):
                if rv is not None and r_idx != ins_i.dst:
                    val_to_reg[rv] = r_idx

            # Check immediate next few instructions using ins_i.dst while src1/src2 remain intact
            r_track = list(regs_after)
            for j in range(i + 1, min(n, i + 8)):
                ins_j = cur[j]
                va = r_track[ins_j.src1]
                vb = r_track[ins_j.src2]
                if va is None or vb is None:
                    break
                res_j = apply_op(ins_j.op, va, vb, puzzle.value_limit)
                if res_j is None:
                    break

                if ins_j.src1 == ins_i.dst or ins_j.src2 == ins_i.dst:
                    other_r = ins_j.src2 if ins_j.src1 == ins_i.dst else ins_j.src1
                    other_v = vb if ins_j.src1 == ins_i.dst else va
                    # Case 1: direct identity shortcut `(a op c1) op2 c2 == a` (or another live reg!)
                    if r_track[ins_i.src1] == a and res_j == a and ins_i.src1 != ins_i.dst:
                        # Replace j with identity? Only if future reads of j's dst can use src1
                        cand = list(cur)
                        src_live = ins_i.src1
                        ok_alias = True
                        for k in range(j + 1, n):
                            kn = cand[k]
                            ns1 = src_live if kn.src1 == ins_j.dst else kn.src1
                            ns2 = src_live if kn.src2 == ins_j.dst else kn.src2
                            if ns1 != kn.src1 or ns2 != kn.src2:
                                cand[k] = Instruction(kn.op, ns1, ns2, kn.dst)
                            if cand[k].dst == ins_j.dst or cand[k].dst == src_live:
                                break
                        if ok_alias:
                            cc = remove_dead_ops(puzzle, cand)
                            if len(cc) < len(cur) and verify_program(puzzle, cc).valid:
                                return _peephole_algebra_pass(puzzle, cc)

                    # Case 2: `(a + b) + c2` or `(a - b) + c2` where `b +- c2` is already live
                    if r_track[ins_i.src1] == a and ins_i.op in ("+", "-") and ins_j.op in ("+", "-") and ins_j.src1 == ins_i.dst:
                        delta = (b if ins_i.op == "+" else -b) + (other_v if ins_j.op == "+" else -other_v)
                        if delta != 0 and abs(delta) in val_to_reg and r_track[val_to_reg[abs(delta)]] == abs(delta):
                            op_new = "+" if delta > 0 else "-"
                            cand = list(cur)
                            cand[j] = Instruction(op_new, ins_i.src1, val_to_reg[abs(delta)], ins_j.dst)
                            cc = remove_dead_ops(puzzle, cand)
                            if len(cc) < len(cur) and verify_program(puzzle, cc).valid:
                                return _peephole_algebra_pass(puzzle, cc)

                    # Case 3: `(a * b) * c2` where `b * c2` is already live
                    if r_track[ins_i.src1] == a and ins_i.op == "*" and ins_j.op == "*":
                        prod_bc = b * other_v
                        if prod_bc in val_to_reg and r_track[val_to_reg[prod_bc]] == prod_bc:
                            cand = list(cur)
                            cand[j] = Instruction("*", ins_i.src1, val_to_reg[prod_bc], ins_j.dst)
                            cc = remove_dead_ops(puzzle, cand)
                            if len(cc) < len(cur) and verify_program(puzzle, cc).valid:
                                return _peephole_algebra_pass(puzzle, cc)

                r_track[ins_j.dst] = res_j
                if ins_j.dst == ins_i.dst or ins_j.dst == ins_i.src1 or ins_j.dst == ins_i.src2:
                    break

        regs[ins_i.dst] = res_i
        if res_i in targets:
            targets_hit.add(res_i)

    return cur


def peephole(puzzle: Puzzle, program: list[Instruction]) -> list[Instruction]:
    """
    Apply local rewrites:
      1. Dead-op elimination (via remove_dead_ops).
      2. Live-register value forwarding & algebraic chain collapsing.
    """
    result = remove_dead_ops(puzzle, program)
    result = _live_shortcut_pass(puzzle, result)
    result = _peephole_algebra_pass(puzzle, result)
    return result


# ---------------------------------------------------------------------------
# Pass 3: Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    puzzle: Puzzle,
    program: list[Instruction],
    *,
    time_budget: float = 2.0,
    initial_temp: float = 2.0,
    final_temp: float = 0.01,
    random_seed: int | None = None,
) -> list[Instruction]:
    """
    Probabilistic local search over the instruction sequence.

    Move types (weighted):
      1. TryRemove (60%): remove one random instruction; accept if still valid.
      2. TryRemoveBlock (20%): remove 2-3 consecutive instructions; accept if valid.
      3. TrySwap (10%): swap two instructions.
      4. TryReplaceOp (10%): change operator of one instruction.

    Returns the best program found within the time budget.
    """
    rng = random.Random(random_seed)
    best = list(program)
    best_count = len(best)

    if best_count == 0:
        return best

    current = list(best)
    current_count = best_count

    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= time_budget:
            break

        # Geometric cooling
        frac = elapsed / time_budget
        temp = initial_temp * ((final_temp / initial_temp) ** frac)

        # --- Pick a random move ---
        r = rng.random()
        if r < 0.60 and current_count > 1:
            # TryRemove: remove one instruction
            idx = rng.randrange(current_count)
            candidate = current[:idx] + current[idx + 1:]
        elif r < 0.80 and current_count > 3:
            # TryRemoveBlock: remove 2-4 consecutive instructions
            block = rng.randint(2, min(4, current_count - 1))
            idx = rng.randrange(current_count - block)
            candidate = current[:idx] + current[idx + block:]
        elif r < 0.90 and current_count > 2:
            # TrySwap
            i, j = sorted(rng.sample(range(current_count), 2))
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif current_count > 0:
            # TryReplaceOp
            idx = rng.randrange(current_count)
            new_op = rng.choice(["+", "-", "*", "/"])
            inst = current[idx]
            candidate = list(current)
            candidate[idx] = Instruction(op=new_op, src1=inst.src1,
                                         src2=inst.src2, dst=inst.dst)
        else:
            continue

        # Validate candidate
        result = verify_program(puzzle, candidate)
        if result.valid:
            delta = len(candidate) - current_count
            if delta < 0:
                # Always accept improvements
                current = candidate
                current_count = len(candidate)
                if current_count < best_count:
                    best = candidate[:]
                    best_count = current_count
            elif delta == 0:
                # Accept neutral moves unconditionally (exploration)
                current = candidate
            # delta > 0: reject

    return best


# ---------------------------------------------------------------------------
# Combined post-pass pipeline
# ---------------------------------------------------------------------------

def postprocess(
    puzzle: Puzzle,
    program: list[Instruction],
    *,
    run_annealing: bool = True,
    annealing_budget: float = 2.0,
    random_seed: int | None = None,
) -> list[Instruction]:
    """
    Run the full post-processing pipeline on a valid program.
    Returns the (hopefully shorter) improved program.
    """
    # Pass 1: dead-op elimination
    program = remove_dead_ops(puzzle, program)

    # Pass 2: peephole
    program = peephole(puzzle, program)

    # Pass 3: simulated annealing (optional, time-bounded)
    if run_annealing and len(program) > 0:
        program = simulated_annealing(
            puzzle, program,
            time_budget=annealing_budget,
            random_seed=random_seed,
        )
        # Final dead-op pass after annealing
        program = remove_dead_ops(puzzle, program)

    return program

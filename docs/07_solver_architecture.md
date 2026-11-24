# Solver Architecture: First Submission Design

Grounded in measured data from `puzzles.csv` and `practice_puzzles.csv`. No unverified SOTA claims.

---

## 1. Core Constraints Driving Architecture Decisions

Before any design, the measured data imposes hard architectural constraints:

| Constraint | Measured Value | Design Implication |
| :--- | :--- | :--- |
| Targets per puzzle | mean 26.90, median 30 | Per-target cost must be ≤ ~5 ops to beat 130k |
| Seed values | 1–1,998, $K$ ≤ 8 | Seeds are small; multiplication reaches ~10^6 in 2 ops from seeds |
| Target magnitudes | up to ~10^18 | Need ~60 bits; binary doubling needs ~60 ops alone — must use *, / |
| value_limit | as low as 7 | 16 puzzles require exact BFS; no overshoot allowed |
| Tight-limit puzzles | 2 puzzles: vl == max(target) | Zero headroom; cannot compute anything larger than the max target |
| Close adjacent targets | 3.79% of pairs | Delta-encoding from anchor reduces cost on those clusters |
| Smooth targets (lpf ≤ 2000) | 13.8% of scored sample | Factor-based synthesis worth attempting, not dominant |
| Arithmetic overflow | Products of two 10^18 values | Solver must use 128-bit arithmetic (Python `int` or Rust `i128`) |
| Register limit | 48 | Pool-first design; don't treat as a DAG packing afterthought |
| All 1,000 puzzles independent | confirmed | Trivially parallelizable across cores |

---

## 2. Pipeline Overview

Each puzzle goes through the same five-stage pipeline. Stages after **Dispatch** run in parallel across puzzles.

```
puzzles.csv
     │
     ▼
┌─────────────┐
│  Classifier │  Labels each puzzle: tiny | smooth | clustered | generic
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│              Solver Portfolio               │
│  ┌───────────┐  ┌─────────────┐  ┌───────┐ │
│  │ Tiny BFS  │  │ Factor/Delta│  │ M-i-M │ │
│  │ (exact)   │  │ Cluster     │  │ Core  │ │
│  └───────────┘  └─────────────┘  └───────┘ │
│                       + Fallback (always)   │
└──────────────────┬──────────────────────────┘
                   │ best program per solver
                   ▼
         ┌──────────────────┐
         │  Verifier        │  replays program, rejects if any fault
         └────────┬─────────┘
                  │ verified program
                  ▼
         ┌──────────────────┐
         │  Post-pass       │  dead-op removal → peephole → annealing
         └────────┬─────────┘
                  │ polished program
                  ▼
         ┌──────────────────┐
         │  Orchestrator    │  merges best across runs, writes CSV
         └──────────────────┘
```

---

## 3. Classifier (Stage 1)

Lightweight; runs in microseconds per puzzle.

**Input features:**
* `value_limit` (absolute and ratio to `max(targets)`)
* `len(seeds)`, `len(targets)`
* Fraction of targets with largest prime factor ≤ 2000 (smoothness)
* Fraction of adjacent sorted target pairs within 0.1% (clustering)
* Max `gcd(targets[i], targets[j])` over sampled pairs

**Label routing:**
* `tiny` → `value_limit < 100`, send to BFS exact solver.
* `smooth` → smoothness fraction > 0.4, send to factor-based solver.
* `clustered` → clustering fraction > 0.05 or high GCD, send to delta/cluster solver.
* `generic` → everything else, send to meet-in-the-middle core.
* All routes also run the fallback, keeping the minimum over results.

---

## 4. Solver Modules (Stage 2)

### 4a. Tiny Exact Solver (BFS / IDA*)

**When:** `value_limit` ≤ 100 (the 16 tiny scored puzzles; more in practice).

**Method:**
* State: frozenset of values currently in registers (order-independent for search).
* Expand: all pairwise `+`, `-`, `*`, exact `/` between current pool members.
* BFS finds optimal solution. For value_limit ≤ 100, the state space is tiny.
* Prune: values already exceeding value_limit, non-positive results.
* Special care for tight-limit puzzles: intermediate values must not exceed `vl`.

**Expected cost:** 1–6 instructions for all 16 tiny puzzles.

---

### 4b. Delta / Cluster Solver

**When:** Clustering fraction > 0.05, or the classifier identifies close adjacent targets.

**Method:**
1. Sort targets. Find clusters: runs of targets within 0.5% of each other.
2. Build an **anchor** for the cluster centroid using the meet-in-the-middle core (§4d).
3. Compute deltas: for each target near the anchor, `delta = target - anchor` (or `anchor - target`).
4. Build deltas cheaply using the pool: a small delta may already exist in the seed pool or cost 1 extra op from seeds.
5. Each cluster member then costs: 1 instruction (anchor ± delta), after anchor is built once.

**Expected saving:** 1–2 ops per clustered target pair vs. independent synthesis.

---

### 4c. Factor / Smooth Solver

**When:** Smoothness fraction > 0.4 (target's prime factors are all ≤ 2000).

**Method:**
1. For each smooth target, factor it completely.
2. Build a **factor basis**: the set of prime factors ≤ 2000, plus products of small pairs.
3. The factor basis is built from seeds using the pool (multiply/double/add).
4. Express each target as a product of basis elements: `t = a * b * c` → 2 instructions if `a*b` is already in pool.
5. For semi-smooth targets: `t = a * b ± c` where `a`, `b`, `c` are in pool.

**Cost model:** Smooth target ≈ 2–3 ops (basis build amortized across targets).

---

### 4d. Meet-in-the-Middle Core (Generic Solver)

**This is the workhorse for the majority of scored puzzles.**

**Concept:** The 48-register live pool is a workspace. Process targets in a chosen order, keeping solved targets and useful intermediates live as long as they benefit later targets.

#### Pool-based execution model

```
pool = {seed_0, seed_1, ..., seed_{K-1}}   # starts with seeds
instructions = []

for target in ordering(puzzle.targets):
    if target in pool:
        mark_hit(target); continue          # free if already in pool
    
    path = meet_in_middle(target, pool)     # find cheapest derivation
    for inst in path:
        execute(inst, pool, instructions)   # add intermediate values to pool
    
    evict(pool) if len(pool) > 48           # pool size management
```

#### Meet-in-the-middle search (per target)

```
Given target t, current pool P (|P| ≤ 48 values):

Forward set F:  all one-op results from P × P under {+, -, *, /}
                |F| ≤ 4 × 48 × 48 ≈ 9,000 values

Backward set B: for each b in F:
                  t - b,  b - t,  t * ? such that t/b is integer,  t + b
                look up each in P or F

Depth-1 hit:    t ∈ P  (cost 0)
Depth-2 hit:    t = p op q,  p,q ∈ P  (cost 1)
Depth-3 hit:    t = (p op q) op r,  p,q,r ∈ P  (cost 2, via F lookup)
Depth-4:        t = f op g,  f,g ∈ F  (cost 3, build two intermediates + combine)
```

For depth 3–4, build a hash map `F_map: value → (op, src1, src2)` once per target. The backward candidates from `t` are:
* `t - f` for all `f ∈ F` (look up in `P ∪ F`)
* `f - t` for all `f ∈ F` where `f > t`
* `t / f` for all `f ∈ F` where `f` divides `t`
* `t * f` for all `f ∈ F` only if `t * f ≤ value_limit` (then look up in `P ∪ F`)

Best path found across depths is used. If depth > 4 is needed, fall back to the binary construction for that target.

#### Target ordering policy

Order matters because earlier targets add intermediate values to the pool, cheapening later targets. Strategy:
1. Sort by ascending difficulty estimate: targets already in pool first (cost 0), then smooth, then others.
2. Among unsolved targets, prefer those that share the most candidate intermediates with already-solved targets (greedy reuse).

#### Pool eviction policy

When pool exceeds 48 slots (accounting for the 8 seed slots), evict by scoring each non-seed value:
```
eviction_score(v) = - (reuse_count(v) across remaining targets)
                   + (1 / depth_to_derive_v)   # penalty: cheap to re-derive = cheaper to evict
```
Evict the value with the lowest score (least reuse, cheapest to re-derive). Seeds are never evicted.

---

### 4e. Fallback Solver (Always Runs)

**Binary construction (guaranteed valid, upper-bound score):**
1. Produce `1` via `r0 / r0 → r_tmp`.
2. For each target `t`, build it in a scratch register by binary decomposition of its bits:
   * Double register (add to itself) for each `1` bit.
   * Add `1` for each set bit.
3. Never produces any value above `t`, so `value_limit` is automatically respected.

This is what the official baseline does. Cost: ~$\sum_t \lfloor\log_2 t\rfloor + \text{popcount}(t)$ per target, roughly 50–60 ops per target → ~1,500 per puzzle.

The fallback ensures we can always produce a valid submission, even if the M-i-M core fails.

---

## 5. Verifier (Stage 3)

An independent, clean simulator that replays any candidate program against the puzzle rules:
* Checks all 4 arithmetic constraint types (see `docs/02_rules_and_scoring.md`).
* Checks every target has been produced at least once.
* Returns `(valid: bool, hit_targets: set, error: str, instruction_count: int)`.

The solver **never trusts itself** — every program output is verified before being accepted.

> [!CAUTION]
> Never submit a CSV without running every row through the verifier. One invalid program fails the entire submission.

---

## 6. Post-Pass (Stage 4)

Applied after any valid program is produced:

### Dead Instruction Elimination
* Replay the program backwards. An instruction whose result is never read by any subsequent instruction or never hits a target is dead. Remove it.
* Implementation: a single forward replay recording which registers are read; then backward scan.
* Cost: free (no search required), guaranteed improvement ≥ 0.

### Peephole Rewrites
Local 2–3 instruction window rewrites:
* `[+, r, r, r]` followed by `[+, r, r, r]` → `[*, r_original, 4_const, r]` if `4` is in pool.
* `[/, a, a, r]` → produces `1`; check if `1` is already in pool and skip if so.
* Chained adds that build a constant `k` can be replaced by `* k` if `k` is in pool.

### Simulated Annealing (Local Search)
Applied when time budget allows, after the per-puzzle greedy solution:
* **Move types:**
  1. **Reorder**: swap two non-dependent instructions.
  2. **Replace intermediate**: change one intermediate helper value to an alternative derivation.
  3. **Re-derive target**: replace a target's derivation path with an alternative.
  4. **Insert/delete**: add a new shared helper, or remove a now-redundant one.
* **Scoring oracle**: the verifier counts instructions on the modified program.
* **Schedule**: geometric cooling, starting at T=2 (accept moves costing +2 extra ops), ending at T=0.01.
* **Time budget**: 2–5 seconds per puzzle on a single core.

---

## 7. Orchestrator (Stage 5)

* Runs the solver portfolio for each puzzle with multiple random seeds (target orderings, pool eviction tie-breaking).
* Tracks the best verified program per puzzle across all runs.
* Parallelizes via `multiprocessing.Pool` (Python) or thread pool (Rust/C++).
* Writes the final CSV using Python's `csv.writer` with correct quoting (no manual JSON string escaping).

**Output format per row:**
```python
import csv, json
writer = csv.writer(f)
writer.writerow([puzzle_id, json.dumps(program)])
# program = [["+", 0, 1, 2], ["*", 2, 2, 3], ...]
```

---

## 8. Implementation Language Choice

| Component | Language | Rationale |
| :--- | :--- | :--- |
| Verifier (reference) | Python | Already written; correct; easy to audit |
| Classifier | Python | Lightweight; Python speed is fine |
| Tiny BFS exact | Python | Small state space; Python speed adequate |
| M-i-M core | **Rust or C++** | Tight inner loop; 128-bit arithmetic; ~100× faster than Python |
| Post-pass annealing | Rust/C++ or Python | Needs simulator speed; prefer native |
| Orchestrator & CSV I/O | Python | Convenience; multiprocessing for parallelism |

Rust is preferred over C++ for memory safety and cross-compilation. The Python orchestrator calls the Rust binary via subprocess or `PyO3` bindings.

---

## 9. What 2026 Techniques Are Worth Using

| Technique | Role | Priority |
| :--- | :--- | :--- |
| **Classical M-i-M + Bos-Coster basis** | Core solver | **P0** |
| **Addition-subtraction chains (lit.)** | Informs M-i-M backward search | **P0** |
| **Yao/Pippenger multi-target heuristics** | Target ordering in M-i-M | **P1** |
| **OR-Tools CP-SAT** | Optimal basis selection for clustered puzzles | **P2** |
| **LLM-guided heuristic evolution (AlphaEvolve-style)** | Tune ordering policy, pool eviction weights on practice set | **P3 / experimental** |
| **MCTS with RL reward** | Guide deep search per puzzle | **P3 / experimental** |

Classical methods dominate. ML/LLM approaches would tune the *policies*, not the arithmetic search itself. Test on the practice set with `family` and `lower_bound` labels before running on scored puzzles.

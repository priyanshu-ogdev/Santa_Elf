# Santa 2026 - The Elf Calculator: Run Log

## Run 1: Baseline (`binary_shared` effective path)
- **Score:** `1,036,476` total instructions (~38.5 instructions/target across 26,902 scored targets; ~30.5% below the `1,490,379` sample submission).
- **Notes:**
  - `shared_hub.py` was silently crashing with `RuntimeError("Cannot evict: pool entirely occupied by seeds")` after exceeding the 48-register limit and falling back 100% of the time to `binary_shared`.
  - `wNAF` re-ran Horner doubling from the top bit for every target without base multiplication or cross-target DAG sharing.
  - Bundler alias imports (`fb_solve`, `bs_solve`) posed `NameError` risks in single-cell builds.

## Run 2: Upgraded 5-Engine DAG Synthesizer + C99 __int128 + Sethi-Ullman Compiler
- **Architecture Upgrades:**
  - `src/solvers/fast_engine.py`: C99 `unsigned __int128` dictionary (`210,000` nodes, `HT_SIZE = 2^20`, squaring backbone `2..2^32`, `O(N)` two-pointer scan) + `_PyDictBackend` fallback.
  - `src/solvers/dag_synthesizer.py`: 5-engine marginal-node DAG synthesis (Exact 0/1-op, Smooth-Neighbor + Pollard's Rho + Factor-BPE CSE, Multi-Radix Horner `B in {8,10,12,16,32,64}`, Active Anchors + Latent Common Roots, Recursive Two-Pointer + 20-bit Split), Rip-and-Reroute sweeps, and bounded SciPy HiGHS MILP polish.
  - `src/dag_compiler.py`: Sethi-Ullman postorder DFS + Subtree-Affinity target ordering, `seed_last_needed` reclamation, Read-Before-Write in-place reuse, and Belady rematerialization.
  - `src/solvers/tiny_bfs.py`: Bounded Best-First Search (`max_states=80,000`, `time_limit=0.25s`) compiled via `compile_dag`.
  - `src/postpass.py` & `build_submit.py`: `_live_shortcut_pass` peephole and self-contained `submit.py` / `submissions/kaggle_submission_cell.py` (`170,154` bytes).
- **Verified Local Checks:**
  - `pytest tests/ -v`: `38 passed in 255.56s` (`0` errors).
  - Stratified `practice_puzzles.csv` (15 puzzles, `lb=345`): **3,257** instructions vs `14,206` (`binary_shared`, **-77.1%**) and `19,281` (`fallback`, **-83.1%**).
  - Standalone `kaggle_submission_cell.py` E2E slice (`scored_0001..0008` + 4 tiny): `2,408` instructions (`2,392` vs prior `13,339` on `0001..0008`, **-82.1%**), `Verification PASSED with 0 errors!`.
- **Status:** Ready for full 1,000-puzzle Kaggle run to record final leaderboard score.

## Run 3 (Pre-Submission): V5 High-RAM Kaggle Architecture (`20-24 GB` Target)
- **Kaggle High-RAM Profile (`kaggle_high_ram` auto-enabled on `/kaggle` / `>=24 GB` Linux):**
  - **Shared Read-Only Universal `{1}` Dictionary `U`** (`init_universal_dict` before `fork()`): up to `300,000,000` packed `16 B` nodes + `2^30` 4-byte hash slots (`~9.09 GB` Read-Only Copy-on-Write shared `1x` across all 4 workers), covering 100% of `< 2^20` plus dense 20..36-bit blocks.
  - **Per-Worker Seed Dictionary `D_local`**: `4,000,000` nodes (`1,500,000` sorted `two_pointer` scan slice) + C99 `fe_mitm_probe` (`T = a*q +- r` across active DAG nodes + `450k` divisors over `D_local U U`) + C99 `fe_screen_smooth` (`+-256` offsets).
  - **Compiler & Global Optimization**: Min-Cut live register ordering + randomized swap hill-climbing (`64` trials) in `compile_dag`, Pair Rip-and-Reroute LNS, pairwise `gcd(A+-k, B+-k)` roots, and bounded HiGHS MILP up to `3,500` edges (`2.0s`).
  - **Bundled Cell**: `submit.py` & `submissions/kaggle_submission_cell.py` (`201.5 KB`).
- **Verified Local Checks (post-final edit):**
  - `pytest tests/ -v`: `38 passed in 343.68s` (`0` errors).
  - Standalone `kaggle_submission_cell.py` E2E slice (`scored_0001..0008` + 4 tiny, `U=8M` compact test mode): `2,305` total instructions (`2,289` vs V4 `2,392` vs baseline `13,339`, **-82.8%** vs baseline), `Verification PASSED with 0 errors!`.
  - Stratified 15-puzzle practice benchmark (`U=8M` compact test mode): `3,111` total vs V4 `3,257` (**-4.5%**) and `binary_shared` `14,206` (**-78.1%**).
- **Next:** Run `submissions/kaggle_submission_cell.py` on Kaggle (`4` CPUs, `~30 GB` RAM) and log final 1,000-puzzle score.

## Run 4 (Pre-Submission): V6 Parallel 64-Shard `U` (500M) + Bottleneck Fixes + RAM Telemetry
- **Root-Cause Fixes Applied After User Kaggle Log Audit (`238.4s` startup + silent output):**
  1. **Actual Kaggle RAM audit:** V5 only used `9.09 GB` (`U=300M`, `HT=2^30`) + `~1.2 GB` across 4 workers (`~10.3 GB` total). Upgraded `kaggle_high_ram` to **`U=500,000,000`** (`8.00 GB` vals+meta + **`2^31` 4-byte HT = `8.59 GB`**, **`~16.59 GB` CoW shared 1x**) + `4` workers x `D_local=10,000,000` (`HT=2^25`, `~300 MB/worker` + Python/HiGHS) = **`~20-24 GB` real Kaggle RSS** with zero OOM risk.
  2. **64-Shard Lock-Free 4-Thread `U` Builder (`fe_univ_create`):** Parallelizes `Phase B` across 4 OS threads using disjoint 64-shard hashing + segment compaction (`8M` local init dropped **`9.4s -> 1.4s`**, **`6.7x` faster**).
  3. **Fixed `fe_two_pointer` Multi-GB DRAM Stalls:** Uses the `2 MiB` L3-resident `seen24` bitset (`~1ns`) instead of missing a multi-GB DRAM hash table on every loop pair.
  4. **Fixed `fe_local_create` Layer 3 Blowup:** Bounded Layer 3 cross (`<=150k x 600` cheap constants) so `10M`-node local dicts build rapidly per puzzle.
  5. **Streaming Logs + Live Memory Telemetry (`orchestrator.py`):** Switched `imap_unordered` from `chunksize=62` (15-min silent stalls) to **`chunksize=1`** with `flush=True`, printing pre-init RAM, `U` allocation RSS/GB, every puzzle immediately, and aggregate Host Used/Avail GB every 25 puzzles.
- **Verified Local Checks (strictly post-final edit):**
  - `pytest tests/ -v`: **`38 passed in 579.63s`** (`0` errors).
  - Isolated standalone `submissions/kaggle_submission_cell.py` (`212.0 KB`, `U=8M` test cap): **`2,301` total instructions** (`scored_0001..0008`: `2,285`, `-82.9%` vs `13,339` baseline), `U` built in `1.4s`, `Verification PASSED with 0 errors!`.

## Run 4.1 (Hotfix Verified): Explicit `fork` CoW, BLAS Thread Caps & `Phase B/C` Stall Guards
- **Fixes Applied:**
  - `src/orchestrator.py`: Forced `multiprocessing.get_context("fork")` on Linux/Kaggle for guaranteed 1x CoW `U` sharing, pinned `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` per worker to stop HiGHS/OpenBLAS thread oversubscription across 4 CPUs, and enabled `sys.stdout.reconfigure(line_buffering=True)`.
  - `src/solvers/fast_engine.py`: Focused `Phase B` on the `260` lowest-depth `<= d5` multipliers with `since_add > 10M` / `since_c > 15M` progress-stall breaks so `U` initialization finishes rapidly (`8M` local init cut **`9.4s -> 0.9s`**).
- **Verified Post-Edit Checks:**
  - `pytest tests/ -v`: **`38 passed in 319.68s`** (nearly 2x faster than `579.63s`).
  - Isolated `submissions/kaggle_submission_cell.py` (`212.8 KB`, 12 scored puzzles, `U=8M` test cap): **`2,298` instructions in `33.4s`** (`77.8s -> 33.4s`, **2.3x faster**), `U` built in **`0.9s`**, `Verification PASSED with 0 errors!`.

## Run 5 (Kaggle Verified): `265,096` Total Instructions (-74.4% vs Baseline)
- **Verified Kaggle Output:** `Final Score: 265,096 total instructions.` (`0` verification errors across all 1,000 scored puzzles; down `-771,380` instructions from `1,036,476` baseline).
- **Host Telemetry:** `U=500,000,000` nodes (`16.59 GB CoW`) initialized in `437.8s`; steady Host RAM `~18.0 / 31.35 GB` (`~13.3 GB Avail`); Total Elapsed `5,317.1s` (88.6 min).
- **Post-Run Forensic Audit Findings:**
  1. **Critical Bug #1 (Silent Fallback to `binary_shared` on Prime `<50k` Targets):** When `< 2^20` primes (e.g., `3637`, `45821`) hit `U` directly inside `_precompute_smooth_bpe`, `de.has(p)` skipped NAF generation with `flat = [t]`, appending empty `ops = []` to `extra_cands[t]`. `consider_ops([])` returned `{}` (`len 0`), causing `best_defs` check `t in active_defs` to fail and throwing away the entire puzzle DAG to `binary_shared.solve()` (`scored_0259: 794` instead of `173`; `scored_0884: 814` instead of `164`; `scored_0429: 831`; `scored_0835: 820`, etc.).
  2. **Critical Bug #2 (Fake Cost=1 for Depth-8/9 `U` Nodes):** `de.has(rem)` scored `500M` `depth=8/9` nodes as cost `1`, causing `smooth_bpe` to discard true smooth factorizations and inflating `scored_0001` (`341` -> `403`).
  3. **Hardware Bottleneck (`437.8s` `U` Init + Under-Utilized `13.3 GB` RAM):** Multi-GB random DRAM misses in single-threaded compaction & `Phase B` saturated GCP DRAM bandwidth while leaving `13.3 GB` RAM unused.

## Run 6 (Pre-Submission): V7 Root Bug Fixes + SOTA Multi-Radix/Target-MST + 26M Local Dict
- **Score-Crushing Fixes & SOTA Upgrades Applied:**
  - **Bug #1 Fixed (`dag_synthesizer.py`):** Eliminated empty `ops = []` emissions on `< 50k` prime targets hit directly in `U`, guarded `consider_ops` with `v in nd`, and added single-target `_naf_ops` rescue so 1 hard target never dumps an entire puzzle to `binary_shared` (verified 8 Kaggle outliers collapsed **`5,501 -> 1,490` (`-72.9%`)**).
  - **Bug #2 Fixed (`fast_engine.py` + `dag_synthesizer.py`):** Added `DictEngine.lookup_cost()` scoring C99 `meta_cost` + `+3` `U` penalty accurately and retaining `cands[:2]` diverse smooth factorizations inside `_precompute_smooth_bpe`.
  - **SOTA Extensions:** Added `_target_mst_order` (Prim minimum directed transformation tree over target deltas & quotient bridges), extended Horner bases `(256, 1024, 65536, 100)`, and trimmed active anchor scan (`cand_anchors[-25:]`).
  - **High-RAM Re-Balancing & Startup Speedup (`kaggle_high_ram`):**
    - Removed `437.8s` single-threaded `u_ht` compaction loop and single-threaded `Phase C` bottleneck; parallelized `Phase B + Phase C` lock-free across 4 threads (`U = 360M` nodes, `HT = 2^31` $\implies$ `14.35 GB` CoW).
    - Upgraded per-worker `D_local` to **`26,000,000`** seed-specific nodes (`~690 MB/worker × 4 = ~2.76 GB`, cutting two-pointer gap on `scored_0001` from `16` bits down to `10` bits at `20M`), `local_scan = 1,000,000`, and `milp_max_edges = 5,500` (`~20–24 GB` peak RSS).
- **Verified Post-Final Edit Checks:**
  - `pytest tests/ -v`: **`38 passed in 290.23s`** (`0` failures).
  - Standalone `submissions/kaggle_submission_cell.py` (`216.2 KB`) E2E (`12` scored puzzles, `--cores 2 --ram-gb 4.0 --verify`): `2,318` instructions in `40.4s`, `U=8M` built in `1.2s`, `Verification PASSED with 0 errors!`.

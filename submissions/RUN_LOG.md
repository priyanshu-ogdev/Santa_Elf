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

## Run 7 (Pre-Submission): V8 Value-Set Beam (`fe_value_beam64`), Dual-Mode Seed-Pure/U Search & Balanced High-RAM (`240M U + 36M D_local`)
- **Architectural & Algorithmic Upgrades:**
  1. **C99 64-bit 2-Step Lookahead Value-Set Beam (`fast_engine.py`, `orchestrator.py`):** Added `fe_value_beam64` + `solve_value_beam(puzzle, beam_w=140, max_steps=40)` invoked alongside `dag_synthesizer` when `40 <= value_limit <= 25,000,000` and `len(targets) <= 15`. Uses sorted 1-step-from-target requirement indexing (`VBReq`) + 5-binary-search multi-target residual distance + exact set-equality deduplication, cutting small/medium puzzles cleanly (`scored_0014: 43 -> 31`, `scored_0019: 39 -> 29`, `scored_0024: 48 -> 31`). Also fixed Windows MinGW `-static-libgcc` + `os.add_dll_directory` linking and bumped shared-lib version tag (`santa2026_fast_engine_v8`).
  2. **Dual `allow_univ` Search & Deep Anchor/MITM Synthesis (`fast_engine.py`, `dag_synthesizer.py`):** Added zero-rebuild `DictEngine.set_allow_univ(allow)` (keying `_mitm_cache` by `(tgt, allow_univ)` and refreshing active `powers`), allowing `dag_synthesizer.solve()` to evaluate both `{1}`-assisted (`U`) and seed-pure (`D_local`-only) orderings while seeding `cand_recipes` with both for HiGHS MILP (`scored_0001: 403 -> 350`, `scored_0884: 814 -> 159`). Expanded `fe_mitm_probe` near-miss bound (`2^22 -> 2^28`), upgraded top-2 anchor residuals to `depth=1`, and added multi-power `2^s` splits (`s in {20,24,28,32,36}`) + exact dictionary/power checks.
  3. **Re-Balanced `kaggle_high_ram` Profile (`fast_engine.py`):** Re-allocated host RAM to `U=240,000,000` (`HT=2^30`, `~8.14 GB` CoW shared) + per-worker `D_local=36,000,000` (`local_scan=1,200,000`, `per_bucket=600,000`, `mitm_div_scan=260,000`, `milp_max_edges=5,500`), accelerating `U` init and expanding seed-specific two-pointer/MITM coverage across all 4 workers (~`14–21 GB` aggregate RSS on Kaggle's `31.35 GB` host).
- **Verified Post-Final Edit Checks (`ram_gb=4.0`, `U=8M` local test mode):**
  - **Stratified 50-puzzle `practice_puzzles.csv` (`lb=1,152`):** **`10,009`** (`8.69x lb`, down `-3.26%` from V7 `10,346`: `tiny=36`, `multiplicative=1,665`, `shared=1,793`, `mixed=3,148`, `independent=3,367`; `0` errors).
  - **11-puzzle Kaggle sample (`0001..0003, 0014, 0019, 0024, 0029, 0259, 0429, 0835, 0884`):** `1,966` total (`0` errors).
  - **Standalone `submissions/kaggle_submission_cell.py` (`238.5 KB`) E2E (`12` scored puzzles `scored_0001..0008 + 4 tiny`, `--cores 2 --ram-gb 4.0 --verify`):** **`2,280`** instructions in `51.7s` (`scored_0001..0008: 2,265` vs V7 `2,302` and Kaggle 265k run `2,407`; `U` ready in `1.0s`; `Verification PASSED with 0 errors!`).
  - **`python -m pytest tests/ -v`:** **`38 passed in 302.36s`** (`0` failures, `0` skips).
- **Status:** Ready to run `submissions/kaggle_submission_cell.py` on Kaggle (`4` CPUs, `~31 GB` RAM) for the full 1,000-puzzle submission.

## Run 8 (Pre-Submission): V9 C99 Radix-Sorted 10M Scan, Full Anchor/Rational Bridges, Bridge-Releasing Block-LNS & Spill-Minimizing Compiler
- **Root Audit & Bottleneck Discoveries (`data/puzzles.csv` = 1,000 puzzles, 26,898 non-seed targets):**
  - Classified all 1,000 competition puzzles structurally: **`shared_or_clustered` = `513` puzzles (`53.2%` of targets)**, **`mixed` = `189` (`20.9%`)**, **`independent_random` = `219` (`23.5%`)**, and `medium/tiny` (`79` puzzles, `2.4%`).
  - Identified 5 concrete bugs/bottlenecks in V8 (`20,789` on the 10% `data/puzzles.csv` slice at `4.0 GB`): (1) `scan_vals` truncated `96.7%` of `D_local` (`300k..1.2M` strided instead of full Radix-sorted scan) + slow `qsort` & `1.1 GB` Python copy; (2) `cand_anchors[-25:]` dropped all `15..35`-bit shared intermediates; (3) Single/Pair-Rip skipped every hub target `t in keep_other`, locking suboptimal bridge directions; (4) `fe_screen_smooth` skipped `k=0` & `>50k` prime factors in `practice_0016` (`2 * 29 * 4637 * 412609`); (5) `419` rematerialization spills across 100 puzzles (`~4,190` wasted ops) and excessive Python `set(active_defs.keys())` unions inside `_sub_dag_new_nodes`.
- **V9 Architectural Upgrades Implemented:**
  1. **C99 Engine (`fast_engine.py`, `santa2026_fast_engine_v9`):** Added 8-pass LSD Radix Sort (`radix_sort_sp`) + cost-prioritized `scan_vals` (`local_cap=24M`, `local_scan=10M` on Kaggle; `800k/800k` at `4 GB`), zero-copy `fe_dict_copy_scan_prefix` (`<=200k` Python slice), bounded strided `fe_two_pointer_s` (`stride=1` for `depth=2`, `stride=N//500k` for recursive gaps) preserving `_tp_cache` across `allow_univ` toggles, C99 `fe_anchor_scan` over ALL active intermediates, and expanded `U` divisor slice (`max_udiv=1,200,000`, `mitm_div_scan=420k`, `milp_max_edges=7000`).
  2. **Synthesizer & Portfolio (`dag_synthesizer.py`, `orchestrator.py`):** Added `k in {0,+-1..+-5}` guarantees & `2^25` Pollard-Rho + recursive `two_pointer` in `_precompute_smooth_bpe`; full `O(T^2)` Gauss/continued-fraction roots & rational `(u // d)*q +- r` bridges in `_discover_latent_roots`, `_target_mst_order`, and `5b`; `_sub_dag_new_nodes(..., max_nodes)` early cutoff (5x speedup) + live-read `ext` tie-break; Bridge-Releasing Subtree-Cluster Block-LNS (ripping `{t} U dependents(t)` together) + standalone seed-rooted MILP seeding; extended `solve_value_beam(beam_w=160, max_steps=44)` trigger to any puzzle with `max(targets) <= 25M` & `<= 15` targets.
  3. **Compiler & Postpass (`dag_compiler.py`, `postpass.py`):** Added Kahn min-live scheduler (`_schedule_min_live`), multi-use Belady eviction (`rem_uses * remat_cost`), `0.90s` spill hill-climbing, and `O(n)` algebraic peephole collapsing (`_peephole_algebra_pass`).
- **Verified Post-Final Edit Checks (`ram_gb=4.0`, `U=8,000,000`, `D_local=800,000` local compact mode):**
  - **Stratified 50-puzzle `practice_puzzles.csv` (`lb=1,152`):** **`9,577`** (`8.31x lb`, **`-4.32%`** vs V8 `10,009` and **`-7.43%`** vs V7 `10,346`: `tiny=36`, `multiplicative=1,587`, `shared=1,739`, `mixed=3,026`, `independent=3,189`; `total_remat=222`; `0` errors).
  - **12-puzzle Core Scored + Hard Practice Comparison:** `2,869` total in `172.3s` (`scored_0001: 339`, `scored_0002: 394`, `scored_0003: 353`, `practice_0016: 297`, `practice_0003: 388`, `practice_0034: 441`).
  - **Standalone `submissions/kaggle_submission_cell.py` (`268.4 KB`) E2E `slice12` (`scored_0001..0008 + 4 tiny`, `--cores 2 --ram-gb 4.0 --verify`):** **`2,194`** instructions in `48.5s` (`scored_0001..0008: 2,179` vs V8 `2,265`, V7 `2,302`, Kaggle V5 `2,407`; `U` built in `0.9s`; `Verification PASSED with 0 errors!`).
  - **Standalone `submissions/kaggle_submission_cell.py` E2E `slice100` (`10%` stratified `data/puzzles.csv`, `--cores 2 --ram-gb 4.0 --verify`):** **`19,956`** instructions in `391.9s` (**`-833` ops / `-4.01%`** vs V8 `20,789`; `Verification PASSED with 0 errors!`).
  - **`python -m pytest tests/ -v`:** **`38 passed in 324.98s`** (`0` failures, `0` skips).
- **Status:** Ready to run `submissions/kaggle_submission_cell.py` on Kaggle (`4` CPUs, `~31 GB` RAM, `U=240M + D_local=24M / 10M scan`) for the full 1,000-puzzle submission.

## Run 8.1 (Architecture Audit & Pipeline Streamlining)
- **Full-Pipeline Audit & Fixes Applied:**
  1. **Bundle Namespace & Kaggle Compatibility (`build_submit.py`, `orchestrator.py`):** AST-verified `0` top-level symbol collisions across all `95` top-level definitions in `14` bundled modules (`277.4 KB` self-contained `submit.py` & `submissions/kaggle_submission_cell.py`), checked `max(value_limit) < 2^60` & `max_seeds = 8` across all `1,000` puzzles, and added explicit error reporting around `init_universal_dict` & `dag_synthesizer` instead of silent `pass`.
  2. **C99 Safety Guards, `fe_find_active_1ops` & Sorted `fe_two_pointer_s` (`fast_engine.py`):**
     - Added NULL/empty-scan/hash-load guards in `alloc_bigdict`, `fe_dict_copy_scan`, `fe_dict_copy_scan_prefix`, `fe_two_pointer_s`, and `fe_mitm_probe`; expanded `VB_SEEN_HT` (`32,768 -> 65,536`).
     - Added C99 `fe_find_active_1ops` + reusable `ctypes` buffers (`_act_buf`), eliminating `4.33M+` Python `de.has()` calls per puzzle (`~4.4s -> <0.05s` wrapper overhead).
     - Fixed `two_pointer` cache truncation when `K=1/2` precedes `K=16`, sorted `hits_gap` & `hits_sc` deterministically in C99 `fe_two_pointer_s`, replaced `round()` float approximation with exact 60-bit integer nearest division `(v + (d >> 1)) // d`, and auto-versioned the compiled C99 shared library with `zlib.crc32(_C_SOURCE)`.
     - Fixed named `--ram-gb` profile parsing (`"kaggle_high_ram"`, `"high_ram_16g"`, `"medium_ram"`, `"compact_ram"`, `"low"`), split `DictEngine._caches` by `allow_univ` to prevent cache wipes on toggle, and spaced `split_muls` across `18..24`-bit low-cost multipliers.
  3. **Synthesizer & Orchestrator De-Duplication (`dag_synthesizer.py`, `orchestrator.py`):**
     - Added single-op fast path in `_sub_dag_new_nodes` (skipping redundant 3-color cycle DFS and dict-edge re-verification), fixed `1`-op/`2`-op early-exit guards when `recipe_collector` is active, cached `_radix_horner_ops` per `(v, allow_univ, depth >= 1)`, deduplicated `split_shifts` quotients, and guarded redundant `binary_shared` runs when `dag_synthesizer` already beats `55%` of target bits.
- **Verified Post-Final-Edit Checks:**
  - **Bundle & AST Verification:** `0` collisions across `95` top-level symbols; `py_compile` passed on `submit.py` & `submissions/kaggle_submission_cell.py` (`277.4 KB`).
  - **`python -m pytest tests/ -v`:** **`38 passed in 320.80s`** (`0` failures, `0` skips).
  - **Standalone `submissions/kaggle_submission_cell.py` E2E `slice12` (`scored_0001..0008 + 4 tiny`, `--cores 2 --ram-gb 4.0 --verify`):** **`2,199`** instructions in **`43.5s`** (down `-5.0s` from `48.5s`, `U=8M` ready in `1.2s`, `Verification PASSED with 0 errors!`).
  - **Direct C99 `kaggle_high_ram` Local-Dict Probe (`24M` cap, `10M` scan, pre-edit C99 kernel baseline):** `fe_local_create` completed in `1.73s` (`scored_0001`, `14.66M` nodes) and `2.12s` (`scored_0002`, `17.48M` nodes, `0.745 GB`).

## Run 8.2 (Algorithmic Gating Audit, Attribution & Widened C99 Value Beam)
- **Empirical Gating & Diagnostic Upgrades:**
  1. **Compiler Low-Pressure Fallback & Heavy-Spill Rescue (`dag_compiler.py`, `dag_synthesizer.py`):** Added `_emergency_compile` Sethi-Ullman low-pressure spill fallback (`55`-wide live-leaf DAG compiled validly in `103` ops) and zero-regression conditional `n_trials=256` re-compile when initial spills exceed `6` (`len(best_prog) > len(cur_defs) + 6`, shaving `-4` on `scored_0941` and `-6` on `scored_0106`).
  2. **Per-Op Attribution, LPT Dispatch & Warning Surfacing (`dag_synthesizer.py`, `orchestrator.py`):** Added `_diag_out` (`dict_vals` extraction inside `DictEngine`) and `_attribute_program` breaking every solution into `make_one`, `target_leaf`, `shared_hub`, `dict_anc`, `glue`, and `remat`; added LPT puzzle dispatch sorting, `--archive` monotonic best-of CSV support, parent-process warning surfacing across worker pools, and tightened `binary_shared` fallback (`len(candidates) <= 1`).
     - **Attribution Audit (`100` stratified puzzles, `2,675` targets, `22,237` ops, `8.31 ops/tgt`):** `dict_anc=9,039 (40.6%)`, `shared_hub=6,116 (27.5%)`, `glue=4,923 (22.1%)`, `target_leaf=2,060 (9.3%)`, `make_one=99 (0.4%)`, `remat=415 (1.9%)` — confirming dictionary ancestry dominates marginal target cost and bounds practical 1,000-puzzle totals above ~`150–185k` vs `100k`.
  3. **Widened Dictionary-Guided C99 Value Beam (`fast_engine.py`, `orchestrator.py`):** Expanded `VB_MAX_POOL` to `120`, added L3 `seen24`-fast `vb_dict_cost` / `vb_dict_pair_ctg` lookahead inside `fe_value_beam64`, and widened `orchestrator` trigger to `max(targets) <= 100,000,000` and `len(targets) <= 20` (`beam_w=240`, `max_steps=72`, `use_univ=True`), saving **`-23` ops (`-8.4%`)** on the `<=100M` subset of `strat50.csv` (`scored_0066: 54->43`, `0121: 65->60`, `0828: 47->40`, `0019: 30->29`, `0024: 31->29`).
- **Hypotheses Tested & Gated Out (Not Shipped):**
  - *Standalone Depth-3 Backward BFS:* Expanded `358,752`, `287,138`, `307,438` inverse states on top-3 targets of `scored_0001` (`~1.6–1.8s` each) with **`0` intersections** against `U(8M)+D(800k)`.
  - *Hub-First Iterative Column Generation (`closed_recipes` HiGHS Master MIP):* Saved **`1` op (`0.018%`)** across `25` stratified puzzles for **`+76.0s` CPU overhead** (`+0` even with bounded top-80 exact `status=0` MIPs). Dropped cleanly.
  - *Dictionary-Guided Recipe-DAG Beam (`width=6`, 2 orders, elitism + bridge lookahead):* Net **`-24 / 5,451` (`-0.44%`)** across `25` stratified puzzles at **`+37%` CPU time** (`218.1s -> 298.4s`, `0` diff on `17/25` puzzles). Dropped cleanly.
- **Verified Post-Final-Edit Checks:**
  - **Bundle & AST Verification:** `submit.py` & `submissions/kaggle_submission_cell.py` (`290.4 KB`, `93` top-level symbols, `0` AST collisions, `py_compile` clean).
  - **`python -m pytest tests/ -v`:** **`38 passed in 284.31s`** (`0` failures, `0` skips).
  - **Standalone `submissions/kaggle_submission_cell.py` E2E `strat50.csv` (`50` stratified puzzles, `1,333` targets, `--cores 3 --ram-gb 4.0 --verify`):** **`10,779` total instructions** in **`127.9s`** (down from `10,807` baseline on `strat50.csv`; `8.09 ops/tgt`; `dict_anc=39.8%`; `Solver wins: {'dag_synthesizer': 45, 'solve_value_beam': 4, 'tiny_bfs': 1}`; `Warnings: 0`; `Verification PASSED with 0 errors!`). Note: on Windows `spawn` workers `_UNIV_CTX` is absent in child processes (`scored_0066`/`0828`), whereas Kaggle Linux `fork()` shares `U` CoW across workers for additional `<=100M` Value Beam wins.

## Run 9 (Pre-Submission): SOTA Phase A/B/D Upgrades — Bridges, Ershov Compiler & C99 Window Superoptimizer
- **Hidden-Structure & Floor Audit (`data/puzzles.csv` = 1,000 puzzles, 26,898 non-seed targets, 1,008,539 bits):**
  - Topology-deduplicated DAG counting floor (`1.5*m^2` minus `L!` isomorphism & `k!` ordering): **`137,714` ops (`5.12 ops/tgt`)** (loose un-deduplicated `82,449` ops).
  - Structural checks across `ALL_1000`: `exact_AP = 0`, `exact_geom = 0`, genuine 3-term small linear relations (`>=36b`, `|c_i|<=2`, `gap<4096`) in only `60/1000` (`shared`), 8-target Fraction LLL found only Minkowski pigeonhole norms, and strict Approximate-GCD LLL (`|r_i| < max(10k, H>>12)`) hit `1/50` (`scored_0433` multiple of `t_0`), proving no hidden `100k` shortcut exists in `data/puzzles.csv`.
- **Upgrades Implemented & Regressively Kept:**
  1. **Phase A — Deep Bridges & Dual-Mode Value Beam (`dag_synthesizer.py`, `orchestrator.py`):** Added active pair sum/diff 2-op bridge (`v ≈ (a +- b) +- gap`), C99 `two_pointer(K=2, fast=True)` residual fallback inside `split_shifts` (`14..38b`) & `anchor_scan` (`14..36b`), expanded `split_muls[:8]`, extended shifted GCDs `off in -2..2` + equal-shift `|k| <= 8` in `_discover_latent_roots`, and dual `use_univ in (True, False)` evaluation inside `solve_value_beam` (`scored_0121: 65 -> 61`, `scored_0221: 42 -> 41`). Clean `2,444 -> 2,432 (-12)` on 10-puzzle `D24M_10M` probe with `0` regressions.
  2. **Phase B — Ershov Sethi-Ullman & Remat-Aware Compiler (`dag_compiler.py`):** Added `_compute_ershov` register-pressure metric, `_hub_cluster_target_order`, in-place dead-temp overwrite inside `ensure_live`, contiguous cluster moves, and post-climb `soft_remat` (`1 + 0.15*(rem_uses - 1)`) tie-break (`scored_0075` remat `32 -> 18`, `411 -> 397`).
  3. **Phase D — Targeted C99 Sliding-Window Superoptimizer (`dag_synthesizer.py`, `fast_engine.py`):** Added `_superopt_windows` (`solve_value_beam(beam_w=64)` over `w in (6, 9, 12)` live-register snapshots, remat-duplicate sites prioritized first, strictly verified via `verify_program`, profile `superopt_time = 1.2..2.0s`), cutting `2,444 -> 2,408 (-36 / -1.47%)` on the 10-puzzle `D24M_10M` probe.
- **Hypotheses Tested & Gated Out (Reverted Cleanly):**
  - *`fe_local_create` 2-pass Layer 2 + `per_bucket=750k` + `fe_sample_band` (`fast_engine.py`):* Reverted after causing `+47` (`2,444 -> 2,491`) regression (`cost=7..8` dilution in `scan_vals` and `19`-bit `split_muls` dilution vs `HEAD`'s 24-bit `[8.76M..14.45M]`).
  - *Phase C `Hcub` pre-seeding (`[hcub_hubs] + mst_order`):* Reverted after `+1/+2` regressions and `+30s`/puzzle runtime bloat.
  - *Local DAG E-Graph distributivity/associativity (`dist`, `assoc` scan):* Measured `dist=0, assoc=0..1` across `~1,877` instructions on 5 hardest puzzles (`0` yield).
- **Verified Post-Final-Edit Checks:**
  - **Bundle & AST Verification:** `submit.py` & `submissions/kaggle_submission_cell.py` (`306.1 KB`, `96` top-level symbols, `0` AST collisions, `py_compile` clean).
  - **`python -m pytest tests/ -v`:** **`38 passed in 381.11s`** (`0` failures, `0` skips).
  - **Standalone `submissions/kaggle_submission_cell.py` E2E `strat50.csv` (`50` stratified puzzles, `1,333` targets, `--cores 3 --ram-gb 4.0 --verify`):** **`10,625` total instructions** (**`-154` ops / `-1.43%`** vs Run 8.2 `10,779`, `7.97 ops/tgt`) in `168.1s`; `remat` spills fell **`196 (1.8%) -> 116 (1.1%) (-40.8%)`**, `dict_anc` fell **`4,288 -> 4,088`**, `Solver wins: {'dag_synthesizer': 45, 'solve_value_beam': 4, 'tiny_bfs': 1}`, `Warnings: 0`, **`Verification PASSED with 0 errors!`**.


## Run 10 — C99 Dynamic Active-Pair Engine, Liveness SA & Shared-Hub Block-LNS (`head`)

### Key Upgrades Since Run 9 (`262c924`)
1. **Topological Min-Live Beam Scheduler (`src/dag_compiler.py`):** Added `_schedule_min_live_beam` (Kahn beam minimizing spills, peak live registers, and soft excess pressure) invoked in `compile_dag` only when greedy `_schedule_min_live` leaves spills. Verified `-1` op (`2146 -> 2145`) on `6`-puzzle ablation and `-3` (`2408 -> 2405`) vs Run 9 (`depth=1` NAF/1-op gap fallback was tested and rejected after regressive `+6` greedy drift ablation).
2. **C99 Dynamic Active-Pair Virtual Dictionary Engine (`src/solvers/fast_engine.py`, `src/solvers/dag_synthesizer.py`):** Added `ActPairSet`, `fe_act_pairs_create` (LSD radix sort over 1-op `+`, `-`, `*` combinations of active DAG values `<= 180`), `fe_act_pairs_free`, `fe_act_probe` (Probe 1: `E +- gap`, Probe 2: `E * q +- gap` over `D_local U U` gated by L2 `seen24`, Probe 3: strided two-pointer `E * d_local +- gap`), cached on `DictEngine` (`act_probe`), and wired into `_synthesize_best_marginal` as `Engine 8b` (`depth >= 2`). Reduced the `10`-puzzle `D24M_10M` probe from `2405` to `2299`.
3. **Multi-Start Fast Liveness SA, Shared-Internal-Hub Block-LNS & Alt-DAG Checkpoints (`src/dag_compiler.py`, `src/solvers/dag_synthesizer.py`):**
   - Upgraded `compile_dag` with fast liveness SA pre-search (`_fast_pressure`, multi-start from `best_order` and `hub_cluster` with adaptive time budget `0.35s–1.35s`) plus `soft_remat=True` across `top_scheds[-4:]`.
   - Updated Block-LNS in `solve()` to iterate over all shared internal DAG hub clusters `val_users.values()` (`2 <= len(cl) <= 5`) instead of only target nodes.
   - Checkpointed `alt_init_defs`, `pre_pair_defs`, and `pre_block_defs` so heavy-spill DAGs (`len(best_prog) > len(cur_defs) + 6`) evaluate alternate smaller-spill DAGs instead of duplicating deterministic `compile_dag` calls on `cur_defs`.

### Verified Results (`Run 10`)
- **`D24M_10M` 10-Puzzle High-RAM Dict Probe (`U=8M, D_local=10M`):** **`2,268`** vs Run 9 `2,408` (**`-140` ops / `-5.81%`**, beats or ties Run 9 on 10/10 puzzles: `0075: 399->384`, `0612: 389->387`, `0899: 343->326`, `0106: 363->331`, `0941: 365->339`, `0442: 280->251`, `0966: 121->108`, `0066: 43->43`, `0828: 44->42`, `0121: 61->57`).
- **Full `strat50.csv` E2E Standalone Verification (`submissions/kaggle_submission_cell.py --cores 3 --ram-gb 4.0 --verify`):**
  - **`9,921` total ops (`7.44 ops/tgt`)** vs Run 9 `10,625` (**`-704` ops / `-6.63%`**), `W/L/T = 45/2/3`, elapsed `255.9s`, `Warnings: 0`, `Verification PASSED with 0 errors!`.
  - Attribution (`1,333` targets): `dict_anc=3,277 (33.0%)`, `shared_hub=3,124 (31.5%)`, `glue=2,438 (24.6%)`, `target_leaf=1,038 (10.5%)`, `make_one=44 (0.4%)`, `remat=84 (0.8%)`.
- **Unit Test Suite (`python -m pytest tests/ -v`):** **`38 passed in 405.94s`** (`0` failures, `0` skipped).
- **Bundle Build (`python build_submit.py`):** `submit.py` and `submissions/kaggle_submission_cell.py` built (`337.1 KB`), `0` AST collisions.

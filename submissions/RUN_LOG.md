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

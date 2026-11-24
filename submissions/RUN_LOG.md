# Santa 2026 - The Elf Calculator: Run Log

## Run 1: Baseline (`binary_shared` effective path)
- **Score:** `1,036,476` total instructions (~38.5 instructions/target across 26,902 scored targets; ~30.5% below the `1,490,379` sample submission).
- **Notes:**
  - `shared_hub.py` was silently crashing with `RuntimeError("Cannot evict: pool entirely occupied by seeds")` after exceeding the 48-register limit and falling back 100% of the time to `binary_shared`.
  - `wNAF` re-ran Horner doubling from the top bit for every target without base multiplication or cross-target DAG sharing.
  - Bundler alias imports (`fb_solve`, `bs_solve`) posed `NameError` risks in single-cell builds.

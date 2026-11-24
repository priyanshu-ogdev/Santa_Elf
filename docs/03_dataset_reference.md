# Dataset Reference & Submission Specification

This guide outlines the files, schemas, statistical profiles, and submission format for the competition.

---

## 1. Directory Structure

The competition dataset is organized in `d:\Santa\data`:

```
d:\Santa\data\
├── puzzles.csv             # 1,000 official scored puzzles (Leaderboard evaluation)
├── practice_puzzles.csv    # 200 diagnostic benchmark puzzles with metadata
└── sample_submission.csv   # 1,000 baseline solutions (Total instructions: 1,490,379)
```

---

## 2. Statistical Profiles

### A. Scored Set (`puzzles.csv` — 1,000 rows)
* **Columns:** `id`, `starting_values`, `targets`, `value_limit`, `registers`
* **Starting Values ($K$):**
  * Range: $[1, 8]$ (Mean: 3.51)
  * Distribution: 4 seeds (246), 3 seeds (236), 2 seeds (226), 5 seeds (197), 6 seeds (48), 1 seed (39), 7 seeds (7), 8 seeds (1).
  * Value Range: $[1, 1998]$
* **Targets ($T$):**
  * Range: $[1, 32]$ (Mean: 26.90, Median: 30)
  * Value Range: $[1, 999940663716834356]$ ($\approx 10^{18}$)
* **`value_limit`:**
  * Dynamic per puzzle: $[7, 999983866824517514]$
* **`registers`:** Fixed at **48** across all puzzles.

### B. Practice Set (`practice_puzzles.csv` — 200 rows)
* **Columns:** `id`, `starting_values`, `targets`, `value_limit`, `registers`, `family`, `design_tier`, `lower_bound`
* **Family Breakdown:**
  * `shared` (51 puzzles): Targets with high structural overlap sharing intermediate sub-expressions.
  * `mixed` (48 puzzles): Complex realistic hybrid instances.
  * `independent` (45 puzzles): Targets with low factor correlation requiring disjoint paths.
  * `multiplicative` (40 puzzles): Targets sharing common prime factor bases.
  * `tiny` (16 puzzles): Small seeds and small targets for rapid sanity tests and unit checks.
* **Design Tier Breakdown:**
  * `hard` (170 puzzles, 85%)
  * `easy` (16 puzzles, 8%)
  * `medium` (14 puzzles, 7%)
* **Theoretical Lower Bound:**
  * Range: $[1, 32]$ (Mean: 26.44, Median: 29.5)
  * Reflects the minimum non-seed target count $\approx T - \text{seeds\_matching\_targets}$.

---

## 3. Submission Format

Submissions require a UTF-8 encoded CSV file with exactly two columns: `id` and `program`.

```csv
id,program
scored_0001,"[[""/"" 0, 0, 1], [""+"", 0, 0, 2], [""*"", 2, 2, 3]]"
scored_0002,"[[""/"" 0, 0, 1], [""+"", 1, 1, 2]]"
...
scored_1000,"[[""/"" 0, 0, 1]]"
```

### Submission Checklist
1. **Row Count:** Exactly 1,000 rows matching `scored_0001` through `scored_1000`.
2. **Column Headers:** Exactly `id,program`.
3. **Escaped JSON:** The `program` cell must be a valid JSON string representing an array of 4-element arrays: `["op", src1, src2, dst]`.
4. **Register Range:** All register references must be integers in $[0, 47]$.
5. **Deterministic Validation:** You can validate any submission file against `puzzles.csv` locally before uploading using `scratch/validator.py`.

# Exploratory Data Analysis (EDA): Santa 2026

Empirical breakdown of the official dataset files in `d:\Santa\data`.

---

## 1. Dataset Summary

| Metric | `puzzles.csv` | `practice_puzzles.csv` | `sample_submission.csv` |
| :--- | :--- | :--- | :--- |
| **Row Count** | 1,000 puzzles | 200 puzzles | 1,000 solutions |
| **File Size** | 416.6 KB | 87.9 KB | 20.88 MB |
| **Identifier Prefix** | `scored_0001` – `scored_1000` | `practice_0001` – `practice_0200` | `scored_0001` – `scored_1000` |
| **Registers** | Fixed at 48 | Fixed at 48 | Max register used: **9** (baseline uses ≤10 registers) |

---

## 2. Starting Values Distribution

### Scored Puzzles (`puzzles.csv`)
* **Count per Puzzle:** Min: 1, Max: 8, Mean: **3.51**
* **Frequency Breakdown:**
  * 4 starting values: 246 puzzles (24.6%)
  * 3 starting values: 236 puzzles (23.6%)
  * 2 starting values: 226 puzzles (22.6%)
  * 5 starting values: 197 puzzles (19.7%)
  * 6 starting values: 48 puzzles (4.8%)
  * 1 starting value: 39 puzzles (3.9%)
  * 7 starting values: 7 puzzles (0.7%)
  * 8 starting values: 1 puzzle (0.1%)
* **Starting Value Magnitudes:** Min: **1**, Max: **1,998** (all small positive integers).

---

## 3. Targets Distribution

### Scored Puzzles (`puzzles.csv`)
* **Targets per Puzzle:** Min: 1, Max: 32, Mean: **26.90**, Median: **30.0**
  * Most scored puzzles contain 25 to 32 target values to hit.
* **Target Magnitudes:**
  * Minimum Target: **1**
  * Maximum Target: **$999,940,663,716,834,356$** ($\approx 10^{18}$)
  * Targets span across small constants, medium integers, and 60-bit large integers.

---

## 4. `value_limit` Distribution
* `value_limit` is dynamic across puzzles:
  * Minimum `value_limit`: **7** (for tiny puzzles where intermediate calculations cannot exceed single digits).
  * Maximum `value_limit`: **$999,983,866,824,517,514$** ($\approx 10^{18}$).
  * Fits safely within signed 64-bit integers (`int64` max is $\approx 9.22 \times 10^{18}$).

---

## 5. Practice Puzzles Metadata Breakdown

### By Puzzle Family (`practice_puzzles.csv`)
| Family | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `shared` | 51 | 25.5% | Targets share intermediate calculations. |
| `mixed` | 48 | 24.0% | Realistic complex combinations. |
| `independent` | 45 | 22.5% | Disjoint targets requiring independent paths. |
| `multiplicative`| 40 | 20.0% | Targets sharing common prime factor bases. |
| `tiny` | 16 | 8.0% | Micro puzzles for rapid unit tests. |

### By Design Tier
| Tier | Count | Percentage |
| :--- | :--- | :--- |
| `hard` | 170 | 85.0% |
| `easy` | 16 | 8.0% |
| `medium` | 14 | 7.0% |

### Theoretical Lower Bounds
* Range: $[1, 32]$ (Mean: 26.44, Median: 29.5)
* Matches the number of non-seed targets, providing a theoretical absolute lower bound benchmark.

---

## 6. Official Baseline Submission Analysis (`sample_submission.csv`)

* **Total Scored Instructions:** **1,490,379**
* **Instructions per Puzzle:**
  * Min: 3 instructions (`scored_0255`)
  * Max: 2,873 instructions (`scored_0947`)
  * Mean: 1,490.38 instructions
  * Median: **1,596** instructions *(corrected from earlier 1,593 estimate)*
* **Operator Frequency:**
  * `+` (Addition): **1,462,481** (98.13%)
  * `*` (Multiplication): **26,898** (1.80%)
  * `/` (Division): **1,000** (0.07% — exactly 1 division per puzzle: `["/", 0, 0, 1]` to produce the constant `1`)
  * `-` (Subtraction): **0** (0.00%)
## 7. Structural Analysis of Scored Targets (Corrections & Additions)

> [!NOTE]
> This section supersedes earlier claims that scored targets were "essentially unrelated". The scored set is a **mixture** of problem families.

### A. Close Adjacent Targets
* **3.79%** of adjacent sorted target pairs in the scored set are within 0.01% of each other.
* This matches the practice set (4.8% at 0.1% threshold) and significantly exceeds a log-uniform control (~0.1%), indicating structure.
* The `shared` practice family reaches 9.7% on this measure; `independent` is 0.2%.

### B. Target Smoothness (Largest Prime Factor ≤ 2000)
* **13.8%** of scored targets in the first 300 puzzles have all prime factors ≤ 2000 ("smooth").
* Compared to: `multiplicative` family (practice) 39%, `independent` family ~1.5%.
* This suggests the scored set mixes multiplicative-structured and independent puzzles.

### C. Pairwise GCD > 1000
* In the first 20 scored puzzles, every puzzle has at least some target pairs with large GCDs.
* Puzzle `scored_0003` has 8 such pairs out of 378; `scored_0008` has 7.
* This is not "under 1%" — the scored set has significant pairwise structure in many puzzles.

### D. Free Target Hits (Seeds That Equal Targets)
* Exactly **4 targets** across all 1,000 puzzles are equal to a seed value and cost zero instructions.

### E. Tiny-Limit Puzzles (value_limit < 1,000)
Exactly 16 puzzles have tiny value limits. These are solvable optimally via BFS and require special handling:

| ID | value_limit | Seeds | Targets | Max Target |
| :--- | :--- | :--- | :--- | :--- |
| `scored_0061` | 12 | [1, 5] | 4 | 8 |
| `scored_0126` | 9 | [3] | 4 | 8 |
| `scored_0301` | 12 | [3, 6, 12] | 3 | 10 |
| `scored_0348` | 7 | [1] | 3 | 5 |
| `scored_0365` | 12 | [11] | 2 | 7 |
| `scored_0485` | 10 | [7, 8] | 1 | 2 |
| `scored_0490` | 16 | [1, 16] | 3 | 13 |
| `scored_0537` | 10 | [2] | 2 | 4 |
| `scored_0597` | 15 | [3, 6] | 4 | 10 |
| `scored_0600` | 8 | [1, 2] | 2 | 6 |
| `scored_0676` | 14 | [11] | 4 | 7 |
| `scored_0721` | 16 | [3] | 3 | 10 |
| `scored_0798` | 16 | [11, 15] | 3 | 15 |
| `scored_0883` | 13 | [10] | 4 | 13 ← tight (vl == max\_target) |
| `scored_0887` | 10 | [5] | 1 | 8 |
| `scored_0931` | 8 | [5, 6, 7] | 2 | 8 ← tight (vl == max\_target) |

### F. Tight-Limit Puzzles (value_limit == max(targets))
* **2 puzzles** have zero headroom: `scored_0883` (vl=13) and `scored_0931` (vl=8).
* No "overshoot and divide" tricks are possible here — every intermediate must stay ≤ the final target.

### G. 128-bit Arithmetic Requirement
* Seed values reach ~2000, and value limits reach ~10^18.
* A multiply of two values near 10^18 overflows `int64` (max ~9.22 × 10^18).
* The solver core must use **128-bit integers** (Python `int` is unbounded; C/Rust needs `__int128` or `i128`).

# Santa 2026 – The Elf Calculator: Competition Overview

> **Competition:** [Santa 2026 - The Elf Calculator](https://www.kaggle.com/competitions/santa-2026)  
> **Host / Sponsor:** Google & Kaggle  
> **Prize Pool:** $50,000 USD  
> **Timeline:** November 18, 2026 – January 28, 2027 (11:59 PM UTC)  
> **Baseline Benchmark:** 1,490,379 instructions (~1,490 instructions / puzzle)  

---

## 1. Lore & Background

The holiday rush at the North Pole is in full swing. Elves across toy assembly and packaging divisions rely heavily on the **Workshop Calculator** to tally inventory, bundle presents into dispatch cartons, and optimize cargo loads for Santa's sleigh.

Right before the final shipping wave, the calculator suffered a critical hardware breakdown. Trapped in a bare-metal fallback state, it has lost all complex logic functions and can only execute single binary integer arithmetic instructions. Furthermore, its memory is constrained to just **48 registers**.

Every instruction executed introduces clock delays into the packaging line. Santa needs programmatic solutions that generate every target tally in the fewest instructions possible.

---

## 2. Core Problem & Motive

For each of the **1,000 scored puzzles** (`scored_0001` to `scored_1000`), you are given:
1. **Starting Seeds:** 1 to 8 small positive integers ($1 \le s_i \le 1998$) loaded into the initial registers.
2. **Targets:** 1 to 32 large positive integers (up to $\sim 10^{18}$, mean $26.9$ targets/puzzle) that must be calculated.
3. **Hardware Boundary:** Exactly **48 registers** (`0` to `47`).
4. **Dynamic Value Limit:** A puzzle-specific ceiling (ranging from $7$ to $\sim 10^{18}$) that no register may exceed.

---

## 3. Key Quantitative Benchmarks (From Real Data)

| Metric | Scored Set (`puzzles.csv`) | Practice Set (`practice_puzzles.csv`) |
| :--- | :--- | :--- |
| **Total Puzzles** | 1,000 | 200 |
| **Available Registers** | 48 (all puzzles) | 48 (all puzzles) |
| **Starting Seeds / Puzzle** | 1 to 8 (Mean: 3.51) | 1 to 8 (Mean: 3.41) |
| **Targets / Puzzle** | 1 to 32 (Mean: 26.90, Median: 30) | 1 to 32 (Mean: 26.47, Median: 29.5) |
| **Max Seed Value** | 1,998 | 1,973 |
| **Max Target Value** | $999,940,663,716,834,356$ ($\approx 10^{18}$) | $998,263,746,689,377,336$ ($\approx 10^{18}$) |
| **Official Baseline Score** | **1,490,379 instructions** | N/A |
| **Baseline Median** | **1,596 instructions / puzzle** | N/A |
| **Top Competitor Range** | **~130,000 instructions (unverified estimate)** | N/A |

---

## 4. High-Level Objectives

* **Hit Every Target:** Every target integer must appear in at least one register during execution. Targets do not need to be preserved until the end.
* **Global Instruction Minimization:**
  $$\text{Global Score} = \sum_{p=1}^{1000} \text{Instructions}(p)$$
  **Lower is better.** Every instruction eliminated improves your rank.
* **Zero Fault Tolerance:** Any inexact division, non-positive result, out-of-bounds register reference, uninitialized read, or value overflow invalidates the submission.

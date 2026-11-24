# Elf Calculator: Rules, Architecture & Scoring Mechanics

This document provides the formal operational specification of the Elf Calculator machine, instruction semantics, arithmetic constraints, and empirical baseline insights.

---

## 1. Machine Architecture

The Elf Calculator is an abstract register-transfer machine with fixed memory and binary operations.

* **Registers:** Exactly **48 registers** indexed `0` to `47` ($r_0, r_1, \dots, r_{47}$).
* **Initial State:**
  * For a puzzle with $K$ starting values ($[s_0, s_1, \dots, s_{K-1}]$ where $1 \le K \le 8$):
    $$r_i = s_i \quad \text{for } 0 \le i < K$$
  * All remaining registers ($r_K$ through $r_{47}$) start **uninitialized** (`None`).
  * Attempting to read an uninitialized register causes an immediate runtime error.

---

## 2. Instruction Specification

Each instruction is serialized as a 4-element JSON array:
```json
["op", src1, src2, dst]
```

### Components
1. **`op` (Operator):** A single string specifying the arithmetic operation:
   * `"+"`: Integer addition ($r_{\text{src1}} + r_{\text{src2}}$)
   * `"-"`: Integer subtraction ($r_{\text{src1}} - r_{\text{src2}}$)
   * `"*"`: Integer multiplication ($r_{\text{src1}} \times r_{\text{src2}}$)
   * `"/"`: Exact integer division ($r_{\text{src1}} \div r_{\text{src2}}$)  
     *(Note: `"//"` is NOT valid)*
2. **`src1` (Source Register 1):** Integer in range $[0, 47]$. Must be initialized.
3. **`src2` (Source Register 2):** Integer in range $[0, 47]$. Must be initialized.
4. **`dst` (Destination Register):** Integer in range $[0, 47]$. Overwrites previous value.

### Execution Semantics
* **Read-Before-Write:** Operands are read from `src1` and `src2` before the write to `dst` occurs.
* **In-Place Reuse:** The destination register `dst` may be identical to `src1` or `src2`:
  * `["+", 0, 0, 0]` doubles the value of $r_0$ in-place ($r_0 \leftarrow r_0 + r_0$).
  * `["/", 0, 0, 1]` generates constant $1$ using self-division ($r_1 \leftarrow r_0 / r_0$).
* **Target Satisfaction:** Targets only need to be produced in any register **at least once** during execution. Once a target is hit, that register may be freely overwritten by subsequent instructions without losing credit.

---

## 3. Strict Arithmetic Constraints

Any violation immediately invalidates the entire submission.

1. **Strictly Positive Integers ($\mathbb{Z}^+$):**
   * Every intermediate and final value stored in any register must be an integer strictly greater than zero ($> 0$).
   * Non-positive results ($0$ or negative values) cause an immediate failure.
2. **Subtraction Constraint:**
   * `["-", src1, src2, dst]` is legal if and only if:
     $$r_{\text{src1}} > r_{\text{src2}}$$
   * Results where $r_{\text{src1}} \le r_{\text{src2}}$ cause an immediate failure.
3. **Exact Division Constraint:**
   * `["/", src1, src2, dst]` is legal if and only if division yields **zero remainder**:
     $$r_{\text{src1}} \pmod{r_{\text{src2}}} = 0$$
   * Inexact divisions, fractional quotients, or division by zero cause an immediate failure.
4. **Value Limit Ceiling:**
   * Each puzzle defines an integer `value_limit` (ranging up to $\sim 10^{18}$).
   * No register value may exceed this limit at any point during execution.

---

## 4. Official Baseline Analysis (`sample_submission.csv`)

Analyzing the official Kaggle sample submission reveals the construction strategy used by the benchmark:

* **Total Baseline Instructions:** **1,490,379** (Average: 1,490.38 / puzzle, Median: **1,596**, Min: 3, Max: 2,873).
* **Register Usage:** The baseline uses at most **register index 9** (10 registers total). Registers 10–47 are never touched.
* **Operator Breakdown:**
  * `+` (Addition): **1,462,481** (98.1%)
  * `*` (Multiplication): **26,898** (1.8%)
  * `/` (Division): **1,000** (0.07% — exactly 1 division per puzzle)
  * `-` (Subtraction): **0** (0%)
* **Baseline Mechanism:**
  1. The baseline uses self-division on register 0 (`["/", 0, 0, 1]`) once to produce the integer `1`.
  2. It synthesizes large targets using binary decomposition (doubling with `+` and adding `1`).
  3. It **completely ignores subtraction** and makes virtually no attempt to share intermediate values across multiple targets.
  4. This highlights immense room for optimization: top solutions achieve **~130,000 instructions** (~91% reduction) by exploiting shared value DAGs and inverse operations (`-`, `/`).

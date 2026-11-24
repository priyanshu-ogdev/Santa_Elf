"""
src/solvers/dag_synthesizer.py
Multi-engine shared-DAG solver for the Santa 2026 Elf Calculator.

Combines five complementary synthesis engines over a shared value DAG:
  1. Exact Dictionary & Active-DAG 0/1-op lookup (via C99/Python DictEngine)
  2. Smooth-Neighbor factoring (t +- 28, seeds, 2^k) + bounded Pollard Rho +
     Byte-Pair Encoding (BPE) factor-multiset CSE across targets
  3. Target-to-Target Anchor quotients & rational-ratio latent common roots
  4. Recursive Two-Pointer Product Residual (depth <= 2) + Scaled 20-bit Split
  5. Multi-Radix Horner (B in {8, 10, 12, 16, 32, 64}) & Squaring-Backbone NAF ceiling

Optimization stages:
  - Multi-ordering initial DAG construction (scoring strictly by true marginal
    added DAG nodes beyond the already-selected DAG)
  - Monotone Rip-and-Reroute iterative refinement across targets
  - Optional tight bounded SciPy HiGHS MILP polish (skipped gracefully if unavailable)
  - Sethi-Ullman / Subtree-Affinity 48-register compilation (`compile_dag`)
"""
from __future__ import annotations
import bisect
import math
from collections import Counter, defaultdict
from typing import Optional

from src.utils import Puzzle, Instruction, apply_op
from src.solvers.fast_engine import DictEngine
from src.dag_compiler import compile_dag


# ---------------------------------------------------------------------------
# Primes & Fast Hybrid Factoring (Trial <= 2000 + Miller-Rabin + Pollard Rho)
# ---------------------------------------------------------------------------

def _sieve_primes(limit: int) -> list[int]:
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i, v in enumerate(sieve) if v]


_SMALL_PRIMES = _sieve_primes(2000)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    for c in (1, 2, 3):
        x = y = 2
        d = 1
        steps = 0
        while d == 1 and steps < 2500:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
            steps += 1
        if 1 < d < n:
            return d
    return n


def _factor_hybrid(n: int, rho_limit: int = 10**9) -> tuple[list[int], int]:
    """
    Factor `n` using primes <= 2000 plus bounded Pollard's Rho up to `rho_limit`.
    Returns (list_of_small_or_medium_factors, remaining_unfactored).
    """
    facs: list[int] = []
    for p in _SMALL_PRIMES:
        if p * p > n:
            if 1 < n <= 50000:
                facs.append(n)
                n = 1
            break
        while n % p == 0:
            facs.append(p)
            n //= p

    if 1 < n <= rho_limit:
        stk = [n]
        rem_out = 1
        while stk:
            m = stk.pop()
            if m <= 1:
                continue
            if m <= 50000 or _is_prime(m):
                if m <= 50000:
                    facs.append(m)
                else:
                    rem_out *= m
            else:
                d = _pollard_rho(m)
                if d == m:
                    rem_out *= m
                else:
                    stk.append(d)
                    stk.append(m // d)
        return facs, rem_out
    return facs, n


# ---------------------------------------------------------------------------
# Value-Level Helpers: Sub-DAG Extraction + NAF + Radix Horner
# ---------------------------------------------------------------------------

def _sub_dag_new_nodes(
    ops_list: list[tuple[str, int, int, int]],
    active_defs: dict[int, tuple[str, int, int]],
    de: DictEngine,
) -> Optional[dict[int, tuple[str, int, int]]]:
    """
    Resolve missing ancestors from `de` (dictionary) beyond `active_defs`.
    Returns `{v: (op, a, b)}` of newly required nodes, or None if invalid/cyclic.
    """
    added: dict[int, tuple[str, int, int]] = {}
    local_edges: dict[int, tuple[str, int, int]] = {}
    vl = de.vl
    for op, a, b, res in ops_list:
        if apply_op(op, a, b, vl) != res:
            return None
        local_edges[res] = (op, a, b)

    seeds = de.seeds_set
    seen = seeds | set(active_defs.keys())
    stk = [res for (_, _, _, res) in ops_list]
    for _, a, b, _ in ops_list:
        stk.append(a)
        stk.append(b)

    while stk:
        u = stk.pop()
        if u in seen or u in added:
            continue
        if u in local_edges:
            op, a, b = local_edges[u]
        else:
            ddef = de.get_dict_def(u)
            if ddef is None:
                return None
            op, a, b = ddef
            if apply_op(op, a, b, vl) != u:
                return None
        added[u] = (op, a, b)
        if a not in seen and a not in added:
            stk.append(a)
        if b not in seen and b not in added:
            stk.append(b)

    # Strict cycle check within `added`
    state: dict[int, int] = {}
    for start in added:
        if start in state:
            continue
        path_stk = [(start, 0)]
        while path_stk:
            node, ph = path_stk.pop()
            if node not in added:
                continue
            if ph == 0:
                st = state.get(node)
                if st == 1:
                    continue
                if st == 0:
                    return None
                state[node] = 0
                path_stk.append((node, 1))
                _, pa, pb = added[node]
                if pa in added:
                    path_stk.append((pa, 0))
                if pb in added and pb != pa:
                    path_stk.append((pb, 0))
            else:
                state[node] = 1
    return added


def _naf_ops(x: int, vl: int, powers: set[int]) -> list[tuple[str, int, int, int]]:
    """Build `x` as signed sum/diff of powers of 2 from `powers` (binary fallback if needed)."""
    if x <= 0:
        return []
    terms: list[tuple[int, int]] = []
    n = x
    pos = 0
    while n > 0:
        if n & 1:
            d = 2 - (n & 3)
            n -= d
            terms.append((pos, d))
        n >>= 1
        pos += 1
    top_pwr = 1 << terms[-1][0]
    if top_pwr > vl or top_pwr not in powers:
        terms = [(i, 1) for i in range(x.bit_length()) if (x >> i) & 1]
    if (1 << terms[-1][0]) not in powers:
        return []
    ops: list[tuple[str, int, int, int]] = []
    acc = 1 << terms[-1][0]
    for pos, sgn in reversed(terms[:-1]):
        pwr = 1 << pos
        if pwr not in powers:
            return []
        nxt = acc + pwr if sgn > 0 else acc - pwr
        if nxt <= 0 or nxt > vl:
            return []
        ops.append(("+" if sgn > 0 else "-", acc, pwr, nxt))
        acc = nxt
    return ops if acc == x else []


def _radix_horner_ops(x: int, B: int, de: DictEngine) -> list[tuple[str, int, int, int]]:
    """Build `x` via signed-digit Horner in base B using digits already in `de`."""
    vl = de.vl
    if x <= B or not de.has(B):
        return []
    half = B // 2
    digits: list[int] = []
    n = x
    while n > 0:
        r = n % B
        if r > half:
            r -= B
        digits.append(r)
        n = (n - r) // B
    msb = list(reversed(digits))
    if not msb or msb[0] <= 0:
        return []
    acc = msb[0]
    if acc not in de.seeds_set and not de.has(acc):
        return []
    ops: list[tuple[str, int, int, int]] = []
    for d in msb[1:]:
        nxt = acc * B
        if nxt > vl:
            return []
        ops.append(("*", acc, B, nxt))
        acc = nxt
        if d > 0:
            if d not in de.seeds_set and not de.has(d):
                return []
            nxt2 = acc + d
            if nxt2 > vl:
                return []
            ops.append(("+", acc, d, nxt2))
            acc = nxt2
        elif d < 0:
            ad = -d
            if ad not in de.seeds_set and not de.has(ad):
                return []
            if acc <= ad:
                return []
            nxt2 = acc - ad
            ops.append(("-", acc, ad, nxt2))
            acc = nxt2
    return ops if acc == x else []


# ---------------------------------------------------------------------------
# Smooth-Neighbor + Byte-Pair Encoding (BPE) Prime-Multiset CSE
# ---------------------------------------------------------------------------

def _precompute_smooth_bpe(
    targets: list[int], de: DictEngine
) -> dict[int, list[list[tuple[str, int, int, int]]]]:
    """
    Find smooth-neighbor factorizations (`t + k` with C99-screened shifts |k|<=256
    plus seeds) and run BPE factor-multiset CSE so targets sharing prime blocks reuse them.
    """
    vl = de.vl
    off_lim = int(de.profile.get("smooth_offset_lim", 64))
    raw_k = [0]
    for d in range(1, off_lim + 1):
        if d in de.seeds_set or de.has(d):
            raw_k.extend([d, -d])
    for s in de.seeds:
        if s > off_lim:
            raw_k.extend([s, -s])

    target_best: dict[int, tuple[int, int, Counter[int], list[tuple[str, int, int, int]]]] = {}
    for t in targets:
        # Use C99 trial-division screener to quickly isolate top 18 smoothest shifts
        k_short = de.screen_smooth_offsets(t, raw_k, _SMALL_PRIMES, top_m=18)
        cands = []
        seen_k = set()
        for k in k_short:
            if k in seen_k:
                continue
            seen_k.add(k)
            cand = t + k
            if cand <= 0 or cand > vl:
                continue
            if k != 0 and abs(k) not in de.seeds_set and not de.has(abs(k)):
                continue
            facs, rem = _factor_hybrid(cand)
            if rem > 1 and rem not in de.seeds_set and not de.has(rem) and rem > 10**8:
                continue
            sc = len(facs) + (0 if rem == 1 else (1 if de.has(rem) else 6)) + (0 if k == 0 else 1)
            cands.append((sc, k, cand, facs, rem))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        _, k, cand, facs, rem = cands[0]
        counts = Counter(facs)
        parts: list[int] = []
        ops: list[tuple[str, int, int, int]] = []
        ok = True
        for p, e in counts.items():
            if not de.has(p) and p not in de.seeds_set:
                np = _naf_ops(p, vl, de.powers)
                if not np:
                    ok = False
                    break
                ops.extend(np)
            pe = p ** e
            if pe > vl:
                ok = False
                break
            if not de.has(pe) and e > 1:
                res_p: Optional[int] = None
                base = p
                kk = e
                while kk > 0:
                    if kk & 1:
                        if res_p is None:
                            res_p = base
                        else:
                            nv = res_p * base
                            if nv > vl:
                                ok = False
                                break
                            if not de.has(nv):
                                ops.append(("*", res_p, base, nv))
                            res_p = nv
                    kk >>= 1
                    if kk > 0:
                        nb = base * base
                        if nb > vl:
                            ok = False
                            break
                        if not de.has(nb):
                            ops.append(("*", base, base, nb))
                        base = nb
                if not ok:
                    break
            parts.append(pe)
        if not ok or not parts:
            continue
        if rem > 1:
            if not de.has(rem) and rem not in de.seeds_set:
                tp = de.two_pointer(rem, K=4)
                added_rem = False
                for d1, d2, prod, gap, sgn in tp:
                    if gap == 0 or de.has(gap):
                        if not de.has(prod):
                            ops.append(("*", d1, d2, prod))
                        if gap > 0:
                            ops.append(("+" if sgn == 1 else "-", prod, gap, rem))
                        added_rem = True
                        break
                if not added_rem:
                    nr = _naf_ops(rem, vl, de.powers)
                    if not nr:
                        continue
                    ops.extend(nr)
            parts.append(rem)
        target_best[t] = (k, cand, Counter(parts), ops)

    # Byte-Pair Encoding CSE over factor multisets
    multisets = {t: info[2].copy() for t, info in target_best.items()}
    bpe_ops: dict[int, tuple[str, int, int, int]] = {}
    for _ in range(25):
        pair_freq: Counter[tuple[int, int]] = Counter()
        for ms in multisets.values():
            elems = sorted(ms.keys())
            for i, a in enumerate(elems):
                if ms[a] >= 2 and a * a <= vl:
                    pair_freq[(a, a)] += 1
                for b in elems[i + 1 :]:
                    if a * b <= vl:
                        pair_freq[(a, b)] += 1
        if not pair_freq:
            break
        (pa, pb), freq = pair_freq.most_common(1)[0]
        if freq < 2:
            break
        merged = pa * pb
        bpe_ops[merged] = ("*", pa, pb, merged)
        for ms in multisets.values():
            if pa == pb:
                c = ms[pa] // 2
                if c > 0:
                    ms[pa] -= 2 * c
                    if ms[pa] == 0:
                        del ms[pa]
                    ms[merged] += c
            elif ms[pa] > 0 and ms[pb] > 0:
                c = min(ms[pa], ms[pb])
                ms[pa] -= c
                ms[pb] -= c
                if ms[pa] == 0:
                    del ms[pa]
                if ms[pb] == 0:
                    del ms[pb]
                ms[merged] += c

    out: dict[int, list[list[tuple[str, int, int, int]]]] = defaultdict(list)
    for t, (k, cand, _, base_ops) in target_best.items():
        ms = multisets[t]
        ops = list(base_ops)
        flat: list[int] = []
        seen_b: set[int] = set()
        for val, cnt in sorted(ms.items()):
            flat.extend([val] * cnt)
            stk = [val]
            while stk:
                u = stk.pop()
                if u in seen_b or u not in bpe_ops:
                    continue
                seen_b.add(u)
                op_e = bpe_ops[u]
                ops.append(op_e)
                stk.append(op_e[1])
                stk.append(op_e[2])
        flat.sort()
        acc = flat[0]
        ok = True
        for f in flat[1:]:
            nv = acc * f
            if nv > vl:
                ok = False
                break
            ops.append(("*", acc, f, nv))
            acc = nv
        if ok and acc == cand:
            if k != 0:
                ops.append(("-" if k > 0 else "+", cand, abs(k), t))
            out[t].append(ops)
    return out


# ---------------------------------------------------------------------------
# Latent Common Root Discovery
# ---------------------------------------------------------------------------

def _discover_latent_roots(targets: list[int], vl: int) -> list[int]:
    """
    Discover hidden roots H where pairs satisfy A ≈ d*H, B ≈ n*H (2 <= d <= 32)
    OR share nontrivial pairwise GCDs `gcd(A +- k1, B +- k2) > 100`.
    """
    roots: Counter[int] = Counter()
    ts = sorted(targets)
    n_t = len(ts)
    for i, A in enumerate(ts):
        if A <= 200:
            continue
        for B in ts[i + 1 : min(n_t, i + 12)]:
            g0 = math.gcd(A, B)
            if 100 < g0 < A:
                roots[g0] += 4
            for off_a in (-1, 0, 1):
                for off_b in (-1, 0, 1):
                    if off_a == 0 and off_b == 0:
                        continue
                    ga = math.gcd(A + off_a, B + off_b)
                    if 100 < ga < A:
                        roots[ga] += 2
            for d in range(2, 33):
                n = round(B * d / A)
                if n <= d or math.gcd(n, d) != 1:
                    continue
                H = round(A / d)
                if H <= 100 or d * H > vl or n * H > vl:
                    continue
                if abs(A - d * H) + abs(B - n * H) < max(2000, A // 2000):
                    roots[H] += 1
    return [h for h, _ in roots.most_common(10)]


# ---------------------------------------------------------------------------
# Marginal Synthesis Over All Engines
# ---------------------------------------------------------------------------

def _synthesize_best_marginal(
    v: int,
    active_defs: dict[int, tuple[str, int, int]],
    active_sorted: list[int],
    de: DictEngine,
    *,
    depth: int = 2,
    extra_cands: Optional[list[list[tuple[str, int, int, int]]]] = None,
    recipe_collector: Optional[list[dict[int, tuple[str, int, int]]]] = None,
) -> Optional[dict[int, tuple[str, int, int]]]:
    """
    Evaluate candidate recipes for `v`, picking the one minimizing newly added DAG
    nodes beyond `active_defs`. Optionally appends compact candidates to `recipe_collector`.
    """
    seeds = de.seeds_set
    vl = de.vl
    if v in seeds or v in active_defs:
        return {}

    best_added: Optional[dict[int, tuple[str, int, int]]] = None

    def consider_ops(ops_list: Optional[list[tuple[str, int, int, int]]]) -> None:
        nonlocal best_added
        if ops_list is None:
            return
        nd = _sub_dag_new_nodes(ops_list, active_defs, de)
        if nd is None:
            return
        if recipe_collector is not None and len(nd) <= 28:
            recipe_collector.append(nd)
        if best_added is None or len(nd) < len(best_added):
            best_added = nd

    # 0. Direct dictionary lookup
    if de.has(v):
        ddef = de.get_dict_def(v)
        if ddef is not None:
            consider_ops([(ddef[0], ddef[1], ddef[2], v)])

    # 1. 1-op from active_defs directly
    active_set = set(active_sorted)
    for a in active_sorted:
        if a < v:
            b = v - a
            if b in active_set:
                consider_ops([("+", a, b, v)])
                if best_added is not None and len(best_added) <= 1 and recipe_collector is None:
                    return best_added
            if a > 1 and v % a == 0 and (v // a) in active_set:
                consider_ops([("*", a, v // a, v)])
                if best_added is not None and len(best_added) <= 1 and recipe_collector is None:
                    return best_added
        elif a > v:
            b = a - v
            if b in active_set:
                consider_ops([("-", a, b, v)])
                if best_added is not None and len(best_added) <= 1 and recipe_collector is None:
                    return best_added
            if v > 0 and a % v == 0 and (a // v) in active_set:
                consider_ops([("/", a, a // v, v)])
                if best_added is not None and len(best_added) <= 1 and recipe_collector is None:
                    return best_added

    # 2. 1-op combining active_defs x DictEngine
    for a in active_sorted:
        if a < v:
            b = v - a
            if de.has(b):
                consider_ops([("+", a, b, v)])
            if a > 1 and v % a == 0 and de.has(v // a):
                consider_ops([("*", a, v // a, v)])
        elif a > v:
            b = a - v
            if de.has(b):
                consider_ops([("-", a, b, v)])
            if v > 0 and a % v == 0 and de.has(a // v):
                consider_ops([("/", a, a // v, v)])
        if a > 1 and v * a <= vl and de.has(v * a):
            consider_ops([("/", v * a, a, v)])

    # 3. Precomputed smooth-BPE candidates
    if extra_cands:
        for ops in extra_cands:
            consider_ops(ops)

    # 4. Multi-Radix Horner (B in {8, 10, 12, 16, 32, 64}) & Squaring NAF / Chunk ceiling
    for B in (16, 32, 64, 10, 12, 8):
        if B < v and de.has(B):
            hops = _radix_horner_ops(v, B, de)
            if hops:
                consider_ops(hops)

    nops = _naf_ops(v, vl, de.powers)
    if nops:
        consider_ops(nops)
    elif de.powers:
        S = max(p.bit_length() - 1 for p in de.powers)
        bp = 1 << max(1, S - 1)
        if bp in de.powers and bp > 1:
            q, r = divmod(v, bp)
            if r > bp // 2 and (q + 1) * bp <= vl:
                q += 1
                r = q * bp - v
                sgn = "-"
            else:
                sgn = "+"
            if 0 < q * bp <= vl:
                qo = _naf_ops(q, vl, de.powers)
                ro = _naf_ops(r, vl, de.powers) if r > 0 else []
                if (q == 1 or qo) and (r == 0 or ro):
                    sub = list(qo)
                    if q > 1:
                        sub.append(("*", q, bp, q * bp))
                    if r > 0:
                        sub.extend(ro)
                        sub.append((sgn, q * bp, r, v))
                    consider_ops(sub)

    if depth <= 0 or (best_added is not None and len(best_added) <= 2 and recipe_collector is None):
        return best_added

    # 5. Active-node * D binary search (anchors cost 0!)
    D_s = de.D_sorted
    nD = len(D_s)
    anchor_proposals = []
    cand_anchors = [a for a in active_sorted if 1 < a < v and a not in de.powers]
    if len(cand_anchors) > 45:
        cand_anchors = cand_anchors[-45:]
    for a in cand_anchors:
        q_int = v // a
        for qq in (q_int, q_int + 1):
            if qq >= 1 and qq * a <= vl and (qq == 1 or de.has(qq) or qq <= 128):
                prod = qq * a
                anchor_proposals.append((abs(v - prod), a, qq, prod, "+" if v >= prod else "-"))
        pos = bisect.bisect_left(D_s, q_int)
        for off in (-1, 0, 1):
            if 0 <= pos + off < nD:
                qq = D_s[pos + off]
                prod = a * qq
                if 0 < prod <= vl:
                    anchor_proposals.append((abs(v - prod), a, qq, prod, "+" if v >= prod else "-"))

    anchor_proposals.sort(key=lambda x: x[0])
    seen_prods: set[int] = set()
    tried_ap = 0
    for gap, a, qq, prod, sgn in anchor_proposals:
        if prod in seen_prods:
            continue
        seen_prods.add(prod)
        tried_ap += 1
        if tried_ap > 4:
            break
        if qq == 1:
            sub = []
        elif de.has(qq):
            sub = [("*", a, qq, prod)]
        else:
            qops = _naf_ops(qq, vl, de.powers)
            if not qops:
                continue
            sub = qops + [("*", a, qq, prod)]
        if gap == 0:
            consider_ops(sub)
        elif gap in active_set or de.has(gap):
            consider_ops(sub + [(sgn, prod, gap, v)])
        elif depth >= 2:
            g_added = _synthesize_best_marginal(gap, active_defs, active_sorted, de, depth=0)
            if g_added is not None:
                consider_ops(
                    list(sub)
                    + [(op, ax, bx, rx) for rx, (op, ax, bx) in g_added.items()]
                    + [(sgn, prod, gap, v)]
                )

    # 6. Scaled 20-bit multiplier split for >= 42-bit targets (top 6 fastest candidates!)
    if depth >= 2 and v.bit_length() >= 42 and de.split_muls:
        q_cands = []
        for m in de.split_muls[:8]:
            q, r = divmod(v, m)
            if r > m // 2 and (q + 1) * m <= vl:
                q += 1
                r = q * m - v
                sgn = "-"
            else:
                sgn = "+"
            if q > 100:
                tp_q = de.two_pointer(q, K=2)
                if tp_q:
                    d1q, d2q, pq, gq, sgq = tp_q[0]
                    q_cands.append((gq.bit_length() + r.bit_length(), m, q, r, sgn, tp_q[0]))
        q_cands.sort(key=lambda x: x[0])
        for _, m, q, r, sgn, (d1q, d2q, pq, gq, sgq) in q_cands[:2]:
            sub_q: Optional[list[tuple[str, int, int, int]]] = (
                [("*", d1q, d2q, pq)] if not de.has(pq) else []
            )
            if gq > 0:
                if de.has(gq):
                    sub_q.append(("+" if sgq == 1 else "-", pq, gq, q))
                else:
                    ga = _synthesize_best_marginal(gq, active_defs, active_sorted, de, depth=0)
                    if ga is not None:
                        sub_q.extend((op, ax, bx, rx) for rx, (op, ax, bx) in ga.items())
                        sub_q.append(("+" if sgq == 1 else "-", pq, gq, q))
                    else:
                        sub_q = None
            if sub_q is not None and q * m <= vl:
                prod = m * q
                sub_q.append(("*", m, q, prod))
                if r == 0:
                    consider_ops(sub_q)
                elif de.has(r) or r in active_set:
                    consider_ops(sub_q + [(sgn, prod, r, v)])
                else:
                    ra = _synthesize_best_marginal(r, active_defs, active_sorted, de, depth=0)
                    if ra is not None:
                        consider_ops(
                            sub_q
                            + [(op, ax, bx, rx) for rx, (op, ax, bx) in ra.items()]
                            + [(sgn, prod, r, v)]
                        )

    # 7. Bidirectional C99 MITM quotient-remainder probe over D_local + Universal U
    if depth >= 2 and v > 100:
        mitm_hits = de.mitm_probe(v, active_sorted, K=16)
        seen_mh: set[tuple[int, int]] = set()
        mh_rec = 0
        for a, q, r, sgn, _ in mitm_hits:
            if (a, q) in seen_mh:
                continue
            seen_mh.add((a, q))
            prod = a * q
            if prod <= 0 or prod > vl:
                continue
            op_r = "+" if sgn == 1 else "-"
            sub_m: list[tuple[str, int, int, int]] = [] if prod in active_set else [("*", a, q, prod)]
            if r == 0:
                consider_ops(sub_m if sub_m else [("*", a, q, prod)])
            elif r in active_set or de.has(r):
                consider_ops(sub_m + [(op_r, prod, r, v)])
            elif mh_rec < 4:
                mh_rec += 1
                r_added = _synthesize_best_marginal(r, active_defs, active_sorted, de, depth=1)
                if r_added is not None:
                    consider_ops(
                        list(sub_m)
                        + [(op, ax, bx, rx) for rx, (op, ax, bx) in r_added.items()]
                        + [(op_r, prod, r, v)]
                    )

    # 8. Two-Pointer Product Residual (C99 or Python)
    if v > 100:
        tp_hits = de.two_pointer(v, K=12)
        seen_tp: set[int] = set()
        tp_rec = 0
        for d1, d2, prod, gap, sgn in tp_hits:
            if prod in seen_tp:
                continue
            seen_tp.add(prod)
            op_r = "+" if sgn == 1 else "-"
            sub = [("*", d1, d2, prod)] if not (prod in active_set or prod in seeds) else []
            if gap == 0:
                consider_ops(sub if sub else [("*", d1, d2, prod)])
            elif gap in active_set or de.has(gap):
                consider_ops(sub + [(op_r, prod, gap, v)])
            elif depth >= 2 and tp_rec < 4:
                tp_rec += 1
                g_added = _synthesize_best_marginal(gap, active_defs, active_sorted, de, depth=1)
                if g_added is not None:
                    consider_ops(
                        list(sub)
                        + [(op, ax, bx, rx) for rx, (op, ax, bx) in g_added.items()]
                        + [(op_r, prod, gap, v)]
                    )

    return best_added


# ---------------------------------------------------------------------------
# Optional Tight Bounded HiGHS MILP Polish
# ---------------------------------------------------------------------------

def _milp_polish(
    targets: list[int],
    seeds_set: set[int],
    cur_defs: dict[int, tuple[str, int, int]],
    candidate_recipes: dict[int, list[dict[int, tuple[str, int, int]]]],
    *,
    max_edges: int = 3500,
    time_limit: float = 2.0,
) -> Optional[dict[int, tuple[str, int, int]]]:
    """
    Formulate a bounded MILP over the union of `cur_defs` and top compact
    target recipes (up to `max_edges` candidate edges) via `scipy.optimize.milp` (HiGHS).
    Returns an improved acyclic `defs` dict, or None if unavailable / slower / larger.
    """
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None

    edge_set: set[tuple[str, int, int, int]] = set()
    for res, (op, a, b) in cur_defs.items():
        edge_set.add((op, a, b, res))

    for t in targets:
        recs = sorted(candidate_recipes.get(t, []), key=len)[:7]
        for rmap in recs:
            for res, (op, a, b) in rmap.items():
                edge_set.add((op, a, b, res))

    if len(edge_set) > max_edges or len(edge_set) <= len(cur_defs):
        return None

    edges = sorted(edge_set)
    all_vals = set(targets)
    for _, a, b, res in edges:
        all_vals.add(a)
        all_vals.add(b)
        all_vals.add(res)
    non_seed = sorted(all_vals - seeds_set)
    val_to_idx = {v: i for i, v in enumerate(non_seed)}
    nv = len(non_seed)
    ne = len(edges)

    out_edges: dict[int, list[int]] = defaultdict(list)
    for ei, (_, _, _, res) in enumerate(edges):
        if res in val_to_idx:
            out_edges[val_to_idx[res]].append(ei)

    banned_cycles: list[list[int]] = []
    for _ in range(8):
        n_vars = nv + ne
        c = np.zeros(n_vars, dtype=float)
        c[:nv] = 1.0
        c[nv:] = 0.001

        lb_v = np.zeros(n_vars)
        ub_v = np.ones(n_vars)
        for t in targets:
            lb_v[val_to_idx[t]] = 1.0

        rows, cols, vals_sp = [], [], []
        b_l, b_u = [], []
        rc = 0

        for vi in range(nv):
            elist = out_edges.get(vi, [])
            if not elist:
                ub_v[vi] = 0.0
                continue
            rows.append(rc)
            cols.append(vi)
            vals_sp.append(1.0)
            for ei in elist:
                rows.append(rc)
                cols.append(nv + ei)
                vals_sp.append(-1.0)
            b_l.append(-np.inf)
            b_u.append(0.0)
            rc += 1

        for ei, (_, a, b, _) in enumerate(edges):
            if a in val_to_idx:
                rows.extend([rc, rc])
                cols.extend([nv + ei, val_to_idx[a]])
                vals_sp.extend([1.0, -1.0])
                b_l.append(-np.inf)
                b_u.append(0.0)
                rc += 1
            if b in val_to_idx and b != a:
                rows.extend([rc, rc])
                cols.extend([nv + ei, val_to_idx[b]])
                vals_sp.extend([1.0, -1.0])
                b_l.append(-np.inf)
                b_u.append(0.0)
                rc += 1

        for cyc in banned_cycles:
            for ei in cyc:
                rows.append(rc)
                cols.append(nv + ei)
                vals_sp.append(1.0)
            b_l.append(-np.inf)
            b_u.append(float(len(cyc) - 1))
            rc += 1

        A = sp.csc_matrix((vals_sp, (rows, cols)), shape=(rc, n_vars))
        res = milp(
            c=c,
            integrality=np.ones(n_vars),
            bounds=Bounds(lb_v, ub_v),
            constraints=LinearConstraint(A, b_l, b_u),
            options={"time_limit": time_limit, "mip_rel_gap": 0.005},
        )
        if res.x is None:
            return None

        sel_edges = [ei for ei in range(ne) if res.x[nv + ei] > 0.5]
        adj: dict[int, tuple[str, int, int]] = {}
        edge_by_res: dict[int, int] = {}
        for ei in sel_edges:
            op, a, b, rv = edges[ei]
            if rv not in adj and rv not in seeds_set:
                adj[rv] = (op, a, b)
                edge_by_res[rv] = ei

        # Check all targets reachable and cycle-free
        state: dict[int, int] = {}
        found_cyc: Optional[list[int]] = None
        path_vals: list[int] = []

        def dfs(u: int) -> bool:
            nonlocal found_cyc
            if u in seeds_set or state.get(u) == 1:
                return False
            if state.get(u) == 0:
                idx0 = path_vals.index(u)
                found_cyc = [edge_by_res[w] for w in path_vals[idx0:]]
                return True
            if u not in adj:
                return False
            state[u] = 0
            path_vals.append(u)
            _, pa, pb = adj[u]
            if dfs(pa) or dfs(pb):
                return True
            path_vals.pop()
            state[u] = 1
            return False

        for t in targets:
            if dfs(t):
                break

        if found_cyc is None:
            # Prune reachable subgraph
            keep: dict[int, tuple[str, int, int]] = {}
            stk = list(targets)
            vis = set(seeds_set)
            ok = True
            while stk:
                u = stk.pop()
                if u in vis:
                    continue
                if u not in adj:
                    ok = False
                    break
                vis.add(u)
                keep[u] = adj[u]
                _, pa, pb = adj[u]
                if pa not in vis:
                    stk.append(pa)
                if pb not in vis:
                    stk.append(pb)
            if ok and len(keep) < len(cur_defs):
                return keep
            return None
        banned_cycles.append(found_cyc)
    return None


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def solve(
    puzzle: Puzzle, *, force_python: bool = False, ram_gb: str | float = "auto"
) -> list[Instruction]:
    """
    Solve `puzzle` using the 6-engine shared-DAG synthesizer, Rip-and-Reroute +
    Pair-Rip LNS, bounded HiGHS MILP polish, Min-Cut/Sethi-Ullman compiler,
    and dead-op cleanup.
    """
    if puzzle.value_limit < 40:
        from src.solvers import tiny_bfs
        return tiny_bfs.solve(puzzle)

    from src.verifier import verify_program
    from src.postpass import remove_dead_ops

    seeds_set = set(puzzle.seeds)
    targets = sorted(set(puzzle.targets) - seeds_set)
    if not targets:
        return []

    with DictEngine(puzzle, force_python=force_python, ram_gb=ram_gb) as de:
        smooth_map = _precompute_smooth_bpe(targets, de)
        latent_roots = _discover_latent_roots(targets, de.vl)

        orderings = [
            latent_roots + sorted(targets, key=lambda t: (t.bit_length(), t)),
            latent_roots + sorted(targets, key=lambda t: (0 if t in smooth_map else 1, t.bit_length(), t)),
            latent_roots + sorted(targets, key=lambda t: (-t.bit_length(), t)),
        ]

        best_defs: Optional[dict[int, tuple[str, int, int]]] = None
        cand_recipes: dict[int, list[dict[int, tuple[str, int, int]]]] = defaultdict(list)

        for order in orderings:
            active_defs: dict[int, tuple[str, int, int]] = {}
            active_sorted = sorted(de.seeds)
            active_set = set(active_sorted)

            for item in order:
                col: list[dict[int, tuple[str, int, int]]] = [] if item in targets else None  # type: ignore[assignment]
                added = _synthesize_best_marginal(
                    item,
                    active_defs,
                    active_sorted,
                    de,
                    depth=2,
                    extra_cands=smooth_map.get(item),
                    recipe_collector=col,
                )
                if col:
                    cand_recipes[item].extend(col[:5])
                if added:
                    active_defs.update(added)
                    active_set.update(added.keys())
                    active_sorted = sorted(active_set)

            if all(t in active_defs or t in seeds_set for t in targets):
                if best_defs is None or len(active_defs) < len(best_defs):
                    best_defs = active_defs

        if best_defs is None:
            from src.solvers import binary_shared
            return binary_shared.solve(puzzle)

        def prune_dag(dmap: dict[int, tuple[str, int, int]], req: list[int] = targets) -> dict[int, tuple[str, int, int]]:
            keep: dict[int, tuple[str, int, int]] = {}
            stk = list(req)
            vis = set(seeds_set)
            while stk:
                u = stk.pop()
                if u in vis:
                    continue
                vis.add(u)
                op, a, b = dmap[u]
                keep[u] = (op, a, b)
                if a not in vis:
                    stk.append(a)
                if b not in vis:
                    stk.append(b)
            return keep

        cur_defs = prune_dag(best_defs)

        # Single-target Monotone Rip-and-Reroute sweeps
        for _ in range(3):
            improved = False
            for t in reversed(targets):
                keep_other = prune_dag(cur_defs, [x for x in targets if x != t])
                if t in keep_other or t in seeds_set:
                    if len(keep_other) < len(cur_defs):
                        cur_defs = keep_other
                        improved = True
                    continue

                other_sorted = sorted(set(keep_other.keys()) | seeds_set)
                col_r: list[dict[int, tuple[str, int, int]]] = []
                new_add = _synthesize_best_marginal(
                    t,
                    keep_other,
                    other_sorted,
                    de,
                    depth=2,
                    extra_cands=smooth_map.get(t),
                    recipe_collector=col_r,
                )
                if col_r:
                    cand_recipes[t].extend(col_r[:4])
                if new_add is not None:
                    cand_defs = dict(keep_other)
                    cand_defs.update(new_add)
                    cand_defs = prune_dag(cand_defs)
                    if len(cand_defs) < len(cur_defs):
                        cur_defs = cand_defs
                        improved = True
            if not improved:
                break

        # 2-Target Pair Rip-and-Reroute LNS (escapes locked shared local minima)
        ts_by_bits = sorted(targets, key=lambda x: (x.bit_length(), x))
        pair_budget = min(len(ts_by_bits) - 1, 20)
        for pi in range(pair_budget):
            ta, tb = ts_by_bits[pi], ts_by_bits[pi + 1]
            keep_other = prune_dag(cur_defs, [x for x in targets if x != ta and x != tb])
            g_ab = math.gcd(ta, tb)
            pair_orders = [(ta, tb), (tb, ta)]
            for ord_pair in pair_orders:
                c_defs = dict(keep_other)
                c_set = set(c_defs.keys()) | seeds_set
                c_sorted = sorted(c_set)
                seq_items = ([g_ab] + list(ord_pair)) if (100 < g_ab < min(ta, tb) and g_ab not in c_set) else list(ord_pair)
                ok_pair = True
                for x in seq_items:
                    if x in c_set:
                        continue
                    add_x = _synthesize_best_marginal(
                        x, c_defs, c_sorted, de, depth=2, extra_cands=smooth_map.get(x)
                    )
                    if add_x is None:
                        ok_pair = False
                        break
                    c_defs.update(add_x)
                    c_set.update(add_x.keys())
                    c_sorted = sorted(c_set)
                if ok_pair and ta in c_set and tb in c_set:
                    c_pruned = prune_dag(c_defs)
                    if len(c_pruned) < len(cur_defs):
                        cur_defs = c_pruned

        n_trials = int(de.profile.get("compiler_trials", 48))
        best_prog = remove_dead_ops(puzzle, compile_dag(puzzle, cur_defs, n_trials=n_trials))

        # Bounded HiGHS MILP polish over union of compact recipes
        milp_defs = _milp_polish(
            targets,
            seeds_set,
            cur_defs,
            cand_recipes,
            max_edges=int(de.profile.get("milp_max_edges", 3500)),
            time_limit=float(de.profile.get("milp_time", 2.0)),
        )
        if milp_defs is not None:
            try:
                milp_prog = remove_dead_ops(puzzle, compile_dag(puzzle, milp_defs, n_trials=n_trials))
                if len(milp_prog) < len(best_prog) and verify_program(puzzle, milp_prog).valid:
                    best_prog = milp_prog
            except Exception:
                pass

        return best_prog

"""
src/solvers/fast_engine.py
High-RAM 128-bit overflow-safe Universal + Local Dictionary & C99 MITM Engine.

Architecture:
  1. Global Parent-Shared Universal `{1}`-Dictionary (`U`):
     Initialized once before `multiprocessing.Pool` `fork()`, staying strictly
     read-only across workers for 1x Copy-on-Write RAM usage on Linux/Kaggle.
     Uses 16-byte packed nodes (`uint64_t val` + 29b/29b/2b/4b `meta`) + 4-byte
     open-addressing index table + 2 MiB L3 duplicate bitset (`< 2^24`).
     Covers 100% of integers < 2^20 plus dense 20..36-bit arithmetic blocks.
  2. Per-Puzzle Local Dictionary (`D_local`):
     Seed-specific products, sums, differences, exact quotients, and squaring
     backbone (`2..2^32`), plus a contiguous sorted scan slice for `O(N)`
     two-pointer product-residual search (`fe_two_pointer`).
  3. Exact C99 Meet-in-the-Middle Probes:
     - `fe_mitm_probe`: tests `T = a * q +- r` over active DAG values (0 marginal
       cost) and low-cost divisors across `D_local U U` in `O(1)` hash lookup.
     - `fe_screen_smooth`: screens hundreds of small shifts `T + k` by trial
       dividing primes <= 2000 in C99 (< 1ms per target).
  4. Pure-Python fallback backend (`_PyDictBackend`) when `gcc` is absent.
"""
from __future__ import annotations
import ctypes
import math
import os
import shutil
import subprocess
import tempfile
from typing import Optional
from src.utils import Puzzle, apply_op


# ---------------------------------------------------------------------------
# Host / Kaggle RAM Auto-Detection & Sizing Profiles
# ---------------------------------------------------------------------------

def detect_host_ram_gb() -> tuple[float, float]:
    """Return (total_ram_gb, avail_ram_gb) cleanly across Linux and Windows."""
    # Linux (/proc/meminfo - Kaggle runs Ubuntu Linux)
    if os.path.exists("/proc/meminfo"):
        try:
            tot_kb = avail_kb = 0
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        tot_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
            if tot_kb > 0:
                if avail_kb <= 0:
                    avail_kb = int(tot_kb * 0.8)
                return tot_kb / (1024 * 1024), avail_kb / (1024 * 1024)
        except Exception:
            pass

    # Windows (GlobalMemoryStatusEx)
    if os.name == "nt":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            ms = _MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullTotalPhys / (1024**3), ms.ullAvailPhys / (1024**3)
        except Exception:
            pass

    return 8.0, 4.0


def get_ram_profile(ram_gb: str | float = "auto") -> dict[str, int | float]:
    """
    Compute balanced C99 Universal `U`, Per-Worker `D_local`, MILP, and Compiler
    budgets. On Kaggle (~29-30 GB RAM, Linux CoW), defaults to high-RAM (~20-24 GB
    aggregate across shared U + 4 parallel workers).
    """
    total_gb, avail_gb = detect_host_ram_gb()
    on_kaggle = os.path.exists("/kaggle")

    if isinstance(ram_gb, (int, float)):
        eff_gb = float(ram_gb)
    elif isinstance(ram_gb, str) and ram_gb != "auto":
        try:
            eff_gb = float(ram_gb)
        except ValueError:
            eff_gb = avail_gb
    else:
        eff_gb = max(avail_gb, 24.0) if (on_kaggle and total_gb >= 24.0) else avail_gb

    if eff_gb >= 18.0:
        # Kaggle 20-24 GB aggregate budget:
        #   U (~300M nodes, HT=2^30 -> ~9.09 GB CoW shared across 4 workers)
        #   + 4 workers x (4M local + 1.5M scan + MILP + Python ~ 2.5-3.5 GB) ~ 12-14 GB
        return {
            "mode": "kaggle_high_ram",
            "eff_gb": eff_gb,
            "univ_cap": 300_000_000,
            "univ_ht_pow": 30,
            "local_cap": 4_000_000,
            "local_ht_pow": 24,
            "local_scan": 1_500_000,
            "per_bucket": 65_000,
            "mitm_div_scan": 450_000,
            "smooth_offset_lim": 256,
            "milp_max_edges": 3500,
            "milp_time": 2.0,
            "compiler_trials": 64,
        }
    if eff_gb >= 6.5:
        return {
            "mode": "medium_ram",
            "eff_gb": eff_gb,
            "univ_cap": 60_000_000,
            "univ_ht_pow": 27,
            "local_cap": 1_500_000,
            "local_ht_pow": 22,
            "local_scan": 600_000,
            "per_bucket": 25_000,
            "mitm_div_scan": 250_000,
            "smooth_offset_lim": 160,
            "milp_max_edges": 2500,
            "milp_time": 1.0,
            "compiler_trials": 48,
        }
    return {
        "mode": "compact_ram",
        "eff_gb": eff_gb,
        "univ_cap": 8_000_000,
        "univ_ht_pow": 25,
        "local_cap": 600_000,
        "local_ht_pow": 21,
        "local_scan": 300_000,
        "per_bucket": 10_000,
        "mitm_div_scan": 150_000,
        "smooth_offset_lim": 96,
        "milp_max_edges": 1500,
        "milp_time": 0.5,
        "compiler_trials": 40,
    }


# ---------------------------------------------------------------------------
# C99 Backend (Packed 16B Nodes + 4B Hash Index + Universal Shared Dict + MITM)
# ---------------------------------------------------------------------------

_C_SOURCE = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef unsigned __int128 u128;

#define EMPTY_PA 0x1FFFFFFFULL
#define OP_ADD 0
#define OP_SUB 1
#define OP_MUL 2
#define OP_DIV 3

static inline uint64_t pack_meta(uint32_t pa, uint32_t pb, uint8_t op, uint8_t cost) {
    return (((uint64_t)pa & 0x1FFFFFFFULL) << 35)
         | (((uint64_t)pb & 0x1FFFFFFFULL) << 6)
         | (((uint64_t)op & 3ULL) << 4)
         | ((uint64_t)(cost > 15 ? 15 : cost) & 15ULL);
}
static inline uint32_t meta_pa(uint64_t m)   { return (uint32_t)((m >> 35) & 0x1FFFFFFFULL); }
static inline uint32_t meta_pb(uint64_t m)   { return (uint32_t)((m >> 6) & 0x1FFFFFFFULL); }
static inline uint8_t  meta_op(uint64_t m)   { return (uint8_t)((m >> 4) & 3ULL); }
static inline uint8_t  meta_cost(uint64_t m) { return (uint8_t)(m & 15ULL); }

static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}
static inline int bl64(uint64_t v) { return v ? (64 - __builtin_clzll(v)) : 0; }

typedef struct {
    uint64_t *vals;
    uint64_t *meta;
    int32_t  *ht;
    uint64_t *seen24;     /* 2^24 bits = 2 MiB fast L3 duplicate filter */
    uint32_t  ht_mask;
    uint32_t  n;
    uint32_t  cap;
    uint64_t  vl;
    int32_t   bucket_cnt[65];
    int32_t   bucket_cap[65];

    /* Sorted contiguous scan slice for O(N) two-pointer & bisect */
    uint64_t *scan_vals;
    uint8_t  *scan_costs;
    uint32_t  n_scan;
} BigDict;

static inline int32_t bd_find(const BigDict *d, uint64_t k) {
    if (!d || k == 0 || k > d->vl) return -1;
    uint32_t h = (uint32_t)hash64(k) & d->ht_mask;
    for (;;) {
        int32_t idx = d->ht[h];
        if (idx < 0) return -1;
        if (d->vals[idx] == k) return idx;
        h = (h + 1) & d->ht_mask;
    }
}

static inline int bd_add(
    BigDict *d, uint64_t v, uint32_t pa, uint32_t pb, uint8_t op, uint8_t cost, int use_cap
) {
    if (v == 0 || v > d->vl || d->n >= d->cap) return 0;
    if (d->seen24 && v < (1ULL << 24)) {
        if (d->seen24[v >> 6] & (1ULL << (v & 63))) return 0;
    }
    int bl = bl64(v);
    if (use_cap && d->bucket_cap[bl] > 0 && d->bucket_cnt[bl] >= d->bucket_cap[bl]) return 0;
    uint32_t h = (uint32_t)hash64(v) & d->ht_mask;
    for (;;) {
        int32_t idx = d->ht[h];
        if (idx < 0) break;
        if (d->vals[idx] == v) return 0;
        h = (h + 1) & d->ht_mask;
    }
    uint32_t idx = d->n++;
    d->vals[idx] = v;
    d->meta[idx] = pack_meta(pa, pb, op, cost);
    d->ht[h] = (int32_t)idx;
    if (d->seen24 && v < (1ULL << 24)) d->seen24[v >> 6] |= (1ULL << (v & 63));
    d->bucket_cnt[bl]++;
    return 1;
}

typedef struct { uint64_t v; uint8_t c; } SortPair;
static int cmp_sp(const void *a, const void *b) {
    uint64_t va = ((const SortPair *)a)->v, vb = ((const SortPair *)b)->v;
    return (va < vb) ? -1 : (va > vb) ? 1 : 0;
}

static void free_bigdict(BigDict *d) {
    if (!d) return;
    free(d->vals);
    free(d->meta);
    free(d->ht);
    free(d->seen24);
    free(d->scan_vals);
    free(d->scan_costs);
    free(d);
}

/* Safe allocation helper: returns NULL cleanly if memory unavailable. */
static BigDict* alloc_bigdict(uint32_t cap, uint32_t ht_pow, uint64_t vl) {
    if (cap > 500000000U) cap = 500000000U;
    BigDict *d = (BigDict *)calloc(1, sizeof(BigDict));
    if (!d) return NULL;
    uint64_t ht_size = 1ULL << ht_pow;
    d->ht_mask = (uint32_t)(ht_size - 1ULL);
    d->cap = cap;
    d->vl = vl;
    d->vals = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    d->meta = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    d->ht   = (int32_t *)malloc((size_t)ht_size * sizeof(int32_t));
    d->seen24 = (uint64_t *)calloc((1U << 18), sizeof(uint64_t));
    if (!d->vals || !d->meta || !d->ht || !d->seen24) {
        free_bigdict(d);
        return NULL;
    }
    memset(d->ht, 0xFF, (size_t)ht_size * sizeof(int32_t));
    return d;
}

/*
 * Global Parent-Shared Read-Only Universal `{1}` Dictionary.
 * Prioritizes full coverage < 2^20 and dense 20..36-bit arithmetic blocks.
 */
EXPORT void* fe_univ_create(uint32_t cap, uint32_t ht_pow) {
    BigDict *d = alloc_bigdict(cap, ht_pow, (1ULL << 61));
    if (!d) return NULL;

    for (int bl = 1; bl <= 64; bl++) {
        if (bl <= 24)      d->bucket_cap[bl] = (int32_t)(cap / 12U);
        else if (bl <= 35) d->bucket_cap[bl] = (int32_t)(cap / 13U);
        else if (bl <= 44) d->bucket_cap[bl] = (int32_t)(cap / 30U);
        else               d->bucket_cap[bl] = (int32_t)(cap / 80U);
    }

    bd_add(d, 1ULL, EMPTY_PA, EMPTY_PA, OP_ADD, 0, 0);

    /* Explicit squaring backbone: 2, 4, 16, 256, 65536, 2^32 */
    int32_t i1 = bd_find(d, 1ULL);
    if (i1 >= 0) bd_add(d, 2ULL, (uint32_t)i1, (uint32_t)i1, OP_ADD, 1, 0);
    uint64_t sq = 2ULL;
    while (sq <= d->vl && sq <= d->vl / sq) {
        int32_t isq = bd_find(d, sq);
        if (isq < 0) break;
        uint8_t c_next = meta_cost(d->meta[isq]) + 1;
        sq = sq * sq;
        bd_add(d, sq, (uint32_t)isq, (uint32_t)isq, OP_MUL, c_next, 0);
    }

    /* Depth 1..6 exhaustive universal DAG BFS */
    uint32_t ls = 0, le = d->n, d5_end = 0;
    for (int depth = 1; depth <= 6 && d->n < cap; depth++) {
        for (uint32_t i = ls; i < le && d->n < cap; i++) {
            uint64_t a = d->vals[i];
            for (uint32_t j = 0; j < le && d->n < cap; j++) {
                if (j > i && j >= ls) continue;
                uint64_t b = d->vals[j];
                uint8_t c_est = (uint8_t)(depth <= 15 ? depth : 15);
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)d->vl) bd_add(d, (uint64_t)p, i, j, OP_MUL, c_est, 0);
                u128 s = (u128)a + (u128)b;
                if (s <= (u128)d->vl) bd_add(d, (uint64_t)s, i, j, OP_ADD, c_est, 0);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                    bd_add(d, hi - lo, phi, plo, OP_SUB, c_est, 0);
                    if (lo > 1 && hi % lo == 0) bd_add(d, hi / lo, phi, plo, OP_DIV, c_est, 0);
                }
            }
        }
        ls = le; le = d->n;
        if (depth == 5) d5_end = d->n;
    }

    uint32_t d6_end = d->n;
    /* Phase A: d6 x <= d5 (dense low-DAG-size expansions, guarantees <2^20 ~100%) */
    for (uint32_t i = d5_end; i < d6_end && d->n < cap; i++) {
        uint64_t a = d->vals[i];
        for (uint32_t j = 0; j < d5_end && d->n < cap; j++) {
            uint64_t b = d->vals[j];
            u128 p = (u128)a * (u128)b;
            if (p <= (u128)d->vl) bd_add(d, (uint64_t)p, i, j, OP_MUL, 7, 1);
            u128 s = (u128)a + (u128)b;
            if (s <= (u128)d->vl) bd_add(d, (uint64_t)s, i, j, OP_ADD, 7, 1);
            if (a != b) {
                uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                bd_add(d, hi - lo, phi, plo, OP_SUB, 7, 1);
            }
        }
    }

    /* Phase B: d6 x d6 filling 24..36-bit dense meet-in-the-middle band */
    for (uint32_t i = d5_end; i < d6_end && d->n < cap; i++) {
        uint64_t a = d->vals[i];
        if (bl64(a) > 36) continue;
        for (uint32_t j = i; j < d6_end && d->n < cap; j++) {
            uint64_t b = d->vals[j];
            if (bl64(b) > 36) continue;
            u128 p = (u128)a * (u128)b;
            if (p <= (u128)d->vl) bd_add(d, (uint64_t)p, i, j, OP_MUL, 8, 1);
            u128 s = (u128)a + (u128)b;
            if (s <= (u128)d->vl) bd_add(d, (uint64_t)s, i, j, OP_ADD, 8, 1);
            if (a != b) {
                uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                bd_add(d, hi - lo, phi, plo, OP_SUB, 8, 1);
            }
        }
    }

    /* Free seen24 before fork to slim Read-Only U footprint (already indexed in ht) */
    free(d->seen24);
    d->seen24 = NULL;
    return d;
}

EXPORT void fe_dict_free(void *p) { free_bigdict((BigDict *)p); }
EXPORT uint32_t fe_dict_size(void *p) { return p ? ((BigDict *)p)->n : 0U; }
EXPORT uint32_t fe_scan_size(void *p) { return p ? ((BigDict *)p)->n_scan : 0U; }

EXPORT int32_t fe_dict_lookup(void *p, uint64_t v) {
    return bd_find((const BigDict *)p, v);
}

EXPORT void fe_dict_get(
    void *p, uint32_t idx, uint64_t *val, uint64_t *va, uint64_t *vb, int *op, int *cost
) {
    const BigDict *d = (const BigDict *)p;
    if (!d || idx >= d->n) { *op = -1; return; }
    uint64_t m = d->meta[idx];
    uint32_t pa = meta_pa(m), pb = meta_pb(m);
    *val = d->vals[idx];
    *cost = (int)meta_cost(m);
    if (pa == EMPTY_PA) {
        *op = 0; *va = 0; *vb = 0;
    } else {
        *op = (int)meta_op(m) + 1; /* 1=+, 2=-, 3=*, 4=/ */
        *va = d->vals[pa];
        *vb = d->vals[pb];
    }
}

EXPORT uint32_t fe_dict_copy_scan(void *p, uint64_t *out) {
    const BigDict *d = (const BigDict *)p;
    for (uint32_t i = 0; i < d->n_scan; i++) out[i] = d->scan_vals[i];
    return d->n_scan;
}

/*
 * Per-puzzle Local Seed Dictionary `D_local`.
 * Builds balanced bit-buckets across seeds, squaring powers, small constants 1..256,
 * and cross expansions, plus builds a sorted `scan_cap` contiguous array for 2-ptr.
 */
EXPORT void* fe_local_create(
    const uint64_t *seeds, int ns, uint64_t vl, uint32_t cap, uint32_t ht_pow,
    uint32_t per_b, uint32_t scan_cap
) {
    BigDict *d = alloc_bigdict(cap, ht_pow, vl);
    if (!d) return NULL;
    for (int bl = 1; bl <= 64; bl++) d->bucket_cap[bl] = (int32_t)per_b;

    for (int i = 0; i < ns; i++) {
        bd_add(d, seeds[i], EMPTY_PA, EMPTY_PA, OP_ADD, 0, 0);
    }
    if (bd_find(d, 1ULL) < 0 && ns > 0) {
        int32_t is0 = bd_find(d, seeds[0]);
        if (is0 >= 0) bd_add(d, 1ULL, (uint32_t)is0, (uint32_t)is0, OP_DIV, 1, 0);
    }

    /* Squaring backbone: 2, 4, 16, 256, 65536, 2^32 */
    int32_t i1 = bd_find(d, 1ULL);
    if (2ULL <= vl && bd_find(d, 2ULL) < 0 && i1 >= 0) {
        bd_add(d, 2ULL, (uint32_t)i1, (uint32_t)i1, OP_ADD, meta_cost(d->meta[i1]) + 1, 0);
    }
    uint64_t sq = 2ULL;
    while (sq <= vl && sq <= vl / sq) {
        int32_t idx = bd_find(d, sq);
        if (idx < 0) break;
        uint8_t co = meta_cost(d->meta[idx]);
        sq = sq * sq;
        bd_add(d, sq, (uint32_t)idx, (uint32_t)idx, OP_MUL, co + 1, 0);
    }

    /* Small doubling 1..1024 */
    uint64_t cur = 1ULL;
    while (cur <= 1024ULL && cur <= vl / 2ULL) {
        int32_t idx = bd_find(d, cur);
        if (idx < 0) break;
        bd_add(d, cur * 2ULL, (uint32_t)idx, (uint32_t)idx, OP_ADD, meta_cost(d->meta[idx]) + 1, 0);
        cur *= 2ULL;
    }

    /* Pairwise power-of-2 products */
    for (int pass = 0; pass < 3; pass++) {
        uint32_t lim = d->n;
        for (uint32_t i = 0; i < lim && d->n < cap; i++) {
            uint64_t a = d->vals[i];
            if ((a & (a - 1)) != 0) continue;
            uint8_t ca = meta_cost(d->meta[i]);
            for (uint32_t j = i; j < lim && d->n < cap; j++) {
                uint64_t b = d->vals[j];
                if ((b & (b - 1)) != 0) continue;
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)vl) bd_add(d, (uint64_t)p, i, j, OP_MUL, ca + meta_cost(d->meta[j]) + 1, 0);
            }
        }
    }

    /* Small constants 1..256 */
    for (int it = 0; it < 3; it++) {
        uint32_t lim = d->n;
        for (uint32_t i = 0; i < lim && d->n < cap; i++) {
            uint64_t a = d->vals[i]; if (a > 256ULL) continue;
            uint8_t ca = meta_cost(d->meta[i]);
            for (uint32_t j = i; j < lim && d->n < cap; j++) {
                uint64_t b = d->vals[j]; if (b > 256ULL) continue;
                uint8_t nc = ca + meta_cost(d->meta[j]) + 1;
                if (a + b <= 256ULL) bd_add(d, a + b, i, j, OP_ADD, nc, 0);
                if (a * b <= 256ULL) bd_add(d, a * b, i, j, OP_MUL, nc, 0);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                    bd_add(d, hi - lo, phi, plo, OP_SUB, nc, 0);
                }
            }
        }
    }

    /* Layer 1: cross seeds, powers & small constants */
    uint32_t base_end = d->n;
    for (uint32_t i = 0; i < base_end && d->n < cap; i++) {
        uint64_t a = d->vals[i]; uint8_t ca = meta_cost(d->meta[i]);
        if (ca > 8) continue;
        for (uint32_t j = i; j < base_end && d->n < cap; j++) {
            uint64_t b = d->vals[j]; uint8_t cb = meta_cost(d->meta[j]);
            if (cb > 8) continue;
            uint8_t nc = ca + cb + 1;
            u128 p = (u128)a * (u128)b;
            if (p <= (u128)vl) bd_add(d, (uint64_t)p, i, j, OP_MUL, nc, 1);
            u128 s = (u128)a + (u128)b;
            if (s <= (u128)vl) bd_add(d, (uint64_t)s, i, j, OP_ADD, nc, 1);
            if (a != b) {
                uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                bd_add(d, hi - lo, phi, plo, OP_SUB, nc, 1);
                if (lo > 1 && hi % lo == 0) bd_add(d, hi / lo, phi, plo, OP_DIV, nc, 1);
            }
        }
    }

    /* Layer 2: top-down expansion from balanced low-cost representatives per bit bucket */
    uint32_t max_reps = 50U * 400U;
    uint32_t *reps = (uint32_t *)malloc(sizeof(uint32_t) * max_reps);
    uint32_t nr = 0;
    if (reps) {
        for (int bl = 1; bl <= 42 && nr < max_reps; bl++) {
            int cnt = 0;
            for (int tc = 0; tc <= 11 && cnt < 300 && nr < max_reps; tc++) {
                for (uint32_t k = 0; k < d->n && cnt < 300 && nr < max_reps; k++) {
                    if (bl64(d->vals[k]) == bl && meta_cost(d->meta[k]) == tc) {
                        reps[nr++] = k;
                        cnt++;
                    }
                }
            }
        }
        for (int32_t ri = (int32_t)nr - 1; ri >= 0 && d->n < cap; ri--) {
            uint32_t ia = reps[ri];
            uint64_t a = d->vals[ia]; uint8_t ca = meta_cost(d->meta[ia]);
            for (int32_t rj = 0; rj <= ri && d->n < cap; rj++) {
                uint32_t ib = reps[rj];
                uint64_t b = d->vals[ib]; uint8_t cb = meta_cost(d->meta[ib]);
                uint8_t nc = (ca + cb + 1) > 15 ? 15 : (ca + cb + 1);
                if (ca + cb + 1 > 16) continue;
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)vl) bd_add(d, (uint64_t)p, ia, ib, OP_MUL, nc, 1);
                u128 s = (u128)a + (u128)b;
                if (s <= (u128)vl) bd_add(d, (uint64_t)s, ia, ib, OP_ADD, nc, 1);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    uint32_t phi = a > b ? ia : ib, plo = a > b ? ib : ia;
                    bd_add(d, hi - lo, phi, plo, OP_SUB, nc, 1);
                }
            }
        }
        free(reps);
    }

    /* Layer 3 (if high local budget remains): fast strided base cross */
    uint32_t cur_n = d->n;
    if (d->n < cap && cur_n > base_end) {
        for (uint32_t i = base_end; i < cur_n && d->n < cap; i += 3) {
            uint64_t a = d->vals[i];
            if (bl64(a) > 36) continue;
            uint8_t ca = meta_cost(d->meta[i]);
            for (uint32_t j = 0; j < base_end && d->n < cap; j++) {
                uint64_t b = d->vals[j];
                uint8_t cb = meta_cost(d->meta[j]);
                uint8_t nc = (ca + cb + 1) > 15 ? 15 : (ca + cb + 1);
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)vl) bd_add(d, (uint64_t)p, i, j, OP_MUL, nc, 1);
                u128 s = (u128)a + (u128)b;
                if (s <= (u128)vl) bd_add(d, (uint64_t)s, i, j, OP_ADD, nc, 1);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                    bd_add(d, hi - lo, phi, plo, OP_SUB, nc, 1);
                }
            }
        }
    }

    /* Build stratified sorted `scan_vals` up to `scan_cap` for O(N) 2-pointer */
    uint32_t ns_tot = d->n < scan_cap ? d->n : scan_cap;
    SortPair *sp = (SortPair *)malloc(sizeof(SortPair) * ns_tot);
    d->scan_vals  = (uint64_t *)malloc(sizeof(uint64_t) * ns_tot);
    d->scan_costs = (uint8_t *)malloc(sizeof(uint8_t) * ns_tot);
    if (sp && d->scan_vals && d->scan_costs) {
        uint32_t filled = 0;
        if (d->n <= scan_cap) {
            for (uint32_t i = 0; i < d->n; i++) {
                sp[filled].v = d->vals[i];
                sp[filled].c = meta_cost(d->meta[i]);
                filled++;
            }
        } else {
            /* Keep all base nodes + balanced stride up to scan_cap */
            uint32_t keep_front = base_end < (ns_tot / 3) ? base_end : (ns_tot / 3);
            for (uint32_t i = 0; i < keep_front; i++) {
                sp[filled].v = d->vals[i];
                sp[filled].c = meta_cost(d->meta[i]);
                filled++;
            }
            uint64_t rem_slots = ns_tot - filled;
            uint64_t avail = d->n - keep_front;
            for (uint64_t s = 0; s < rem_slots && filled < ns_tot; s++) {
                uint32_t idx = keep_front + (uint32_t)((s * avail) / rem_slots);
                sp[filled].v = d->vals[idx];
                sp[filled].c = meta_cost(d->meta[idx]);
                filled++;
            }
        }
        qsort(sp, filled, sizeof(SortPair), cmp_sp);
        for (uint32_t i = 0; i < filled; i++) {
            d->scan_vals[i]  = sp[i].v;
            d->scan_costs[i] = sp[i].c;
        }
        d->n_scan = filled;
    }
    free(sp);
    return d;
}

typedef struct { uint64_t d1, d2, prod, gap; int32_t metric; uint8_t sgn; } Hit;

/* Combined Two-Pointer scan over `D_local` consulting Universal `U` for residual lookup! */
EXPORT int fe_two_pointer(
    void *p_local, void *p_univ, uint64_t tgt, int K,
    uint64_t *od1, uint64_t *od2, uint64_t *oprod, uint64_t *ogap, int *osgn
) {
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    int N = (int)d->n_scan;
    if (N == 0) return 0;
    if (K > 64) K = 64;

    Hit hits_gap[32]; int ng = 0, wg = 0;
    Hit hits_sc[32];  int ns = 0, ws = 0;
    int Kg = K / 2; if (Kg < 1) Kg = 1; if (Kg > 32) Kg = 32;
    int Ks = K - Kg; if (Ks > 32) Ks = 32;

    int i = 0, j = N - 1;
    u128 t128 = (u128)tgt;
    u128 vl128 = (u128)d->vl;

    while (i <= j) {
        uint64_t a = d->scan_vals[i], b = d->scan_vals[j];
        u128 prod = (u128)a * (u128)b;
        if (prod <= vl128) {
            uint64_t g = (prod <= t128) ? (uint64_t)(t128 - prod) : (uint64_t)(prod - t128);
            uint8_t sg = (prod <= t128) ? 1 : 0;
            if (ng < Kg) {
                hits_gap[ng++] = (Hit){a, b, (uint64_t)prod, g, bl64(g), sg};
                if (hits_gap[ng - 1].gap > hits_gap[wg].gap) wg = ng - 1;
            } else if (g < hits_gap[wg].gap) {
                hits_gap[wg] = (Hit){a, b, (uint64_t)prod, g, bl64(g), sg};
                wg = 0;
                for (int h = 1; h < ng; h++) if (hits_gap[h].gap > hits_gap[wg].gap) wg = h;
            }
            if (Ks > 0) {
                int g_est = 0;
                if (g > 0) {
                    int32_t id = bd_find(d, g);
                    if (id >= 0) g_est = (int)meta_cost(d->meta[id]);
                    else if (u) {
                        int32_t iu = bd_find(u, g);
                        g_est = (iu >= 0) ? ((int)meta_cost(u->meta[iu]) + 1) : (bl64(g) / 3 + 3);
                    } else {
                        g_est = bl64(g) / 3 + 2;
                    }
                }
                int m2 = (int)d->scan_costs[i] + (int)d->scan_costs[j] + g_est;
                if (ns < Ks) {
                    hits_sc[ns++] = (Hit){a, b, (uint64_t)prod, g, m2, sg};
                    if (hits_sc[ns - 1].metric > hits_sc[ws].metric) ws = ns - 1;
                } else if (m2 < hits_sc[ws].metric) {
                    hits_sc[ws] = (Hit){a, b, (uint64_t)prod, g, m2, sg};
                    ws = 0;
                    for (int h = 1; h < ns; h++) if (hits_sc[h].metric > hits_sc[ws].metric) ws = h;
                }
            }
        }
        if (prod < t128) i++;
        else if (prod > t128) j--;
        else break;
    }

    int out = 0;
    for (int h = 0; h < ng; h++) {
        od1[out] = hits_gap[h].d1; od2[out] = hits_gap[h].d2;
        oprod[out] = hits_gap[h].prod; ogap[out] = hits_gap[h].gap; osgn[out] = hits_gap[h].sgn;
        out++;
    }
    for (int h = 0; h < ns; h++) {
        od1[out] = hits_sc[h].d1; od2[out] = hits_sc[h].d2;
        oprod[out] = hits_sc[h].prod; ogap[out] = hits_sc[h].gap; osgn[out] = hits_sc[h].sgn;
        out++;
    }
    return out;
}

typedef struct { uint64_t a, q, r; int32_t sc; uint8_t sgn; } MitmHit;

static inline void mitm_push(
    MitmHit *arr, int *n, int *w, int K, uint64_t a, uint64_t q, uint64_t r, int32_t sc, uint8_t sgn
) {
    if (*n < K) {
        arr[*n] = (MitmHit){a, q, r, sc, sgn};
        if (arr[*n].sc > arr[*w].sc) *w = *n;
        (*n)++;
    } else if (sc < arr[*w].sc) {
        arr[*w] = (MitmHit){a, q, r, sc, sgn};
        *w = 0;
        for (int h = 1; h < *n; h++) if (arr[h].sc > arr[*w].sc) *w = h;
    }
}

/* Lookup cost of `x` across `D_local` (preferred) or `U`; -1 if missing. */
static inline int lookup_cost_both(const BigDict *d, const BigDict *u, uint64_t x) {
    int32_t id = bd_find(d, x);
    if (id >= 0) return (int)meta_cost(d->meta[id]);
    if (u) {
        int32_t iu = bd_find(u, x);
        if (iu >= 0) return (int)meta_cost(u->meta[iu]);
    }
    return -1;
}

/*
 * Bidirectional Meet-in-the-Middle quotient-remainder probe `T = a * q +- r`:
 *   1. Probes all `active` DAG values FIRST (marginal cost(a) = 0).
 *   2. Probes lowest-cost divisors in `D_local` (`scan` + low indices) and `U` (`<= max_u_scan`).
 * Returns exact triples `(a, q, r)` where `q, r in D_local U U` AND near-hits where `r < 2^22`.
 */
EXPORT int fe_mitm_probe(
    void *p_local, void *p_univ, uint64_t tgt, uint64_t vl,
    const uint64_t *act_vals, int n_act, uint32_t max_div_scan, int K,
    uint64_t *oa, uint64_t *oq, uint64_t *or_, int *osgn, int *osc
) {
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    if (K > 48) K = 48;
    MitmHit hits[48];
    int nh = 0, wh = 0;

    /* Helper macro-like inline logic per candidate divisor `a` with base cost `ca` */
    #define TRY_DIVISOR(A_VAL, CA_VAL) do { \
        uint64_t a_v = (A_VAL); \
        if (a_v > 1ULL && a_v <= tgt) { \
            int ca_v = (CA_VAL); \
            uint64_t q = tgt / a_v; \
            uint64_t r = tgt % a_v; \
            int cq = lookup_cost_both(d, u, q); \
            if (cq >= 0) { \
                if (r == 0ULL) { \
                    mitm_push(hits, &nh, &wh, K, a_v, q, 0ULL, ca_v + cq + 1, 1); \
                } else { \
                    int cr = lookup_cost_both(d, u, r); \
                    if (cr >= 0) { \
                        mitm_push(hits, &nh, &wh, K, a_v, q, r, ca_v + cq + cr + 2, 1); \
                    } else if (r < (1ULL << 22)) { \
                        mitm_push(hits, &nh, &wh, K, a_v, q, r, ca_v + cq + (bl64(r) / 3) + 4, 1); \
                    } \
                } \
            } \
            if (r > 0ULL && (u128)a_v * (u128)(q + 1ULL) <= (u128)vl) { \
                uint64_t r2 = a_v - r; \
                int cq1 = lookup_cost_both(d, u, q + 1ULL); \
                if (cq1 >= 0) { \
                    int cr2 = lookup_cost_both(d, u, r2); \
                    if (cr2 >= 0) { \
                        mitm_push(hits, &nh, &wh, K, a_v, q + 1ULL, r2, ca_v + cq1 + cr2 + 2, 0); \
                    } else if (r2 < (1ULL << 22)) { \
                        mitm_push(hits, &nh, &wh, K, a_v, q + 1ULL, r2, ca_v + cq1 + (bl64(r2) / 3) + 4, 0); \
                    } \
                } \
            } \
        } \
    } while (0)

    /* 1. Active DAG values: zero marginal ancestor cost! */
    for (int i = 0; i < n_act; i++) {
        TRY_DIVISOR(act_vals[i], 0);
    }

    /* 2. Seed-local dictionary divisors */
    uint32_t d_lim = d->n < max_div_scan ? d->n : max_div_scan;
    for (uint32_t i = 0; i < d_lim; i++) {
        TRY_DIVISOR(d->vals[i], (int)meta_cost(d->meta[i]));
    }

    /* 3. Universal U divisors (includes full <2^20 and dense 24..34-bit mid divisors) */
    if (u) {
        uint32_t u_lim = u->n < max_div_scan ? u->n : max_div_scan;
        for (uint32_t i = 1; i < u_lim; i++) {
            TRY_DIVISOR(u->vals[i], (int)meta_cost(u->meta[i]) + 1);
        }
    }

    #undef TRY_DIVISOR

    for (int i = 0; i < nh; i++) {
        oa[i] = hits[i].a; oq[i] = hits[i].q; or_[i] = hits[i].r;
        osgn[i] = (int)hits[i].sgn; osc[i] = (int)hits[i].sc;
    }
    return nh;
}

/*
 * Fast C99 trial division over `primes[np]` to score smooth candidates `T + k` in < 1ms.
 * Outputs `out_score[i]` = small score (number of prime factors + residual bit length/4);
 * lower is smoother!
 */
EXPORT void fe_screen_smooth(
    uint64_t tgt, uint64_t vl, const int32_t *ks, int nk,
    const uint32_t *primes, int np, int32_t *out_sc
) {
    for (int i = 0; i < nk; i++) {
        int64_t k = (int64_t)ks[i];
        int64_t cand_s = (k >= 0) ? 1 : -1;
        uint64_t cand;
        if (k >= 0) {
            if (tgt > vl - (uint64_t)k) { out_sc[i] = 9999; continue; }
            cand = tgt + (uint64_t)k;
        } else {
            uint64_t ak = (uint64_t)(-k);
            if (tgt <= ak) { out_sc[i] = 9999; continue; }
            cand = tgt - ak;
        }
        uint64_t n = cand;
        int facs = 0;
        for (int pi = 0; pi < np; pi++) {
            uint32_t p = primes[pi];
            if ((u128)p * (u128)p > (u128)n) break;
            while (n % p == 0) {
                n /= p;
                facs++;
            }
        }
        int rb = (n <= 1ULL) ? 0 : bl64(n);
        out_sc[i] = facs + (rb > 0 ? (rb / 3 + 2) : 0) + (k == 0 ? 0 : 1);
        (void)cand_s;
    }
}
'''

_CLIB_CACHE = None
_CLIB_FAILED = False
_OP_MAP = {1: "+", 2: "-", 3: "*", 4: "/"}

# Shared global Universal `{1}`-Dictionary context (Read-Only after init for Linux fork CoW)
_UNIV_CTX: Optional[int] = None
_UNIV_CAP: int = 0
_UNIV_SIZE: int = 0


def get_clib() -> Optional[ctypes.CDLL]:
    """
    Lazily compile and load the V5 C99 acceleration library in tempdir.
    Safe under concurrent multiprocessing (atomic temp write + rename).
    """
    global _CLIB_CACHE, _CLIB_FAILED
    if _CLIB_CACHE is not None:
        return _CLIB_CACHE
    if _CLIB_FAILED:
        return None
    if shutil.which("gcc") is None:
        _CLIB_FAILED = True
        return None

    try:
        ext = ".dll" if os.name == "nt" else ".so"
        tdir = tempfile.gettempdir()
        so_path = os.path.join(tdir, f"santa2026_fast_engine_v5{ext}")
        if not os.path.exists(so_path):
            pid = os.getpid()
            c_path = os.path.join(tdir, f"santa2026_fast_engine_v5_{pid}.c")
            tmp_so = os.path.join(tdir, f"santa2026_fast_engine_v5_{pid}{ext}")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(_C_SOURCE)
            subprocess.run(
                ["gcc", "-O3", "-shared", "-fPIC", c_path, "-o", tmp_so],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            try:
                os.replace(tmp_so, so_path)
            except OSError:
                if not os.path.exists(so_path):
                    so_path = tmp_so
            try:
                os.remove(c_path)
            except OSError:
                pass

        lib = ctypes.CDLL(so_path)
        lib.fe_univ_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.fe_univ_create.restype = ctypes.c_void_p

        lib.fe_local_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.fe_local_create.restype = ctypes.c_void_p

        lib.fe_dict_free.argtypes = [ctypes.c_void_p]
        lib.fe_dict_size.argtypes = [ctypes.c_void_p]
        lib.fe_dict_size.restype = ctypes.c_uint32
        lib.fe_scan_size.argtypes = [ctypes.c_void_p]
        lib.fe_scan_size.restype = ctypes.c_uint32
        lib.fe_dict_lookup.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.fe_dict_lookup.restype = ctypes.c_int32
        lib.fe_dict_get.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_dict_copy_scan.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.fe_dict_copy_scan.restype = ctypes.c_uint32

        lib.fe_two_pointer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_two_pointer.restype = ctypes.c_int

        lib.fe_mitm_probe.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_mitm_probe.restype = ctypes.c_int

        lib.fe_screen_smooth.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]

        _CLIB_CACHE = lib
        return lib
    except Exception:
        _CLIB_FAILED = True
        return None


def init_universal_dict(
    ram_gb: str | float = "auto", *, force_python: bool = False
) -> dict[str, int | float | str]:
    """
    Build the global parent-shared read-only `{1}` Universal Dictionary `U` once.
    On Linux `fork()`, all worker processes share `U` via Copy-on-Write (`1x` RAM).
    Automatically steps down capacity cleanly if host allocation returns NULL.
    """
    global _UNIV_CTX, _UNIV_CAP, _UNIV_SIZE
    prof = dict(get_ram_profile(ram_gb))
    if force_python:
        return {**prof, "univ_size": 0, "c99": 0}

    lib = get_clib()
    if lib is None:
        return {**prof, "univ_size": 0, "c99": 0}

    target_cap = int(prof["univ_cap"])
    target_pow = int(prof["univ_ht_pow"])
    if _UNIV_CTX is not None and _UNIV_CAP >= target_cap:
        return {**prof, "univ_size": _UNIV_SIZE, "c99": 1}

    if _UNIV_CTX is not None:
        lib.fe_dict_free(_UNIV_CTX)
        _UNIV_CTX = None
        _UNIV_CAP = 0
        _UNIV_SIZE = 0

    cap = target_cap
    htp = target_pow
    while cap >= 2_000_000:
        ctx = lib.fe_univ_create(ctypes.c_uint32(cap), ctypes.c_uint32(htp))
        if ctx:
            _UNIV_CTX = ctx
            _UNIV_CAP = cap
            _UNIV_SIZE = int(lib.fe_dict_size(ctx))
            return {**prof, "univ_cap": cap, "univ_ht_pow": htp, "univ_size": _UNIV_SIZE, "c99": 1}
        cap //= 2
        if htp > 23:
            htp -= 1

    return {**prof, "univ_size": 0, "c99": 0}


def release_universal_dict() -> None:
    """Free the global Universal Dictionary cleanly (useful in unit tests)."""
    global _UNIV_CTX, _UNIV_CAP, _UNIV_SIZE
    if _UNIV_CTX is not None and _CLIB_CACHE is not None:
        _CLIB_CACHE.fe_dict_free(_UNIV_CTX)
    _UNIV_CTX = None
    _UNIV_CAP = 0
    _UNIV_SIZE = 0


# ---------------------------------------------------------------------------
# Pure-Python Fallback Backend
# ---------------------------------------------------------------------------

class _PyDictBackend:
    """Pure-Python fallback dictionary engine with identical interface to C backend."""

    def __init__(self, seeds: list[int], vl: int, max_nodes: int = 24000, per_bucket: int = 320):
        self.vl = vl
        self.defs: dict[int, tuple[str, int, int]] = {}
        self.cost: dict[int, int] = {}
        self.buckets: list[list[int]] = [[] for _ in range(65)]

        def add_node(v: int, op: str, a: int, b: int, c: int, cap: int = 0) -> None:
            if v <= 0 or v > vl or v in self.cost or len(self.cost) >= max_nodes:
                return
            bl = v.bit_length()
            if cap > 0 and len(self.buckets[bl]) >= cap:
                return
            self.defs[v] = (op, a, b)
            self.cost[v] = c
            self.buckets[bl].append(v)

        for s in seeds:
            if 0 < s <= vl and s not in self.cost:
                self.cost[s] = 0
                self.buckets[s.bit_length()].append(s)

        if 1 not in self.cost and seeds:
            add_node(1, "/", seeds[0], seeds[0], 1)

        if 2 <= vl and 2 not in self.cost and 1 in self.cost:
            add_node(2, "+", 1, 1, self.cost[1] + 1)
        sq = 2
        while sq <= vl and sq * sq <= vl:
            add_node(sq * sq, "*", sq, sq, self.cost.get(sq, 1) + 1)
            sq *= sq
        cur = 1
        while cur <= 1024 and cur * 2 <= vl:
            add_node(cur * 2, "+", cur, cur, self.cost.get(cur, 1) + 1)
            cur *= 2
        for _ in range(3):
            pwrs = sorted(v for v in self.cost if (v & (v - 1)) == 0)
            for i, a in enumerate(pwrs):
                for b in pwrs[i:]:
                    if a * b <= vl:
                        add_node(a * b, "*", a, b, self.cost[a] + self.cost[b] + 1)

        for _ in range(3):
            smalls = sorted(v for v in self.cost if v <= 128)
            for i, a in enumerate(smalls):
                for b in smalls[i:]:
                    nc = self.cost[a] + self.cost[b] + 1
                    if a + b <= 128:
                        add_node(a + b, "+", a, b, nc)
                    if a * b <= 128:
                        add_node(a * b, "*", a, b, nc)
                    if a != b:
                        hi, lo = (a, b) if a > b else (b, a)
                        add_node(hi - lo, "-", hi, lo, nc)

        base = sorted(v for v, c in self.cost.items() if c <= 9)
        for i, a in enumerate(base):
            ca = self.cost[a]
            for b in base[i:]:
                nc = ca + self.cost[b] + 1
                if a * b <= vl:
                    add_node(a * b, "*", a, b, nc, per_bucket)
                if a + b <= vl:
                    add_node(a + b, "+", a, b, nc, per_bucket)
                if a != b:
                    hi, lo = (a, b) if a > b else (b, a)
                    add_node(hi - lo, "-", hi, lo, nc, per_bucket)
                    if hi % lo == 0:
                        add_node(hi // lo, "/", hi, lo, nc, per_bucket)

        reps = []
        for bl in range(1, 41):
            if self.buckets[bl]:
                bs = sorted(self.buckets[bl], key=lambda x: (self.cost[x], x))
                reps.extend(bs[:35])
        for ri in range(len(reps) - 1, -1, -1):
            if len(self.cost) >= max_nodes:
                break
            a = reps[ri]
            ca = self.cost[a]
            for rj in range(ri + 1):
                if len(self.cost) >= max_nodes:
                    break
                b = reps[rj]
                nc = ca + self.cost[b] + 1
                if nc > 16:
                    continue
                if a * b <= vl:
                    add_node(a * b, "*", a, b, nc, per_bucket)
                if a + b <= vl:
                    add_node(a + b, "+", a, b, nc, per_bucket)
                if a != b:
                    hi, lo = (a, b) if a > b else (b, a)
                    add_node(hi - lo, "-", hi, lo, nc, per_bucket)

        self.D_sorted = sorted(self.cost.keys())

    def two_pointer(self, tgt: int, K: int = 12) -> list[tuple[int, int, int, int, int]]:
        D = self.D_sorted
        n = len(D)
        if n == 0:
            return []
        kg = max(1, K // 2)
        ks = K - kg
        hits_gap = []
        hits_sc = []
        seen_gap = set()
        seen_sc = set()
        i, j = 0, n - 1
        while i <= j:
            a, b = D[i], D[j]
            prod = a * b
            if prod <= self.vl:
                if prod <= tgt:
                    g, sg = tgt - prod, 1
                else:
                    g, sg = prod - tgt, 0
                if prod not in seen_gap:
                    if len(hits_gap) < kg:
                        hits_gap.append((g, a, b, prod, g, sg))
                        seen_gap.add(prod)
                        hits_gap.sort()
                    elif g < hits_gap[-1][0]:
                        seen_gap.discard(hits_gap[-1][3])
                        hits_gap[-1] = (g, a, b, prod, g, sg)
                        seen_gap.add(prod)
                        hits_gap.sort()
                if ks > 0 and prod not in seen_sc:
                    g_est = 0 if g == 0 else (1 if g in self.cost else g.bit_length() // 3 + 2)
                    m2 = self.cost[a] + self.cost[b] + g_est
                    if len(hits_sc) < ks:
                        hits_sc.append((m2, a, b, prod, g, sg))
                        seen_sc.add(prod)
                        hits_sc.sort()
                    elif m2 < hits_sc[-1][0]:
                        seen_sc.discard(hits_sc[-1][3])
                        hits_sc[-1] = (m2, a, b, prod, g, sg)
                        seen_sc.add(prod)
                        hits_sc.sort()
            if prod < tgt:
                i += 1
            elif prod > tgt:
                j -= 1
            else:
                break
        return [(a, b, prod, g, sg) for (_, a, b, prod, g, sg) in hits_gap] + [
            (a, b, prod, g, sg) for (_, a, b, prod, g, sg) in hits_sc
        ]


# ---------------------------------------------------------------------------
# Unified High-RAM DictEngine Interface
# ---------------------------------------------------------------------------

class DictEngine:
    """
    Unified interface to the C99 Universal+Local (or pure-Python fallback)
    dictionary & Meet-in-the-Middle engine. Always use via `with DictEngine(puzzle) as de:`.
    """

    def __init__(
        self,
        puzzle: Puzzle,
        *,
        force_python: bool = False,
        max_nodes: Optional[int] = None,
        per_bucket: Optional[int] = None,
        ram_gb: str | float = "auto",
    ):
        self.puzzle = puzzle
        self.vl = puzzle.value_limit
        self.seeds = list(puzzle.seeds)
        self.seeds_set = set(self.seeds)
        self.profile = get_ram_profile(ram_gb)

        cap = int(max_nodes if max_nodes is not None else self.profile["local_cap"])
        ht_pow = int(self.profile["local_ht_pow"])
        while (1 << ht_pow) < cap * 2 and ht_pow < 27:
            ht_pow += 1
        pb = int(per_bucket if per_bucket is not None else self.profile["per_bucket"])
        scan_cap = min(cap, int(self.profile["local_scan"]))

        self._node_cache: dict[int, Optional[tuple[str, int, int]]] = {}
        self._has_cache: dict[int, bool] = {}
        self._tp_cache: dict[int, list[tuple[int, int, int, int, int]]] = {}
        self._mitm_cache: dict[int, list[tuple[int, int, int, int, int]]] = {}
        self.ctx = None
        self.univ_ctx = None
        self.py_backend: Optional[_PyDictBackend] = None

        self.clib = None if force_python else get_clib()
        if self.clib is not None and self.vl < (1 << 63) and self.seeds:
            arr = (ctypes.c_uint64 * len(self.seeds))(*self.seeds)
            # Retry step-down if local allocation fails
            c_try, h_try = cap, ht_pow
            while c_try >= 200_000:
                self.ctx = self.clib.fe_local_create(
                    arr, len(self.seeds), ctypes.c_uint64(self.vl),
                    ctypes.c_uint32(c_try), ctypes.c_uint32(h_try),
                    ctypes.c_uint32(pb), ctypes.c_uint32(scan_cap),
                )
                if self.ctx:
                    break
                c_try //= 2
                if h_try > 19:
                    h_try -= 1

            if self.ctx:
                self.univ_ctx = _UNIV_CTX
                ns = int(self.clib.fe_scan_size(self.ctx))
                c_arr = (ctypes.c_uint64 * ns)()
                self.clib.fe_dict_copy_scan(self.ctx, c_arr)
                self.D_sorted = list(c_arr)
            else:
                self.clib = None

        if self.ctx is None:
            self.py_backend = _PyDictBackend(self.seeds, self.vl)
            self.D_sorted = self.py_backend.D_sorted

        self.D_scan_set = set(self.D_sorted)
        self.powers: set[int] = set()
        pw = 1
        while pw <= self.vl:
            if self.has(pw):
                self.powers.add(pw)
            pw *= 2
        self.split_muls: list[int] = [v for v in self.D_sorted if 16 <= v.bit_length() <= 24][:12]

    def __enter__(self) -> "DictEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.ctx is not None and self.clib is not None:
            self.clib.fe_dict_free(self.ctx)
            self.ctx = None

    def has(self, v: int) -> bool:
        if v <= 0 or v > self.vl:
            return False
        if v in self.D_scan_set:
            return True
        cached = self._has_cache.get(v)
        if cached is not None:
            return cached
        if self.py_backend is not None:
            res = v in self.py_backend.cost
            self._has_cache[v] = res
            return res
        if self.clib.fe_dict_lookup(self.ctx, ctypes.c_uint64(v)) >= 0:
            self._has_cache[v] = True
            return True
        if self.univ_ctx is not None and self.clib.fe_dict_lookup(self.univ_ctx, ctypes.c_uint64(v)) >= 0:
            self._has_cache[v] = True
            return True
        self._has_cache[v] = False
        return False

    def get_dict_def(self, v: int) -> Optional[tuple[str, int, int]]:
        if v in self._node_cache:
            return self._node_cache[v]
        if v in self.seeds_set or v <= 0 or v > self.vl:
            self._node_cache[v] = None
            return None
        if self.py_backend is not None:
            res = self.py_backend.defs.get(v)
            self._node_cache[v] = res
            return res

        # 1. Prefer seed-local definition (often much shorter via puzzle seeds)
        idx = self.clib.fe_dict_lookup(self.ctx, ctypes.c_uint64(v))
        src_ctx = self.ctx
        if idx < 0 and self.univ_ctx is not None:
            idx = self.clib.fe_dict_lookup(self.univ_ctx, ctypes.c_uint64(v))
            src_ctx = self.univ_ctx
        if idx < 0:
            self._node_cache[v] = None
            return None

        vp = ctypes.c_uint64()
        ap = ctypes.c_uint64()
        bp = ctypes.c_uint64()
        op = ctypes.c_int()
        cp = ctypes.c_int()
        self.clib.fe_dict_get(
            src_ctx, ctypes.c_uint32(idx), ctypes.byref(vp), ctypes.byref(ap),
            ctypes.byref(bp), ctypes.byref(op), ctypes.byref(cp)
        )
        if op.value <= 0:
            self._node_cache[v] = None
            return None
        res = (_OP_MAP[op.value], ap.value, bp.value)
        self._node_cache[v] = res
        return res

    def two_pointer(self, tgt: int, K: int = 14) -> list[tuple[int, int, int, int, int]]:
        if tgt in self._tp_cache:
            return self._tp_cache[tgt]
        if self.py_backend is not None:
            res = self.py_backend.two_pointer(tgt, K=K)
            self._tp_cache[tgt] = res
            return res
        od1 = (ctypes.c_uint64 * K)()
        od2 = (ctypes.c_uint64 * K)()
        oprod = (ctypes.c_uint64 * K)()
        ogap = (ctypes.c_uint64 * K)()
        osgn = (ctypes.c_int * K)()
        nh = self.clib.fe_two_pointer(
            self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), K,
            od1, od2, oprod, ogap, osgn
        )
        res = [(od1[i], od2[i], oprod[i], ogap[i], osgn[i]) for i in range(nh)]
        self._tp_cache[tgt] = res
        return res

    def mitm_probe(
        self, tgt: int, active_vals: list[int], K: int = 20
    ) -> list[tuple[int, int, int, int, int]]:
        """
        Bidirectional C99 MITM quotient-remainder probe `T = a * q +- r`:
          - Caches the expensive static `D_local U U` divisor scan ONCE per `tgt`.
          - Always probes current `active_vals` (marginal cost = 0) dynamically (<0.05ms).
        """
        if self.py_backend is not None or self.ctx is None or tgt <= 100:
            return []
        max_div = int(self.profile["mitm_div_scan"])

        static_hits = self._mitm_cache.get(tgt)
        if static_hits is None:
            empty_act = (ctypes.c_uint64 * 1)(0)
            oa = (ctypes.c_uint64 * K)()
            oq = (ctypes.c_uint64 * K)()
            or_ = (ctypes.c_uint64 * K)()
            osgn = (ctypes.c_int * K)()
            osc = (ctypes.c_int * K)()
            nh = self.clib.fe_mitm_probe(
                self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), ctypes.c_uint64(self.vl),
                empty_act, 0, ctypes.c_uint32(max_div), K, oa, oq, or_, osgn, osc
            )
            static_hits = [(oa[i], oq[i], or_[i], osgn[i], osc[i]) for i in range(nh)]
            self._mitm_cache[tgt] = static_hits

        na = min(len(active_vals), 256)
        if na > 0:
            act_arr = (ctypes.c_uint64 * na)(*active_vals[-na:])
            Ka = min(8, K)
            oa = (ctypes.c_uint64 * Ka)()
            oq = (ctypes.c_uint64 * Ka)()
            or_ = (ctypes.c_uint64 * Ka)()
            osgn = (ctypes.c_int * Ka)()
            osc = (ctypes.c_int * Ka)()
            nh_a = self.clib.fe_mitm_probe(
                self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), ctypes.c_uint64(self.vl),
                act_arr, na, ctypes.c_uint32(0), Ka, oa, oq, or_, osgn, osc
            )
            act_hits = [(oa[i], oq[i], or_[i], osgn[i], osc[i]) for i in range(nh_a)]
            combined = act_hits + static_hits
        else:
            combined = list(static_hits)

        combined.sort(key=lambda x: x[4])
        return combined[:K]

    def screen_smooth_offsets(
        self, tgt: int, k_cands: list[int], primes: list[int], top_m: int = 24
    ) -> list[int]:
        """Rank offset candidates `T + k` by C99 trial-factor smoothness in <1ms."""
        if not k_cands:
            return []
        if self.clib is None:
            return k_cands[:top_m]
        nk = len(k_cands)
        np = len(primes)
        k_arr = (ctypes.c_int32 * nk)(*k_cands)
        p_arr = (ctypes.c_uint32 * np)(*primes)
        out_arr = (ctypes.c_int32 * nk)()
        self.clib.fe_screen_smooth(
            ctypes.c_uint64(tgt), ctypes.c_uint64(self.vl),
            k_arr, nk, p_arr, np, out_arr
        )
        ranked = sorted(zip(out_arr, k_cands), key=lambda x: x[0])
        return [k for sc, k in ranked[:top_m] if sc < 9999]

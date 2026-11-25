"""
src/solvers/fast_engine.py
V6 High-RAM 4-Thread Sharded 128-bit Universal + Local Dictionary & C99 MITM Engine.

Key Engineering & Speed Upgrades:
  1. 4-Thread Lock-Free 64-Shard Universal `{1}`-Dictionary (`U`):
     Uses 64 hash shards (`(hash64(v) >> 58) & 63`) so all 4 CPU threads populate
     independent shards + disjoint value slices with zero locks/atomics (~6 M nodes/s).
     On Kaggle (~29-30 GB RAM), builds up to `500,000,000` packed 16-byte nodes +
     `2^31` 4-byte hash slots (`~16.59 GB` Read-Only Copy-on-Write across all 4 workers).
  2. Zero-Stall `two_pointer` L3 Bitset:
     Retains the `2 MiB` `seen24` (`< 2^24`) bitset so `fe_two_pointer` evaluates
     gap reachability in ~1ns in L3 cache instead of missing DRAM in a multi-GB HT.
  3. High-Yield Divisor Slice (`u->scan_vals`):
     Caches balanced `< 2^20` + `24..34`-bit divisors in `U` for targeted MITM probing.
  4. Fast Per-Worker `D_local` (`fe_local_create`):
     Capped Layer 2/3 (`Phase A` base cross) builds 10M seed-derived nodes in ~1-2s.
"""
from __future__ import annotations
import ctypes
import os
import shutil
import subprocess
import tempfile
from typing import Optional
from src.utils import Puzzle


# ---------------------------------------------------------------------------
# Host / Kaggle RAM & Process Telemetry
# ---------------------------------------------------------------------------

def detect_host_ram_gb() -> tuple[float, float]:
    """Return (total_ram_gb, avail_ram_gb) cleanly across Linux and Windows."""
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


def get_ram_telemetry() -> dict[str, float]:
    """Return current host total/used/avail GB and current process RSS GB."""
    tot_gb, avail_gb = detect_host_ram_gb()
    used_gb = max(0.0, tot_gb - avail_gb)
    rss_gb = 0.0
    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_gb = int(line.split()[1]) / (1024 * 1024)
                        break
        except Exception:
            pass
    return {
        "total_gb": tot_gb,
        "used_gb": used_gb,
        "avail_gb": avail_gb,
        "proc_rss_gb": rss_gb,
    }


def get_ram_profile(ram_gb: str | float = "auto") -> dict[str, int | float | str]:
    """
    Compute C99 Universal `U`, Per-Worker `D_local`, MILP, and Compiler budgets.
    On Kaggle (~29-30 GB RAM, Linux CoW), defaults to `kaggle_high_ram` (~21-24 GB
    aggregate: ~16.6 GB shared CoW `U` + 4 workers x ~1.2-1.8 GB).
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
        eff_gb = max(avail_gb, 25.0) if (on_kaggle and total_gb >= 24.0) else avail_gb

    if eff_gb >= 22.0:
        # Kaggle ~20-25 GB aggregate budget:
        #   U: 360M nodes x 16B (5.76 GB) + HT=2^31 x 4B (8.59 GB) = 14.35 GB Read-Only CoW
        #   + 4 workers x (26M local ~690 MB + 1M scan + MITM + 5.5k MILP/Python) ~ 5-8 GB
        return {
            "mode": "kaggle_high_ram",
            "eff_gb": eff_gb,
            "univ_cap": 360_000_000,
            "univ_ht_pow": 31,
            "local_cap": 26_000_000,
            "local_ht_pow": 26,
            "local_scan": 1_000_000,
            "per_bucket": 450_000,
            "mitm_div_scan": 240_000,
            "smooth_offset_lim": 256,
            "milp_max_edges": 5500,
            "milp_time": 2.0,
            "compiler_trials": 64,
        }
    if eff_gb >= 14.0:
        return {
            "mode": "high_ram_16g",
            "eff_gb": eff_gb,
            "univ_cap": 280_000_000,
            "univ_ht_pow": 30,
            "local_cap": 4_000_000,
            "local_ht_pow": 24,
            "local_scan": 600_000,
            "per_bucket": 65_000,
            "mitm_div_scan": 220_000,
            "smooth_offset_lim": 200,
            "milp_max_edges": 3000,
            "milp_time": 1.5,
            "compiler_trials": 48,
        }
    if eff_gb >= 6.5:
        return {
            "mode": "medium_ram",
            "eff_gb": eff_gb,
            "univ_cap": 60_000_000,
            "univ_ht_pow": 27,
            "local_cap": 1_500_000,
            "local_ht_pow": 22,
            "local_scan": 450_000,
            "per_bucket": 25_000,
            "mitm_div_scan": 180_000,
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
# C99 Backend (64-Shard Lock-Free Parallel U + L3 Seen24 + High-Yield Divs)
# ---------------------------------------------------------------------------

_C_SOURCE = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#include <pthread.h>
#include <unistd.h>
#define EXPORT
#endif

typedef unsigned __int128 u128;

#define EMPTY_PA 0x1FFFFFFFULL
#define N_SHARDS 64
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
    uint64_t *seen24;     /* 2^24 bits = 2 MiB fast L3 duplicate/hit bitset */
    uint32_t  shard_mask;
    uint32_t  shard_shift;
    uint32_t  n;          /* total active logical nodes (may include 0-gaps in parallel build) */
    uint32_t  cap;
    uint64_t  vl;
    int32_t   bucket_cnt[65];
    int32_t   bucket_cap[65];
    uint64_t  alloc_bytes;

    /* Contiguous sorted scan / high-yield divisor slice */
    uint64_t *scan_vals;
    uint8_t  *scan_costs;
    uint32_t  n_scan;
} BigDict;

static inline int32_t bd_find(const BigDict *d, uint64_t k) {
    if (!d || k == 0 || k > d->vl) return -1;
    uint64_t hv = hash64(k);
    uint32_t sh = (uint32_t)(hv >> 58) & (N_SHARDS - 1);
    uint32_t sb = sh << d->shard_shift;
    uint32_t sm = d->shard_mask;
    uint32_t h = (uint32_t)hv & sm;
    for (;;) {
        int32_t idx = d->ht[sb | h];
        if (idx < 0) return -1;
        if (d->vals[idx] == k) return idx;
        h = (h + 1U) & sm;
    }
}

static inline int bd_add(
    BigDict *d, uint64_t v, uint32_t pa, uint32_t pb, uint8_t op, uint8_t cost, int use_cap
) {
    if (v == 0 || v > d->vl || d->n >= d->cap) return 0;
    int bl = bl64(v);
    if (use_cap && d->bucket_cap[bl] > 0 && d->bucket_cnt[bl] >= d->bucket_cap[bl]) return 0;
    if (d->seen24 && v < (1ULL << 24) && (d->seen24[v >> 6] & (1ULL << (v & 63)))) return 0;

    uint64_t hv = hash64(v);
    uint32_t sh = (uint32_t)(hv >> 58) & (N_SHARDS - 1);
    uint32_t sb = sh << d->shard_shift;
    uint32_t sm = d->shard_mask;
    uint32_t h = (uint32_t)hv & sm;
    for (;;) {
        int32_t idx = d->ht[sb | h];
        if (idx < 0) break;
        if (d->vals[idx] == v) return 0;
        h = (h + 1U) & sm;
    }
    uint32_t idx = d->n++;
    d->vals[idx] = v;
    d->meta[idx] = pack_meta(pa, pb, op, cost);
    d->ht[sb | h] = (int32_t)idx;
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

static BigDict* alloc_bigdict(uint32_t cap, uint32_t ht_pow, uint64_t vl) {
    if (cap > 530000000U) cap = 530000000U;
    if (ht_pow < 12U) ht_pow = 12U;
    BigDict *d = (BigDict *)calloc(1, sizeof(BigDict));
    if (!d) return NULL;
    uint64_t ht_size = 1ULL << ht_pow;
    d->shard_shift = ht_pow - 6U;
    d->shard_mask = (1U << d->shard_shift) - 1U;
    d->cap = cap;
    d->vl = vl;

    /* Use calloc for vals so any unused parallel gaps remain 0 cleanly */
    size_t sz_v = (size_t)cap * sizeof(uint64_t);
    size_t sz_m = (size_t)cap * sizeof(uint64_t);
    size_t sz_h = (size_t)ht_size * sizeof(int32_t);
    size_t sz_s = (size_t)(1U << 18) * sizeof(uint64_t);

    d->vals = (uint64_t *)calloc(cap, sizeof(uint64_t));
    d->meta = (uint64_t *)malloc(sz_m);
    d->ht   = (int32_t *)malloc(sz_h);
    d->seen24 = (uint64_t *)calloc((1U << 18), sizeof(uint64_t));
    if (!d->vals || !d->meta || !d->ht || !d->seen24) {
        free_bigdict(d);
        return NULL;
    }
    memset(d->ht, 0xFF, sz_h);
    d->alloc_bytes = (uint64_t)(sz_v + sz_m + sz_h + sz_s + sizeof(BigDict));
    return d;
}

/* Parallel worker context for lock-free 64-shard Universal U construction */
typedef struct {
    BigDict *d;
    int tid;
    int n_threads;
    uint32_t base_idx;
    uint32_t max_idx;
    uint32_t local_n;
    uint32_t d5_end;
    uint32_t d6_end;
    uint32_t phase_a_end;
    int32_t bcnt[65];
    int32_t bcap[65];
} UnivWorkerArg;

static inline void univ_par_add(UnivWorkerArg *w, uint64_t v, uint32_t pa, uint32_t pb, uint8_t op, uint8_t cost) {
    BigDict *d = w->d;
    if (v == 0 || v > d->vl || w->local_n >= w->max_idx) return;
    int bl = bl64(v);
    if (w->bcap[bl] > 0 && w->bcnt[bl] >= w->bcap[bl]) return;
    if (v < (1ULL << 24) && (d->seen24[v >> 6] & (1ULL << (v & 63)))) return;

    uint64_t hv = hash64(v);
    uint32_t sh = (uint32_t)(hv >> 58) & (N_SHARDS - 1);
    if ((int)(sh % (uint32_t)w->n_threads) != w->tid) return;

    uint32_t sb = sh << d->shard_shift;
    uint32_t sm = d->shard_mask;
    uint32_t h = (uint32_t)hv & sm;
    for (;;) {
        int32_t idx = d->ht[sb | h];
        if (idx < 0) break;
        if (d->vals[idx] == v) return;
        h = (h + 1U) & sm;
    }
    uint32_t idx = w->base_idx + w->local_n++;
    d->vals[idx] = v;
    d->meta[idx] = pack_meta(pa, pb, op, cost);
    d->ht[sb | h] = (int32_t)idx;
    w->bcnt[bl]++;
}

static void univ_worker_body(UnivWorkerArg *w) {
    BigDict *d = w->d;
    uint32_t d5 = w->d5_end;
    uint32_t j_lim = d5 < 260U ? d5 : 260U;
    uint32_t pa_end = w->phase_a_end;
    uint64_t since_add = 0ULL;
    for (uint32_t i = w->d6_end; i < pa_end && w->local_n < w->max_idx; i++) {
        uint64_t a = d->vals[i];
        if (a == 0ULL || bl64(a) > 40) continue;
        uint32_t prev_n = w->local_n;
        for (uint32_t j = 0; j < j_lim && w->local_n < w->max_idx; j++) {
            uint64_t b = d->vals[j];
            u128 p = (u128)a * (u128)b;
            if (p <= (u128)d->vl) univ_par_add(w, (uint64_t)p, i, j, OP_MUL, 8);
            u128 s = (u128)a + (u128)b;
            if (s <= (u128)d->vl) univ_par_add(w, (uint64_t)s, i, j, OP_ADD, 8);
            if (a != b) {
                uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                univ_par_add(w, hi - lo, phi, plo, OP_SUB, 8);
            }
        }
        if (w->local_n == prev_n) {
            since_add += j_lim;
            if (since_add > 10000000ULL) break;
        } else {
            since_add = 0ULL;
        }
    }
    /* Parallel Phase C: strided Phase A x d6 across 4 threads to top up 25..36b bands quickly */
    if (w->local_n < (w->max_idx * 90U) / 100U && pa_end > w->d6_end) {
        uint32_t stride_a = (pa_end - w->d6_end) > 30000U ? ((pa_end - w->d6_end) / 30000U) : 1U;
        uint32_t stride_b = (w->d6_end - d5) > 7000U ? ((w->d6_end - d5) / 7000U) : 1U;
        uint64_t since_c = 0ULL;
        for (uint32_t i = w->d6_end; i < pa_end && w->local_n < w->max_idx; i += stride_a) {
            uint64_t a = d->vals[i];
            if (a == 0ULL || bl64(a) > 36) continue;
            uint32_t prev_c = w->local_n;
            for (uint32_t j = d5; j < w->d6_end && w->local_n < w->max_idx; j += stride_b) {
                uint64_t b = d->vals[j];
                if (bl64(b) > 36) continue;
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)d->vl) univ_par_add(w, (uint64_t)p, i, j, OP_MUL, 9);
                u128 s = (u128)a + (u128)b;
                if (s <= (u128)d->vl) univ_par_add(w, (uint64_t)s, i, j, OP_ADD, 9);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    uint32_t phi = a > b ? i : j, plo = a > b ? j : i;
                    univ_par_add(w, hi - lo, phi, plo, OP_SUB, 9);
                }
            }
            if (w->local_n == prev_c) {
                since_c += 7000ULL;
                if (since_c > 12000000ULL) break;
            } else {
                since_c = 0ULL;
            }
        }
    }
}

#ifdef _WIN32
static DWORD WINAPI win_univ_worker(LPVOID arg) { univ_worker_body((UnivWorkerArg *)arg); return 0; }
#else
static void* posix_univ_worker(void *arg) { univ_worker_body((UnivWorkerArg *)arg); return NULL; }
#endif

/*
 * Global Parent-Shared Read-Only Universal `{1}` Dictionary built across `n_threads`
 * lock-free sharded OS threads (~6 M nodes/sec!).
 */
EXPORT void* fe_univ_create(uint32_t cap, uint32_t ht_pow, int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;
    BigDict *d = alloc_bigdict(cap, ht_pow, (1ULL << 61));
    if (!d) return NULL;

    for (int bl = 1; bl <= 64; bl++) {
        if (bl <= 24)      d->bucket_cap[bl] = (int32_t)(cap / 12U);
        else if (bl <= 38) d->bucket_cap[bl] = (int32_t)(cap / 14U);
        else if (bl <= 46) d->bucket_cap[bl] = (int32_t)(cap / 28U);
        else               d->bucket_cap[bl] = (int32_t)(cap / 75U);
    }

    bd_add(d, 1ULL, EMPTY_PA, EMPTY_PA, OP_ADD, 0, 0);
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

    /* Depth 1..6 exhaustive (~75,955 nodes in <25ms) */
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

    /* Phase A (d6 x <= d5): ~6.6M low-cost nodes in ~0.9s */
    uint32_t d6_end = d->n;
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

    /* Phase B (Parallel lock-free sharded expansion across Phase A x <= d5) */
    uint32_t pa_end = d->n;
    uint32_t total_valid = pa_end;
    if (pa_end < cap) {
        UnivWorkerArg wargs[8];
        uint32_t chunk = (cap - pa_end) / (uint32_t)n_threads;
        for (int t = 0; t < n_threads; t++) {
            wargs[t].d = d;
            wargs[t].tid = t;
            wargs[t].n_threads = n_threads;
            wargs[t].base_idx = pa_end + (uint32_t)t * chunk;
            wargs[t].max_idx = chunk;
            wargs[t].local_n = 0;
            wargs[t].d5_end = d5_end;
            wargs[t].d6_end = d6_end;
            wargs[t].phase_a_end = pa_end;
            memset(wargs[t].bcnt, 0, sizeof(wargs[t].bcnt));
            for (int bl = 1; bl <= 64; bl++) {
                int32_t rem_c = d->bucket_cap[bl] - d->bucket_cnt[bl];
                wargs[t].bcap[bl] = (rem_c > 0) ? (rem_c / n_threads + 1) : 1;
            }
        }
#ifdef _WIN32
        HANDLE th[8];
        for (int t = 0; t < n_threads; t++) th[t] = CreateThread(NULL, 0, win_univ_worker, &wargs[t], 0, NULL);
        WaitForMultipleObjects((DWORD)n_threads, th, TRUE, INFINITE);
        for (int t = 0; t < n_threads; t++) CloseHandle(th[t]);
#else
        pthread_t th[8];
        for (int t = 0; t < n_threads; t++) pthread_create(&th[t], NULL, posix_univ_worker, &wargs[t]);
        for (int t = 0; t < n_threads; t++) pthread_join(th[t], NULL);
#endif
        /* Skip multi-GB HT compaction: mark seen24 linearly in ~0.2s, leave shard arrays in place! */
        for (int t = 0; t < n_threads; t++) {
            uint32_t base = wargs[t].base_idx;
            uint32_t ln = wargs[t].local_n;
            for (uint32_t k = 0; k < ln; k++) {
                uint64_t vv = d->vals[base + k];
                if (vv < (1ULL << 24)) d->seen24[vv >> 6] |= (1ULL << (vv & 63));
            }
            total_valid += ln;
        }
        d->n = total_valid;
    }

    /* Extract compact high-yield divisor array `scan_vals` (~300k items: 100k <=23b + 200k 24..34b) */
    uint32_t max_udiv = 300000U;
    d->scan_vals = (uint64_t *)malloc(sizeof(uint64_t) * max_udiv);
    d->scan_costs = (uint8_t *)malloc(sizeof(uint8_t) * max_udiv);
    if (d->scan_vals && d->scan_costs) {
        uint32_t cnt_lo = 0, cnt_mid = 0, out_u = 0;
        for (uint32_t k = 1; k < cap && out_u < max_udiv; k++) {
            uint64_t v = d->vals[k];
            if (v <= 1ULL) continue;
            int bl = bl64(v);
            if (bl <= 23 && cnt_lo < 100000U) {
                d->scan_vals[out_u] = v;
                d->scan_costs[out_u] = meta_cost(d->meta[k]);
                cnt_lo++; out_u++;
            } else if (bl >= 24 && bl <= 34 && cnt_mid < 200000U) {
                d->scan_vals[out_u] = v;
                d->scan_costs[out_u] = meta_cost(d->meta[k]);
                cnt_mid++; out_u++;
            }
            if (cnt_lo >= 100000U && cnt_mid >= 200000U) break;
        }
        d->n_scan = out_u;
        d->alloc_bytes += (uint64_t)(sizeof(uint64_t) + sizeof(uint8_t)) * max_udiv;
    }
    return d;
}

EXPORT void fe_dict_free(void *p) { free_bigdict((BigDict *)p); }
EXPORT uint32_t fe_dict_size(void *p) { return p ? ((BigDict *)p)->n : 0U; }
EXPORT uint32_t fe_scan_size(void *p) { return p ? ((BigDict *)p)->n_scan : 0U; }
EXPORT uint64_t fe_dict_bytes(void *p) { return p ? ((BigDict *)p)->alloc_bytes : 0ULL; }

EXPORT int32_t fe_dict_lookup(void *p, uint64_t v) {
    return bd_find((const BigDict *)p, v);
}

EXPORT void fe_dict_get(
    void *p, uint32_t idx, uint64_t *val, uint64_t *va, uint64_t *vb, int *op, int *cost
) {
    const BigDict *d = (const BigDict *)p;
    if (!d || idx >= d->cap || d->vals[idx] == 0ULL) { *op = -1; return; }
    uint64_t m = d->meta[idx];
    uint32_t pa = meta_pa(m), pb = meta_pb(m);
    *val = d->vals[idx];
    *cost = (int)meta_cost(m);
    if (pa == EMPTY_PA) {
        *op = 0; *va = 0; *vb = 0;
    } else {
        *op = (int)meta_op(m) + 1;
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
 * Fast Per-Puzzle Seed Dictionary `D_local` (builds up to 10M nodes cleanly in ~1-2s).
 */
EXPORT void* fe_local_create(
    const uint64_t *seeds, int ns, uint64_t vl, uint32_t cap, uint32_t ht_pow,
    uint32_t per_b, uint32_t scan_cap
) {
    BigDict *d = alloc_bigdict(cap, ht_pow, vl);
    if (!d) return NULL;
    for (int bl = 1; bl <= 64; bl++) d->bucket_cap[bl] = (int32_t)per_b;

    for (int i = 0; i < ns; i++) bd_add(d, seeds[i], EMPTY_PA, EMPTY_PA, OP_ADD, 0, 0);
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

    /* Power-of-2 products */
    for (int pass = 0; pass < 3; pass++) {
        uint32_t lim = d->n;
        for (uint32_t i = 0; i < lim && d->n < cap; i++) {
            uint64_t a = d->vals[i]; if ((a & (a - 1)) != 0) continue;
            uint8_t ca = meta_cost(d->meta[i]);
            for (uint32_t j = i; j < lim && d->n < cap; j++) {
                uint64_t b = d->vals[j]; if ((b & (b - 1)) != 0) continue;
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

    /* Layer 2: bounded representative pairs (~10,000 reps -> fast) */
    uint32_t max_reps = 44U * 220U;
    uint32_t *reps = (uint32_t *)malloc(sizeof(uint32_t) * max_reps);
    uint32_t nr = 0;
    if (reps) {
        for (int bl = 1; bl <= 40 && nr < max_reps; bl++) {
            int cnt = 0;
            for (int tc = 0; tc <= 11 && cnt < 220 && nr < max_reps; tc++) {
                for (uint32_t k = 0; k < d->n && cnt < 220 && nr < max_reps; k++) {
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
                if (ca + cb + 1 > 16) continue;
                uint8_t nc = (ca + cb + 1) > 15 ? 15 : (ca + cb + 1);
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

    /* Layer 3 (fast, bounded at <= 150k x 600 cheap base constants -> <0.8s!) */
    uint32_t cur_n = d->n;
    if (d->n < cap && cur_n > base_end) {
        uint32_t j_lim = base_end < 600U ? base_end : 600U;
        uint32_t i_step = (cur_n - base_end) > 150000U ? ((cur_n - base_end) / 150000U) : 1U;
        for (uint32_t i = base_end; i < cur_n && d->n < cap; i += i_step) {
            uint64_t a = d->vals[i];
            if (bl64(a) > 38) continue;
            uint8_t ca = meta_cost(d->meta[i]);
            for (uint32_t j = 0; j < j_lim && d->n < cap; j++) {
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

    /* Build stratified sorted scan slice up to scan_cap */
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
        d->alloc_bytes += (uint64_t)(sizeof(uint64_t) + sizeof(uint8_t)) * ns_tot;
    }
    free(sp);
    return d;
}

typedef struct { uint64_t d1, d2, prod, gap; int32_t metric; uint8_t sgn; } Hit;

/*
 * Fast O(N) Two-Pointer Scan with zero multi-GB DRAM misses:
 * Uses L3-resident `seen24` bitsets for instant ~1ns residual hit checks!
 */
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

    const uint64_t *d_s24 = d->seen24;
    const uint64_t *u_s24 = u ? u->seen24 : NULL;
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
                    if (g < (1ULL << 24)) {
                        if (d_s24 && (d_s24[g >> 6] & (1ULL << (g & 63)))) g_est = 4;
                        else if (u_s24 && (u_s24[g >> 6] & (1ULL << (g & 63)))) g_est = 5;
                        else g_est = bl64(g) / 3 + 2;
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
 * Bidirectional C99 MITM probe `T = a * q +- r`:
 * Scans active DAG nodes + D_local + high-yield Universal U divisors (`u->scan_vals`,
 * covering `<2^20` and `24..34`-bit bands!).
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

    for (int i = 0; i < n_act; i++) {
        TRY_DIVISOR(act_vals[i], 0);
    }

    uint32_t d_lim = d->n < max_div_scan ? d->n : max_div_scan;
    for (uint32_t i = 0; i < d_lim; i++) {
        TRY_DIVISOR(d->vals[i], (int)meta_cost(d->meta[i]));
    }

    if (u && max_div_scan > 0) {
        if (u->scan_vals && u->n_scan > 0) {
            uint32_t u_lim = u->n_scan < max_div_scan ? u->n_scan : max_div_scan;
            for (uint32_t i = 0; i < u_lim; i++) {
                TRY_DIVISOR(u->scan_vals[i], (int)u->scan_costs[i] + 1);
            }
        } else {
            uint32_t u_lim = u->n < max_div_scan ? u->n : max_div_scan;
            for (uint32_t i = 1; i < u_lim; i++) {
                if (u->vals[i] > 1ULL) TRY_DIVISOR(u->vals[i], (int)meta_cost(u->meta[i]) + 1);
            }
        }
    }

    #undef TRY_DIVISOR

    for (int i = 0; i < nh; i++) {
        oa[i] = hits[i].a; oq[i] = hits[i].q; or_[i] = hits[i].r;
        osgn[i] = (int)hits[i].sgn; osc[i] = (int)hits[i].sc;
    }
    return nh;
}

EXPORT void fe_screen_smooth(
    uint64_t tgt, uint64_t vl, const int32_t *ks, int nk,
    const uint32_t *primes, int np, int32_t *out_sc
) {
    for (int i = 0; i < nk; i++) {
        int64_t k = (int64_t)ks[i];
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
            while (n % p == 0) { n /= p; facs++; }
        }
        int rb = (n <= 1ULL) ? 0 : bl64(n);
        out_sc[i] = facs + (rb > 0 ? (rb / 3 + 2) : 0) + (k == 0 ? 0 : 1);
    }
}
'''

_CLIB_CACHE = None
_CLIB_FAILED = False
_OP_MAP = {1: "+", 2: "-", 3: "*", 4: "/"}

_UNIV_CTX: Optional[int] = None
_UNIV_CAP: int = 0
_UNIV_SIZE: int = 0
_UNIV_BYTES: int = 0


def get_clib() -> Optional[ctypes.CDLL]:
    """Lazily compile and load the V6 4-thread sharded C99 shared library."""
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
        so_path = os.path.join(tdir, f"santa2026_fast_engine_v6{ext}")
        if not os.path.exists(so_path):
            pid = os.getpid()
            c_path = os.path.join(tdir, f"santa2026_fast_engine_v6_{pid}.c")
            tmp_so = os.path.join(tdir, f"santa2026_fast_engine_v6_{pid}{ext}")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(_C_SOURCE)
            flags = ["gcc", "-O3", "-shared", "-fPIC"] + ([] if os.name == "nt" else ["-pthread"])
            subprocess.run(
                flags + [c_path, "-o", tmp_so],
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
        lib.fe_univ_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
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
        lib.fe_dict_bytes.argtypes = [ctypes.c_void_p]
        lib.fe_dict_bytes.restype = ctypes.c_uint64
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
    ram_gb: str | float = "auto", *, force_python: bool = False, n_threads: Optional[int] = None
) -> dict[str, int | float | str]:
    """
    Build the global parent-shared read-only `{1}` Universal Dictionary `U` once across
    `n_threads` OS threads. Shared 1x Copy-on-Write across Linux `fork()` workers.
    """
    global _UNIV_CTX, _UNIV_CAP, _UNIV_SIZE, _UNIV_BYTES
    prof = dict(get_ram_profile(ram_gb))
    if force_python:
        return {**prof, "univ_size": 0, "univ_gb": 0.0, "c99": 0}

    lib = get_clib()
    if lib is None:
        return {**prof, "univ_size": 0, "univ_gb": 0.0, "c99": 0}

    target_cap = int(prof["univ_cap"])
    target_pow = int(prof["univ_ht_pow"])
    if _UNIV_CTX is not None and _UNIV_CAP >= target_cap:
        return {**prof, "univ_size": _UNIV_SIZE, "univ_gb": _UNIV_BYTES / 1e9, "c99": 1}

    if _UNIV_CTX is not None:
        lib.fe_dict_free(_UNIV_CTX)
        _UNIV_CTX = None
        _UNIV_CAP = 0
        _UNIV_SIZE = 0
        _UNIV_BYTES = 0

    nt = n_threads if n_threads is not None else min(4, max(1, os.cpu_count() or 1))
    cap = target_cap
    htp = target_pow
    while cap >= 2_000_000:
        ctx = lib.fe_univ_create(ctypes.c_uint32(cap), ctypes.c_uint32(htp), ctypes.c_int(nt))
        if ctx:
            _UNIV_CTX = ctx
            _UNIV_CAP = cap
            _UNIV_SIZE = int(lib.fe_dict_size(ctx))
            _UNIV_BYTES = int(lib.fe_dict_bytes(ctx))
            return {
                **prof,
                "univ_cap": cap,
                "univ_ht_pow": htp,
                "univ_size": _UNIV_SIZE,
                "univ_gb": _UNIV_BYTES / 1e9,
                "c99": 1,
            }
        cap = (cap * 2) // 3
        if htp > 23 and (1 << (htp - 1)) >= cap * 2:
            htp -= 1

    return {**prof, "univ_size": 0, "univ_gb": 0.0, "c99": 0}


def release_universal_dict() -> None:
    """Free the global Universal Dictionary cleanly."""
    global _UNIV_CTX, _UNIV_CAP, _UNIV_SIZE, _UNIV_BYTES
    if _UNIV_CTX is not None and _CLIB_CACHE is not None:
        _CLIB_CACHE.fe_dict_free(_UNIV_CTX)
    _UNIV_CTX = None
    _UNIV_CAP = 0
    _UNIV_SIZE = 0
    _UNIV_BYTES = 0


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
    Unified interface to the V6 C99 Universal+Local (or pure-Python fallback)
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
        self._cost_cache: dict[int, int] = {}
        self._tp_cache: dict[int, list[tuple[int, int, int, int, int]]] = {}
        self._mitm_cache: dict[int, list[tuple[int, int, int, int, int]]] = {}
        self.ctx = None
        self.univ_ctx = None
        self.py_backend: Optional[_PyDictBackend] = None

        self.clib = None if force_python else get_clib()
        if self.clib is not None and self.vl < (1 << 63) and self.seeds:
            arr = (ctypes.c_uint64 * len(self.seeds))(*self.seeds)
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

    def lookup_cost(self, v: int) -> int:
        """Return estimated DAG op cost of `v` in `D_local` / `U` (or 99 if absent)."""
        if v in self.seeds_set:
            return 0
        if v <= 0 or v > self.vl:
            return 99
        c_hit = self._cost_cache.get(v)
        if c_hit is not None:
            return c_hit
        if self.py_backend is not None:
            c = self.py_backend.cost.get(v, 99)
            self._cost_cache[v] = c
            return c
        idx = self.clib.fe_dict_lookup(self.ctx, ctypes.c_uint64(v))
        src_ctx = self.ctx
        penalty = 0
        if idx < 0 and self.univ_ctx is not None:
            idx = self.clib.fe_dict_lookup(self.univ_ctx, ctypes.c_uint64(v))
            src_ctx = self.univ_ctx
            penalty = 3  # Penalize `{1}`-rooted `U` depth slightly vs `D_local` seed reuse
        if idx < 0:
            self._cost_cache[v] = 99
            return 99
        vp = ctypes.c_uint64()
        ap = ctypes.c_uint64()
        bp = ctypes.c_uint64()
        op = ctypes.c_int()
        cp = ctypes.c_int()
        self.clib.fe_dict_get(
            src_ctx, ctypes.c_uint32(idx), ctypes.byref(vp), ctypes.byref(ap),
            ctypes.byref(bp), ctypes.byref(op), ctypes.byref(cp)
        )
        res_c = max(1, cp.value + penalty) if op.value >= 0 else 99
        self._cost_cache[v] = res_c
        return res_c

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
          - Caches the static `D_local U U` divisor scan ONCE per `tgt`.
          - Always probes current `active_vals` (marginal cost = 0) dynamically.
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

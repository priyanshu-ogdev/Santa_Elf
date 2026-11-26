"""
src/solvers/fast_engine.py
V8 High-RAM 4-Thread Sharded 128-bit Universal + Local Dictionary, C99 MITM,
and 64-bit Value-Set Beam Engine.

Key Engineering & Speed Upgrades:
  1. 4-Thread Lock-Free 64-Shard Universal `{1}`-Dictionary (`U`):
     Uses 64 hash shards (`(hash64(v) >> 58) & 63`) so all 4 CPU threads populate
     independent shards + disjoint value slices with zero locks/atomics (~6 M nodes/s).
     On Kaggle (~30-31 GB RAM), builds `240,000,000` packed 16-byte nodes + `2^30`
     4-byte hash slots (`~8.14 GB` Read-Only Copy-on-Write across all 4 workers),
     freeing per-worker RAM & cache bandwidth for `D_local = 36,000,000`.
  2. Zero-Stall `two_pointer` L3 Bitset:
     Retains the `2 MiB` `seen24` (`< 2^24`) bitset so `fe_two_pointer` evaluates
     gap reachability in ~1ns in L3 cache instead of missing DRAM in a multi-GB HT.
  3. High-Yield Divisor Slice (`u->scan_vals`) & `< 2^28` MITM Near-Misses:
     Caches balanced `< 2^20` + `24..34`-bit divisors in `U` for targeted MITM probing.
  4. C99 64-bit 2-Step Value-Set Beam Search (`fe_value_beam64`):
     Explores joint shared intermediates across small/medium targets (`vl <= 25M`).
"""
from __future__ import annotations
import ctypes
import os
import shutil
import subprocess
import tempfile
from typing import Optional
from src.utils import Puzzle, Instruction


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
        _named_profiles = {
            "kaggle_high_ram": 25.0,
            "kaggle": 25.0,
            "high_ram_16g": 16.0,
            "high": 16.0,
            "medium_ram": 8.0,
            "medium": 8.0,
            "compact_ram": 4.0,
            "low": 4.0,
        }
        key_ram = ram_gb.strip().lower()
        if key_ram in _named_profiles:
            eff_gb = _named_profiles[key_ram]
        else:
            try:
                eff_gb = float(ram_gb)
            except ValueError:
                eff_gb = avail_gb
    else:
        eff_gb = max(avail_gb, 25.0) if (on_kaggle and total_gb >= 24.0) else avail_gb

    if eff_gb >= 22.0:
        # Kaggle ~13-19 GB aggregate budget across 31.35 GB host:
        #   U: 240M nodes x 16B (3.84 GB) + HT=2^30 x 4B (4.29 GB) = ~8.14 GB Read-Only CoW
        #   + 4 workers x (24M local ~652 MB + 10M radix-sorted scan ~90 MB + MITM + MILP) ~ 5-9 GB
        return {
            "mode": "kaggle_high_ram",
            "eff_gb": eff_gb,
            "univ_cap": 240_000_000,
            "univ_ht_pow": 30,
            "local_cap": 24_000_000,
            "local_ht_pow": 26,
            "local_scan": 10_000_000,
            "per_bucket": 500_000,
            "mitm_div_scan": 420_000,
            "smooth_offset_lim": 256,
            "milp_max_edges": 7000,
            "milp_time": 2.5,
            "compiler_trials": 120,
            "superopt_time": 2.0,
        }
    if eff_gb >= 14.0:
        return {
            "mode": "high_ram_16g",
            "eff_gb": eff_gb,
            "univ_cap": 240_000_000,
            "univ_ht_pow": 30,
            "local_cap": 4_000_000,
            "local_ht_pow": 24,
            "local_scan": 2_500_000,
            "per_bucket": 75_000,
            "mitm_div_scan": 260_000,
            "smooth_offset_lim": 200,
            "milp_max_edges": 4000,
            "milp_time": 1.8,
            "compiler_trials": 80,
            "superopt_time": 1.6,
        }
    if eff_gb >= 6.5:
        return {
            "mode": "medium_ram",
            "eff_gb": eff_gb,
            "univ_cap": 60_000_000,
            "univ_ht_pow": 27,
            "local_cap": 1_800_000,
            "local_ht_pow": 22,
            "local_scan": 1_200_000,
            "per_bucket": 35_000,
            "mitm_div_scan": 200_000,
            "smooth_offset_lim": 160,
            "milp_max_edges": 3000,
            "milp_time": 1.2,
            "compiler_trials": 64,
            "superopt_time": 1.4,
        }
    return {
        "mode": "compact_ram",
        "eff_gb": eff_gb,
        "univ_cap": 8_000_000,
        "univ_ht_pow": 25,
        "local_cap": 800_000,
        "local_ht_pow": 21,
        "local_scan": 800_000,
        "per_bucket": 16_000,
        "mitm_div_scan": 160_000,
        "smooth_offset_lim": 96,
        "milp_max_edges": 2000,
        "milp_time": 0.6,
        "compiler_trials": 64,
        "superopt_time": 1.2,
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

static void radix_sort_sp(SortPair *a, uint32_t n) {
    if (n <= 1U) return;
    if (n <= 2048U) { qsort(a, n, sizeof(SortPair), cmp_sp); return; }
    SortPair *tmp = (SortPair *)malloc((size_t)n * sizeof(SortPair));
    if (!tmp) { qsort(a, n, sizeof(SortPair), cmp_sp); return; }
    SortPair *src = a, *dst = tmp;
    uint32_t cnt[256], off[256];
    for (int shift = 0; shift < 64; shift += 8) {
        memset(cnt, 0, sizeof(cnt));
        for (uint32_t i = 0; i < n; i++) cnt[(src[i].v >> shift) & 255U]++;
        uint32_t fb = (uint32_t)((src[0].v >> shift) & 255U);
        if (cnt[fb] == n) continue;
        uint32_t sum = 0U;
        for (int b = 0; b < 256; b++) { off[b] = sum; sum += cnt[b]; }
        for (uint32_t i = 0; i < n; i++) {
            uint32_t b = (uint32_t)((src[i].v >> shift) & 255U);
            dst[off[b]++] = src[i];
        }
        SortPair *sw = src; src = dst; dst = sw;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof(SortPair));
    free(tmp);
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
    while ((1ULL << ht_pow) < (uint64_t)cap * 2ULL && ht_pow < 31U) ht_pow++;
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

    /* Extract compact high-yield divisor array `scan_vals` (~1.2M items: 400k <=23b + 800k 24..34b) */
    uint32_t max_udiv = 1200000U;
    d->scan_vals = (uint64_t *)malloc(sizeof(uint64_t) * max_udiv);
    d->scan_costs = (uint8_t *)malloc(sizeof(uint8_t) * max_udiv);
    if (d->scan_vals && d->scan_costs) {
        uint32_t cnt_lo = 0, cnt_mid = 0, out_u = 0;
        for (uint32_t k = 1; k < cap && out_u < max_udiv; k++) {
            uint64_t v = d->vals[k];
            if (v <= 1ULL) continue;
            int bl = bl64(v);
            if (bl <= 23 && cnt_lo < 400000U) {
                d->scan_vals[out_u] = v;
                d->scan_costs[out_u] = meta_cost(d->meta[k]);
                cnt_lo++; out_u++;
            } else if (bl >= 24 && bl <= 34 && cnt_mid < 800000U) {
                d->scan_vals[out_u] = v;
                d->scan_costs[out_u] = meta_cost(d->meta[k]);
                cnt_mid++; out_u++;
            }
            if (cnt_lo >= 400000U && cnt_mid >= 800000U) break;
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
    if (!d || !d->scan_vals || d->n_scan == 0) return 0U;
    for (uint32_t i = 0; i < d->n_scan; i++) out[i] = d->scan_vals[i];
    return d->n_scan;
}

EXPORT uint32_t fe_dict_copy_scan_prefix(void *p, uint64_t *out, uint32_t max_n) {
    const BigDict *d = (const BigDict *)p;
    if (!d || !d->scan_vals || d->n_scan == 0 || max_n == 0) return 0U;
    uint32_t n = d->n_scan;
    if (n <= max_n) {
        for (uint32_t i = 0; i < n; i++) out[i] = d->scan_vals[i];
        return n;
    }
    uint32_t half = max_n / 2U;
    for (uint32_t i = 0; i < half; i++) out[i] = d->scan_vals[i];
    uint64_t rem_slots = max_n - half;
    uint64_t avail = n - half;
    for (uint64_t s = 0; s < rem_slots; s++) {
        out[half + (uint32_t)s] = d->scan_vals[half + (uint32_t)((s * avail) / rem_slots)];
    }
    return max_n;
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

    /* Build cost-prioritized stratified sorted scan slice up to scan_cap via LSD Radix Sort */
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
            uint8_t *used = (uint8_t *)calloc((size_t)((d->n + 7U) >> 3), 1);
            if (used) {
                for (uint32_t i = 0; i < d->n && filled < ns_tot; i++) {
                    uint8_t cc = meta_cost(d->meta[i]);
                    if (cc <= 6) {
                        sp[filled].v = d->vals[i];
                        sp[filled].c = cc;
                        filled++;
                        used[i >> 3] |= (uint8_t)(1U << (i & 7U));
                    }
                }
                uint32_t p2_cap = (ns_tot * 85U) / 100U;
                if (filled < p2_cap) {
                    for (uint32_t i = 0; i < d->n && filled < p2_cap; i++) {
                        if (used[i >> 3] & (1U << (i & 7U))) continue;
                        uint8_t cc = meta_cost(d->meta[i]);
                        if (cc <= 8 && bl64(d->vals[i]) <= 38) {
                            sp[filled].v = d->vals[i];
                            sp[filled].c = cc;
                            filled++;
                            used[i >> 3] |= (uint8_t)(1U << (i & 7U));
                        }
                    }
                }
                if (filled < ns_tot) {
                    uint64_t rem_slots = ns_tot - filled;
                    uint64_t avail = d->n;
                    for (uint64_t s = 0; s < rem_slots && filled < ns_tot; s++) {
                        uint32_t idx = (uint32_t)((s * avail) / rem_slots);
                        while (idx < d->n && (used[idx >> 3] & (1U << (idx & 7U)))) idx++;
                        if (idx < d->n) {
                            sp[filled].v = d->vals[idx];
                            sp[filled].c = meta_cost(d->meta[idx]);
                            filled++;
                            used[idx >> 3] |= (uint8_t)(1U << (idx & 7U));
                        }
                    }
                }
                free(used);
            } else {
                for (uint64_t s = 0; s < ns_tot && filled < ns_tot; s++) {
                    uint32_t idx = (uint32_t)((s * d->n) / ns_tot);
                    sp[filled].v = d->vals[idx];
                    sp[filled].c = meta_cost(d->meta[idx]);
                    filled++;
                }
            }
        }
        radix_sort_sp(sp, filled);
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
 * Fast Bounded + Strided O(N/stride) Two-Pointer Scan:
 * Binary-searches upper product bound and uses L3-resident `seen24` bitsets.
 */
EXPORT int fe_two_pointer_s(
    void *p_local, void *p_univ, uint64_t tgt, int K, int stride,
    uint64_t *od1, uint64_t *od2, uint64_t *oprod, uint64_t *ogap, int *osgn
) {
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    if (!d || !d->scan_vals || d->n_scan == 0 || K <= 0) return 0;
    int N = (int)d->n_scan;
    if (K > 64) K = 64;
    if (stride < 1) stride = 1;

    Hit hits_gap[32]; int ng = 0, wg = 0;
    Hit hits_sc[32];  int ns = 0, ws = 0;
    int Kg = K / 2; if (Kg < 1) Kg = 1; if (Kg > 32) Kg = 32;
    int Ks = K - Kg; if (Ks > 32) Ks = 32;

    const uint64_t *d_s24 = d->seen24;
    const uint64_t *u_s24 = u ? u->seen24 : NULL;
    u128 t128 = (u128)tgt;
    u128 vl128 = (u128)d->vl;

    uint64_t a0 = d->scan_vals[0] > 0ULL ? d->scan_vals[0] : 1ULL;
    u128 max_b128 = ((t128 * 2ULL) / (u128)a0) + 4096ULL;
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if ((u128)d->scan_vals[mid] <= max_b128) lo = mid + 1;
        else hi = mid;
    }
    int i = 0, j = (lo > 0) ? (lo - 1) : 0;

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
        if (prod < t128) i += stride;
        else if (prod > t128) j -= stride;
        else break;
    }

    for (int a_i = 1; a_i < ng; a_i++) {
        Hit kv = hits_gap[a_i];
        int b_i = a_i - 1;
        while (b_i >= 0 && hits_gap[b_i].gap > kv.gap) {
            hits_gap[b_i + 1] = hits_gap[b_i];
            b_i--;
        }
        hits_gap[b_i + 1] = kv;
    }
    for (int a_i = 1; a_i < ns; a_i++) {
        Hit kv = hits_sc[a_i];
        int b_i = a_i - 1;
        while (b_i >= 0 && (hits_sc[b_i].metric > kv.metric ||
               (hits_sc[b_i].metric == kv.metric && hits_sc[b_i].gap > kv.gap))) {
            hits_sc[b_i + 1] = hits_sc[b_i];
            b_i--;
        }
        hits_sc[b_i + 1] = kv;
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

EXPORT int fe_two_pointer(
    void *p_local, void *p_univ, uint64_t tgt, int K,
    uint64_t *od1, uint64_t *od2, uint64_t *oprod, uint64_t *ogap, int *osgn
) {
    return fe_two_pointer_s(p_local, p_univ, tgt, K, 1, od1, od2, oprod, ogap, osgn);
}

typedef struct { uint64_t a, qq, prod, gap; int32_t sc; uint8_t sgn; } AncHit;

static inline int anc_gap_cost(const BigDict *d, const BigDict *u, uint64_t g) {
    if (g == 0ULL) return 0;
    if (g < (1ULL << 24)) {
        if (d->seen24 && (d->seen24[g >> 6] & (1ULL << (g & 63)))) {
            int32_t id = bd_find(d, g);
            if (id >= 0) return (int)meta_cost(d->meta[id]) + 1;
        }
        if (u && u->seen24 && (u->seen24[g >> 6] & (1ULL << (g & 63)))) {
            int32_t iu = bd_find(u, g);
            if (iu >= 0) return (int)meta_cost(u->meta[iu]) + 3;
        }
    } else if (g < (1ULL << 38)) {
        int32_t id = bd_find(d, g);
        if (id >= 0) return (int)meta_cost(d->meta[id]) + 1;
    }
    return bl64(g) / 3 + 4;
}

static inline void anc_push(
    AncHit *arr, int *n, int *w, int K, uint64_t a, uint64_t qq, uint64_t prod, uint64_t gap, int32_t sc, uint8_t sgn
) {
    if (*n < K) {
        arr[*n] = (AncHit){a, qq, prod, gap, sc, sgn};
        if (arr[*n].sc > arr[*w].sc) *w = *n;
        (*n)++;
    } else if (sc < arr[*w].sc) {
        arr[*w] = (AncHit){a, qq, prod, gap, sc, sgn};
        *w = 0;
        for (int h = 1; h < *n; h++) if (arr[h].sc > arr[*w].sc) *w = h;
    }
}

/*
 * Fast C99 Active-Anchor Binary Search over ALL live DAG nodes (not truncated!),
 * scoring proposals cleanly by true dict + gap bitset cost instead of raw magnitude.
 */
EXPORT int fe_anchor_scan(
    void *p_local, void *p_univ, uint64_t tgt, uint64_t vl,
    const uint64_t *act_vals, int n_act, int K,
    uint64_t *oa, uint64_t *oq, uint64_t *oprod, uint64_t *ogap, int *osgn, int *osc
) {
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    if (!d || K <= 0) return 0;
    if (K > 32) K = 32;
    AncHit hits[32];
    int nh = 0, wh = 0;
    uint32_t nD = d->n_scan;

    for (int i = 0; i < n_act; i++) {
        uint64_t a = act_vals[i];
        if (a <= 1ULL || a >= tgt || (a & (a - 1ULL)) == 0ULL) continue;
        uint64_t q_int = tgt / a;

        for (int pass = 0; pass < 2; pass++) {
            uint64_t qq = q_int + (uint64_t)pass;
            if (qq < 1ULL || (u128)a * (u128)qq > (u128)vl) continue;
            int cq = -1;
            if (qq == 1ULL) cq = 0;
            else if (qq <= 256ULL) cq = bl64(qq) / 2 + 2;
            else {
                int32_t id = bd_find(d, qq);
                if (id >= 0) cq = (int)meta_cost(d->meta[id]) + 1;
                else if (u) {
                    int32_t iu = bd_find(u, qq);
                    if (iu >= 0) cq = (int)meta_cost(u->meta[iu]) + 3;
                }
            }
            if (cq >= 0) {
                uint64_t prod = a * qq;
                uint64_t gap = (tgt >= prod) ? (tgt - prod) : (prod - tgt);
                uint8_t sgn = (tgt >= prod) ? 1 : 0;
                int32_t sc = cq + anc_gap_cost(d, u, gap) + (gap ? 1 : 0);
                anc_push(hits, &nh, &wh, K, a, qq, prod, gap, sc, sgn);
            }
        }

        if (nD > 0U) {
            uint32_t lo = 0U, hi = nD;
            while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                if (d->scan_vals[mid] < q_int) lo = mid + 1U;
                else hi = mid;
            }
            for (int off = -1; off <= 1; off++) {
                int64_t p = (int64_t)lo + off;
                if (p >= 0 && p < (int64_t)nD) {
                    uint64_t qq = d->scan_vals[p];
                    if (qq == 0ULL || (u128)a * (u128)qq > (u128)vl) continue;
                    uint64_t prod = a * qq;
                    uint64_t gap = (tgt >= prod) ? (tgt - prod) : (prod - tgt);
                    uint8_t sgn = (tgt >= prod) ? 1 : 0;
                    int32_t sc = (int32_t)d->scan_costs[p] + 1 + anc_gap_cost(d, u, gap) + (gap ? 1 : 0);
                    anc_push(hits, &nh, &wh, K, a, qq, prod, gap, sc, sgn);
                }
            }
        }
    }

    for (int i = 0; i < nh; i++) {
        oa[i] = hits[i].a; oq[i] = hits[i].qq; oprod[i] = hits[i].prod;
        ogap[i] = hits[i].gap; osgn[i] = (int)hits[i].sgn; osc[i] = (int)hits[i].sc;
    }
    return nh;
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
    if (!d || K <= 0 || tgt == 0ULL) return 0;
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
                    } else if (r < (1ULL << 28)) { \
                        mitm_push(hits, &nh, &wh, K, a_v, q, r, ca_v + cq + (bl64(r) / 3) + 5, 1); \
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
                    } else if (r2 < (1ULL << 28)) { \
                        mitm_push(hits, &nh, &wh, K, a_v, q + 1ULL, r2, ca_v + cq1 + (bl64(r2) / 3) + 5, 0); \
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

/* Batch probe active_defs x (D_local U U) in C99 in a single call to eliminate Python ctypes overhead. */
EXPORT int fe_find_active_1ops(
    void *p_local, void *p_univ, uint64_t tgt, uint64_t vl,
    const uint64_t *act_vals, int n_act, int max_hits,
    uint8_t *oop, uint64_t *oa, uint64_t *ob
) {
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    if (!d || max_hits <= 0 || tgt == 0ULL || tgt > vl) return 0;
    int nh = 0;
    for (int i = 0; i < n_act && nh < max_hits; i++) {
        uint64_t a = act_vals[i];
        if (a == 0ULL || a > vl) continue;
        if (a < tgt) {
            uint64_t b = tgt - a;
            if (bd_find(d, b) >= 0 || (u && bd_find(u, b) >= 0)) {
                oop[nh] = 1; oa[nh] = a; ob[nh] = b; nh++;
                if (nh >= max_hits) break;
            }
            if (a > 1ULL && tgt % a == 0ULL) {
                uint64_t q = tgt / a;
                if (bd_find(d, q) >= 0 || (u && bd_find(u, q) >= 0)) {
                    oop[nh] = 3; oa[nh] = a; ob[nh] = q; nh++;
                    if (nh >= max_hits) break;
                }
            }
        } else if (a > tgt) {
            uint64_t b = a - tgt;
            if (bd_find(d, b) >= 0 || (u && bd_find(u, b) >= 0)) {
                oop[nh] = 2; oa[nh] = a; ob[nh] = b; nh++;
                if (nh >= max_hits) break;
            }
            if (a % tgt == 0ULL) {
                uint64_t q = a / tgt;
                if (bd_find(d, q) >= 0 || (u && bd_find(u, q) >= 0)) {
                    oop[nh] = 4; oa[nh] = a; ob[nh] = q; nh++;
                    if (nh >= max_hits) break;
                }
            }
        }
        if (a > 1ULL && (u128)tgt * (u128)a <= (u128)vl) {
            uint64_t m = tgt * a;
            if (bd_find(d, m) >= 0 || (u && bd_find(u, m) >= 0)) {
                oop[nh] = 4; oa[nh] = m; ob[nh] = a; nh++;
            }
        }
    }
    return nh;
}

typedef struct {
    uint64_t v;
    uint64_t a;
    uint64_t b;
    uint8_t  op; /* 1='+', 2='-', 3='*' */
} ActPair;

typedef struct {
    ActPair *items;
    uint32_t n;
    uint64_t vl;
} ActPairSet;

static int cmp_ap(const void *x, const void *y) {
    const ActPair *px = (const ActPair *)x, *py = (const ActPair *)y;
    if (px->v != py->v) return (px->v < py->v) ? -1 : 1;
    if (px->op != py->op) return (px->op > py->op) ? -1 : 1;
    return (px->a < py->a) ? -1 : (px->a > py->a) ? 1 : 0;
}

static void radix_sort_ap(ActPair *a, uint32_t n) {
    if (n <= 1U) return;
    if (n <= 1024U) { qsort(a, n, sizeof(ActPair), cmp_ap); return; }
    ActPair *tmp = (ActPair *)malloc((size_t)n * sizeof(ActPair));
    if (!tmp) { qsort(a, n, sizeof(ActPair), cmp_ap); return; }
    ActPair *src = a, *dst = tmp;
    uint32_t cnt[256], off[256];
    for (int shift = 0; shift < 64; shift += 8) {
        memset(cnt, 0, sizeof(cnt));
        for (uint32_t i = 0; i < n; i++) cnt[(src[i].v >> shift) & 255U]++;
        uint32_t fb = (uint32_t)((src[0].v >> shift) & 255U);
        if (cnt[fb] == n) continue;
        uint32_t sum = 0U;
        for (int b = 0; b < 256; b++) { off[b] = sum; sum += cnt[b]; }
        for (uint32_t i = 0; i < n; i++) {
            uint32_t b = (uint32_t)((src[i].v >> shift) & 255U);
            dst[off[b]++] = src[i];
        }
        ActPair *sw = src; src = dst; dst = sw;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof(ActPair));
    free(tmp);
}

static inline int act_bsearch_has(const uint64_t *arr, int n, uint64_t val) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        uint64_t mv = arr[mid];
        if (mv == val) return 1;
        if (mv < val) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

EXPORT void fe_act_pairs_free(void *p) {
    ActPairSet *ap = (ActPairSet *)p;
    if (!ap) return;
    free(ap->items);
    free(ap);
}

EXPORT void* fe_act_pairs_create(const uint64_t *act, int na, uint64_t vl) {
    if (!act || na < 2 || vl < 2ULL) return NULL;
    if (na > 220) na = 220;
    uint32_t cap = (uint32_t)na * (uint32_t)na * 2U + 64U;
    ActPair *raw = (ActPair *)malloc((size_t)cap * sizeof(ActPair));
    if (!raw) return NULL;
    uint32_t cnt = 0U;
    u128 vl128 = (u128)vl;

    for (int i = 0; i < na; i++) {
        uint64_t a = act[i];
        if (a == 0ULL || a > vl) continue;
        if (a <= vl - a && (cnt + 1U) < cap) {
            uint64_t s2 = a + a;
            if (s2 >= 2ULL) raw[cnt++] = (ActPair){s2, a, a, 1};
        }
        if (a > 1ULL && (u128)a * (u128)a <= vl128 && (cnt + 1U) < cap) {
            uint64_t m2 = a * a;
            raw[cnt++] = (ActPair){m2, a, a, 3};
        }
        for (int j = i + 1; j < na; j++) {
            uint64_t b = act[j];
            if (b <= a || b > vl) continue;
            if (a <= vl - b && (cnt + 1U) < cap) {
                uint64_t s = a + b;
                if (s >= 2ULL) raw[cnt++] = (ActPair){s, a, b, 1};
            }
            if ((b - a) >= 2ULL && (cnt + 1U) < cap) {
                raw[cnt++] = (ActPair){b - a, b, a, 2};
            }
            if (a > 1ULL && (u128)a * (u128)b <= vl128 && (cnt + 1U) < cap) {
                raw[cnt++] = (ActPair){a * b, a, b, 3};
            }
        }
    }

    if (cnt == 0U) { free(raw); return NULL; }
    radix_sort_ap(raw, cnt);

    uint32_t out = 0U;
    for (uint32_t i = 0U; i < cnt; i++) {
        uint64_t v = raw[i].v;
        if (v < 2ULL) continue;
        if (out > 0U && raw[out - 1U].v == v) continue;
        if (act_bsearch_has(act, na, v)) continue;
        raw[out++] = raw[i];
    }
    if (out == 0U) { free(raw); return NULL; }

    ActPair *shrunk = (ActPair *)realloc(raw, (size_t)out * sizeof(ActPair));
    ActPairSet *ap = (ActPairSet *)malloc(sizeof(ActPairSet));
    if (!ap) { free(shrunk ? shrunk : raw); return NULL; }
    ap->items = shrunk ? shrunk : raw;
    ap->n = out;
    ap->vl = vl;
    return (void *)ap;
}

typedef struct {
    uint64_t a, b, comb, d, prod, gap;
    int32_t  sc;
    uint8_t  opab, sgn;
} ActHit;

static inline void act_push(
    ActHit *arr, int *n, int *w, int K, uint8_t by_gap,
    uint8_t opab, uint64_t a, uint64_t b, uint64_t comb,
    uint64_t d, uint64_t prod, uint64_t gap, uint8_t sgn, int32_t sc
) {
    for (int h = 0; h < *n; h++) {
        if (arr[h].prod == prod && arr[h].comb == comb) {
            if (sc < arr[h].sc) {
                arr[h] = (ActHit){a, b, comb, d, prod, gap, sc, opab, sgn};
                *w = 0;
                for (int k = 1; k < *n; k++) {
                    if (by_gap ? (arr[k].gap > arr[*w].gap || (arr[k].gap == arr[*w].gap && arr[k].sc > arr[*w].sc))
                               : (arr[k].sc > arr[*w].sc || (arr[k].sc == arr[*w].sc && arr[k].gap > arr[*w].gap))) *w = k;
                }
            }
            return;
        }
    }
    if (*n < K) {
        arr[*n] = (ActHit){a, b, comb, d, prod, gap, sc, opab, sgn};
        int idx = *n;
        (*n)++;
        if (*n == 1) *w = 0;
        else if (by_gap ? (arr[idx].gap > arr[*w].gap || (arr[idx].gap == arr[*w].gap && arr[idx].sc > arr[*w].sc))
                        : (arr[idx].sc > arr[*w].sc || (arr[idx].sc == arr[*w].sc && arr[idx].gap > arr[*w].gap))) *w = idx;
    } else {
        int repl = by_gap ? (gap < arr[*w].gap || (gap == arr[*w].gap && sc < arr[*w].sc))
                          : (sc < arr[*w].sc || (sc == arr[*w].sc && gap < arr[*w].gap));
        if (repl) {
            arr[*w] = (ActHit){a, b, comb, d, prod, gap, sc, opab, sgn};
            *w = 0;
            for (int k = 1; k < *n; k++) {
                if (by_gap ? (arr[k].gap > arr[*w].gap || (arr[k].gap == arr[*w].gap && arr[k].sc > arr[*w].sc))
                           : (arr[k].sc > arr[*w].sc || (arr[k].sc == arr[*w].sc && arr[k].gap > arr[*w].gap))) *w = k;
            }
        }
    }
}

EXPORT int fe_act_probe(
    void *p_act, void *p_local, void *p_univ,
    const uint64_t *act_vals, int n_act,
    uint64_t tgt, uint64_t vl, int K,
    uint8_t *o_opab, uint64_t *oa, uint64_t *ob, uint64_t *ocomb,
    uint64_t *od, uint64_t *oprod, uint64_t *ogap, int *osgn, int *osc
) {
    const ActPairSet *ap = (const ActPairSet *)p_act;
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    if (!ap || !d || ap->n == 0U || tgt == 0ULL || K <= 0) return 0;
    if (K > 32) K = 32;

    ActHit hits_sc[16];  int ns = 0, ws = 0;
    ActHit hits_gap[16]; int ng = 0, wg = 0;
    int Ks = (K + 1) / 2; if (Ks > 16) Ks = 16; if (Ks < 1) Ks = 1;
    int Kg = K - Ks; if (Kg > 16) Kg = 16;

    const uint64_t *d_s24 = d->seen24;
    const uint64_t *u_s24 = u ? u->seen24 : NULL;
    uint32_t np = ap->n;
    u128 t128 = (u128)tgt;
    u128 vl128 = (u128)vl;

    /* Probe 1: Direct E +- gap around tgt */
    uint32_t lo = 0U, hi = np;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (ap->items[mid].v < tgt) lo = mid + 1U;
        else hi = mid;
    }
    int64_t p_lo = (int64_t)lo - 16;
    if (p_lo < 0) p_lo = 0;
    int64_t p_hi = (int64_t)lo + 16;
    if (p_hi > (int64_t)np) p_hi = (int64_t)np;
    for (int64_t idx = p_lo; idx < p_hi; idx++) {
        ActPair e = ap->items[idx];
        uint64_t gap = (e.v <= tgt) ? (tgt - e.v) : (e.v - tgt);
        uint8_t sgn = (e.v <= tgt) ? 1 : 0;
        int32_t gcost;
        if (gap == 0ULL) gcost = 0;
        else if (act_vals && n_act > 0 && act_bsearch_has(act_vals, n_act, gap)) gcost = 1;
        else gcost = anc_gap_cost(d, u, gap);
        int32_t sc = 1 + (gap == 0ULL ? 0 : gcost);
        act_push(hits_sc, &ns, &ws, Ks, 0, e.op, e.a, e.b, e.v, 1ULL, e.v, gap, sgn, sc);
        if (Kg > 0 && gap < (1ULL << 28)) {
            act_push(hits_gap, &ng, &wg, Kg, 1, e.op, e.a, e.b, e.v, 1ULL, e.v, gap, sgn, sc);
        }
    }

    /* Probe 2: Quotient T ~ E * q +- gap where q in (D_local U U) */
    uint64_t half_t = tgt >> 1;
    for (uint32_t i = 0U; i < np; i++) {
        ActPair e = ap->items[i];
        uint64_t ev = e.v;
        if (ev > half_t) break;
        uint64_t q = tgt / ev;
        uint64_t r = tgt % ev;
        if (q >= 2ULL) {
            int cq = -1;
            if (q < (1ULL << 24)) {
                if ((d_s24 && (d_s24[q >> 6] & (1ULL << (q & 63)))) ||
                    (u_s24 && (u_s24[q >> 6] & (1ULL << (q & 63))))) {
                    cq = lookup_cost_both(d, u, q);
                }
            } else if (q < (1ULL << 36)) {
                int32_t id = bd_find(d, q);
                if (id >= 0) cq = (int)meta_cost(d->meta[id]);
            }
            if (cq >= 0) {
                int32_t gcost;
                if (r == 0ULL) gcost = 0;
                else if (act_vals && n_act > 0 && act_bsearch_has(act_vals, n_act, r)) gcost = 1;
                else gcost = anc_gap_cost(d, u, r);
                int32_t sc = 2 + cq + (r == 0ULL ? 0 : gcost);
                act_push(hits_sc, &ns, &ws, Ks, 0, e.op, e.a, e.b, ev, q, ev * q, r, 1, sc);
                if (Kg > 0 && r < (1ULL << 28)) {
                    act_push(hits_gap, &ng, &wg, Kg, 1, e.op, e.a, e.b, ev, q, ev * q, r, 1, sc);
                }
            }
        }
        if (r > 0ULL && (u128)ev * (u128)(q + 1ULL) <= vl128) {
            uint64_t q1 = q + 1ULL;
            uint64_t r2 = ev - r;
            int cq1 = -1;
            if (q1 < (1ULL << 24)) {
                if ((d_s24 && (d_s24[q1 >> 6] & (1ULL << (q1 & 63)))) ||
                    (u_s24 && (u_s24[q1 >> 6] & (1ULL << (q1 & 63))))) {
                    cq1 = lookup_cost_both(d, u, q1);
                }
            } else if (q1 < (1ULL << 36)) {
                int32_t id = bd_find(d, q1);
                if (id >= 0) cq1 = (int)meta_cost(d->meta[id]);
            }
            if (cq1 >= 0) {
                int32_t gcost;
                if (act_vals && n_act > 0 && act_bsearch_has(act_vals, n_act, r2)) gcost = 1;
                else gcost = anc_gap_cost(d, u, r2);
                int32_t sc = 2 + cq1 + gcost;
                act_push(hits_sc, &ns, &ws, Ks, 0, e.op, e.a, e.b, ev, q1, ev * q1, r2, 0, sc);
                if (Kg > 0 && r2 < (1ULL << 28)) {
                    act_push(hits_gap, &ng, &wg, Kg, 1, e.op, e.a, e.b, ev, q1, ev * q1, r2, 0, sc);
                }
            }
        }
    }

    /* Probe 3: Two-Pointer E * d_local +- gap over sorted ActPairSet x d->scan_vals */
    if (d->scan_vals && d->n_scan > 0U && tgt > 100ULL) {
        int nd = (int)d->n_scan;
        uint64_t e0 = ap->items[0].v > 0ULL ? ap->items[0].v : 1ULL;
        u128 max_d128 = ((t128 * 2ULL) / (u128)e0) + 4096ULL;
        int d_lo = 0, d_hi = nd;
        while (d_lo < d_hi) {
            int mid = d_lo + ((d_hi - d_lo) >> 1);
            if ((u128)d->scan_vals[mid] <= max_d128) d_lo = mid + 1;
            else d_hi = mid;
        }
        int j = (d_lo > 0) ? (d_lo - 1) : 0;
        int stride = (d_lo > 300000) ? (d_lo / 300000) : 1;
        uint32_t i = 0U;
        while (i < np && j >= 0) {
            uint64_t dv = d->scan_vals[j];
            if (dv <= 1ULL) break;
            ActPair e = ap->items[i];
            u128 prod = (u128)e.v * (u128)dv;
            if (prod <= vl128) {
                uint64_t g = (prod <= t128) ? (uint64_t)(t128 - prod) : (uint64_t)(prod - t128);
                uint8_t sg = (prod <= t128) ? 1 : 0;
                if (Kg > 0 && g < (1ULL << 28)) {
                    int32_t g_est = (g == 0ULL) ? 0 : (bl64(g) / 3 + 4);
                    int32_t sc_g = 2 + (int32_t)d->scan_costs[j] + g_est;
                    act_push(hits_gap, &ng, &wg, Kg, 1, e.op, e.a, e.b, e.v, dv, (uint64_t)prod, g, sg, sc_g);
                }
                int32_t g_est = 0;
                if (g > 0ULL) {
                    if (g < (1ULL << 24)) {
                        if ((d_s24 && (d_s24[g >> 6] & (1ULL << (g & 63)))) ||
                            (u_s24 && (u_s24[g >> 6] & (1ULL << (g & 63))))) g_est = 4;
                        else g_est = bl64(g) / 3 + 3;
                    } else {
                        g_est = bl64(g) / 3 + 4;
                    }
                }
                int32_t sc = 2 + (int32_t)d->scan_costs[j] + g_est;
                act_push(hits_sc, &ns, &ws, Ks, 0, e.op, e.a, e.b, e.v, dv, (uint64_t)prod, g, sg, sc);
            }
            if (prod < t128) i++;
            else if (prod > t128) j -= stride;
            else break;
        }
    }

    ActHit combined[32];
    int out = 0;
    for (int h = 0; h < ns; h++) combined[out++] = hits_sc[h];
    for (int h = 0; h < ng; h++) {
        int dup = 0;
        for (int k = 0; k < out; k++) {
            if (combined[k].prod == hits_gap[h].prod && combined[k].comb == hits_gap[h].comb) {
                dup = 1; break;
            }
        }
        if (!dup && out < 32) combined[out++] = hits_gap[h];
    }
    for (int a_i = 1; a_i < out; a_i++) {
        ActHit kv = combined[a_i];
        int b_i = a_i - 1;
        while (b_i >= 0 && (combined[b_i].sc > kv.sc ||
               (combined[b_i].sc == kv.sc && combined[b_i].gap > kv.gap))) {
            combined[b_i + 1] = combined[b_i];
            b_i--;
        }
        combined[b_i + 1] = kv;
    }
    if (out > K) out = K;
    for (int h = 0; h < out; h++) {
        o_opab[h] = combined[h].opab;
        oa[h] = combined[h].a;
        ob[h] = combined[h].b;
        ocomb[h] = combined[h].comb;
        od[h] = combined[h].d;
        oprod[h] = combined[h].prod;
        ogap[h] = combined[h].gap;
        osgn[h] = (int)combined[h].sgn;
        osc[h] = (int)combined[h].sc;
    }
    return out;
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

#define VB_MAX_POOL 120
#define VB_K_CHILD  32
#define VB_SEEN_HT  65536U
#define VB_SET_HT   65536U

typedef struct {
    int32_t parent;
    uint8_t op;
    uint64_t a, b, r;
} VBHist;

typedef struct {
    uint64_t vals[VB_MAX_POOL];
    uint32_t rem_mask;
    int32_t nv;
    int32_t hist_id;
} VBState;

typedef struct {
    uint64_t need;
    uint8_t ti;
} VBReq;

typedef struct {
    uint64_t x, a, b;
    uint8_t op;
    int32_t quick_sc;
    uint32_t dmask;
    uint32_t umask;
} VBChild;

typedef struct {
    uint64_t x, a, b;
    uint8_t op;
    int32_t p_idx;
    uint32_t new_rem;
    double score;
} VBCandNext;

static int vb_cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a, vb = *(const uint64_t *)b;
    return (va < vb) ? -1 : (va > vb) ? 1 : 0;
}

static int vb_cmp_req(const void *a, const void *b) {
    const VBReq *ra = (const VBReq *)a, *rb = (const VBReq *)b;
    if (ra->need < rb->need) return -1;
    if (ra->need > rb->need) return 1;
    return (int)ra->ti - (int)rb->ti;
}

static int vb_cmp_next(const void *a, const void *b) {
    const VBCandNext *ca = (const VBCandNext *)a, *cb = (const VBCandNext *)b;
    if (ca->score < cb->score) return -1;
    if (ca->score > cb->score) return 1;
    return (ca->x < cb->x) ? -1 : (ca->x > cb->x) ? 1 : 0;
}

static inline uint64_t vb_isqrt64(uint64_t n) {
    if (n <= 1ULL) return n;
    int s = bl64(n);
    uint64_t x = 1ULL << ((s + 1) >> 1);
    for (;;) {
        uint64_t y = (x + n / x) >> 1;
        if (y >= x) return x;
        x = y;
    }
}

static inline int vb_lb_u64(const uint64_t *arr, int n, uint64_t key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline int vb_lb_req(const VBReq *arr, int n, uint64_t key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid].need < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline uint64_t vb_nearest_diff_u64(const uint64_t *sv, int nv, uint64_t key) {
    int pos = vb_lb_u64(sv, nv, key);
    uint64_t best = UINT64_MAX;
    if (pos < nv) {
        uint64_t d = sv[pos] - key;
        if (d < best) best = d;
    }
    if (pos > 0) {
        uint64_t d = key - sv[pos - 1];
        if (d < best) best = d;
    }
    return best;
}

static inline uint64_t vb_eval_gap_sv(uint64_t t, uint64_t x, const uint64_t *sv, int nv, uint64_t vl) {
    uint64_t best = (t > x) ? (t - x) : (x - t);
    uint64_t d;
    if (t > x) {
        d = vb_nearest_diff_u64(sv, nv, t - x);
        if (d < best) best = d;
    } else if (x > t) {
        d = vb_nearest_diff_u64(sv, nv, x - t);
        if (d < best) best = d;
    }
    if (x <= vl - t) {
        d = vb_nearest_diff_u64(sv, nv, t + x);
        if (d < best) best = d;
    }
    if (x > 0ULL) {
        uint64_t q = t / x;
        int pos = vb_lb_u64(sv, nv, q);
        for (int off = -1; off <= 1; off++) {
            int p = pos + off;
            if (p >= 0 && p < nv) {
                u128 prod = (u128)x * (u128)sv[p];
                if (prod <= (u128)vl) {
                    uint64_t pv = (uint64_t)prod;
                    d = (t > pv) ? (t - pv) : (pv - t);
                    if (d < best) best = d;
                }
            }
        }
    }
    if (t > 0ULL && x > t) {
        uint64_t q = x / t;
        int pos = vb_lb_u64(sv, nv, q);
        for (int off = -1; off <= 1; off++) {
            int p = pos + off;
            if (p >= 0 && p < nv && sv[p] > 1ULL) {
                uint64_t qv = x / sv[p];
                d = (t > qv) ? (t - qv) : (qv - t);
                if (d < best) best = d;
            }
        }
    }
    if (x + x <= vl) {
        uint64_t x2 = x + x;
        d = (t > x2) ? (t - x2) : (x2 - t);
        if (d < best) best = d;
    }
    if (x > 1ULL && (u128)x * (u128)x <= (u128)vl) {
        uint64_t xx = x * x;
        d = (t > xx) ? (t - xx) : (xx - t);
        if (d < best) best = d;
    }
    return best ? best : 1ULL;
}

static inline uint64_t vb_hash_arr(const uint64_t *arr, int n) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ (uint64_t)n;
    for (int i = 0; i < n; i++) {
        h ^= hash64(arr[i] + 0x517cc1b727220a95ULL * (uint64_t)(i + 1));
        h = (h << 13) | (h >> 51);
        h *= 0xbf58476d1ce4e5b9ULL;
    }
    return h ? h : 1ULL;
}

static inline double vb_dict_cost(const BigDict *d, const BigDict *u, uint64_t x) {
    if (x == 0ULL || x >= (1ULL << 24)) return 1e9;
    if (d && d->seen24 && (d->seen24[x >> 6] & (1ULL << (x & 63)))) {
        int32_t id = bd_find(d, x);
        if (id >= 0) return 1.5 + 1.0 * (double)meta_cost(d->meta[id]);
    }
    if (u && u->seen24 && (u->seen24[x >> 6] & (1ULL << (x & 63)))) {
        int32_t iu = bd_find(u, x);
        if (iu >= 0) return 2.5 + 1.0 * (double)meta_cost(u->meta[iu]);
    }
    return 1e9;
}

static inline double vb_dict_pair_ctg(const BigDict *d, const BigDict *u, uint64_t t, uint64_t x, uint64_t vl) {
    if (!d && !u) return 1e9;
    double best = 1e9, c;
    if (t > x) { c = vb_dict_cost(d, u, t - x); if (c < best) best = c; }
    else if (x > t) { c = vb_dict_cost(d, u, x - t); if (c < best) best = c; }
    if (x <= vl - t) { c = vb_dict_cost(d, u, t + x); if (c < best) best = c; }
    if (x > 1ULL && (t % x == 0ULL)) { c = vb_dict_cost(d, u, t / x); if (c < best) best = c; }
    if ((u128)t * (u128)x <= (u128)vl) { c = vb_dict_cost(d, u, t * x); if (c < best) best = c; }
    if (x > 16ULL && t > x) {
        uint64_t q = t / x, r = t % x;
        if (q < (1ULL << 24) && r < (1ULL << 24)) {
            double cq = vb_dict_cost(d, u, q);
            if (cq < 1e8) {
                double cr = (r == 0ULL) ? 0.0 : vb_dict_cost(d, u, r);
                if (cr < 1e8 && (1.0 + cq + cr < best)) best = 1.0 + cq + cr;
            }
        }
    }
    return best;
}

/*
 * C99 64-bit 2-step + Dictionary-Guided Value-Set Beam Search:
 * Discovers non-trivial shared intermediate registers across multiple targets.
 * Returns number of emitted instructions (>0 on success, 0 if unsolved).
 */
EXPORT int fe_value_beam64(
    const uint64_t *seeds, int n_seeds,
    const uint64_t *targets, int n_targets,
    uint64_t vl, int beam_w, int max_steps,
    void *p_local, void *p_univ,
    uint8_t *out_op, uint64_t *out_a, uint64_t *out_b, uint64_t *out_r
) {
    const BigDict *d = (const BigDict *)p_local;
    const BigDict *u = (const BigDict *)p_univ;
    if (n_seeds <= 0 || n_targets <= 0 || n_targets > 31) return 0;
    if (n_seeds >= VB_MAX_POOL - 1 || beam_w <= 0 || max_steps <= 0) return 0;
    if (n_seeds + max_steps > VB_MAX_POOL) max_steps = VB_MAX_POOL - n_seeds;
    if (beam_w > 512) beam_w = 512;

    int ret = 0;
    int max_hist = beam_w * (max_steps + 1) + 64;
    int max_cand_next = beam_w * VB_K_CHILD + 32;
    int req_cap = n_targets * (5 * VB_MAX_POOL + 8);

    VBState *cur = (VBState *)calloc((size_t)beam_w, sizeof(VBState));
    VBState *nxt = (VBState *)calloc((size_t)beam_w, sizeof(VBState));
    VBHist *hist = (VBHist *)malloc((size_t)max_hist * sizeof(VBHist));
    VBCandNext *cand_next = (VBCandNext *)malloc((size_t)max_cand_next * sizeof(VBCandNext));
    VBReq *reqs = (VBReq *)malloc((size_t)req_cap * sizeof(VBReq));
    uint64_t *seen_k = (uint64_t *)calloc(VB_SEEN_HT, sizeof(uint64_t));
    uint32_t *seen_g = (uint32_t *)calloc(VB_SEEN_HT, sizeof(uint32_t));
    int32_t *g_ht = (int32_t *)malloc(VB_SET_HT * sizeof(int32_t));
    uint64_t *nxt_sorted = (uint64_t *)malloc((size_t)beam_w * VB_MAX_POOL * sizeof(uint64_t));
    if (!cur || !nxt || !hist || !cand_next || !reqs || !seen_k || !seen_g || !g_ht || !nxt_sorted) {
        goto cleanup;
    }

    uint32_t init_rem = (1U << n_targets) - 1U;
    cur[0].nv = n_seeds;
    cur[0].hist_id = -1;
    for (int i = 0; i < n_seeds; i++) cur[0].vals[i] = seeds[i];
    for (int ti = 0; ti < n_targets; ti++) {
        for (int i = 0; i < n_seeds; i++) {
            if (seeds[i] == targets[ti]) init_rem &= ~(1U << ti);
        }
    }
    if (init_rem == 0U) { ret = 0; goto cleanup; }
    cur[0].rem_mask = init_rem;

    int n_cur = 1;
    int n_hist = 0;
    uint32_t gen = 1U;

    for (int step = 0; step < max_steps; step++) {
        int n_cnext = 0;
        for (int si = 0; si < n_cur; si++) {
            const VBState *st = &cur[si];
            int nv = st->nv;
            if (nv >= VB_MAX_POOL) continue;

            uint64_t sv[VB_MAX_POOL];
            memcpy(sv, st->vals, (size_t)nv * sizeof(uint64_t));
            qsort(sv, (size_t)nv, sizeof(uint64_t), vb_cmp_u64);

            uint64_t rem_t[32];
            uint8_t rem_ti[32];
            int nr = 0;
            for (int ti = 0; ti < n_targets; ti++) {
                if (st->rem_mask & (1U << ti)) {
                    rem_t[nr] = targets[ti];
                    rem_ti[nr] = (uint8_t)ti;
                    nr++;
                }
            }
            if (nr == 0) continue;

            gen++;
            for (int i = 0; i < nv; i++) {
                uint32_t h = (uint32_t)hash64(sv[i]) & (VB_SEEN_HT - 1U);
                while (seen_g[h] == gen) h = (h + 1U) & (VB_SEEN_HT - 1U);
                seen_g[h] = gen;
                seen_k[h] = sv[i];
            }

            int n_req = 0;
            uint32_t base_can1_mask = 0U;
            double base_d[32];
            for (int k = 0; k < nr; k++) {
                uint64_t t = rem_t[k];
                uint8_t ti = rem_ti[k];
                uint64_t mg = t;
                int c1 = 0;
                if (n_req < req_cap) reqs[n_req++] = (VBReq){t, ti};

                #define PUSH_REQ(VAL) do { \
                    uint64_t _nv = (VAL); \
                    if (_nv > 0ULL && _nv <= vl && _nv != t) { \
                        if (n_req < req_cap) reqs[n_req++] = (VBReq){_nv, ti}; \
                        uint64_t _df = vb_nearest_diff_u64(sv, nv, _nv); \
                        if (_df == 0ULL) c1 = 1; \
                        else if (_df < mg) mg = _df; \
                    } \
                } while (0)

                for (int i = 0; i < nv; i++) {
                    uint64_t u = sv[i];
                    if (t > u) PUSH_REQ(t - u);
                    if (u <= vl - t) PUSH_REQ(t + u);
                    if (u > t) PUSH_REQ(u - t);
                    if (u > 1ULL && (t % u == 0ULL)) PUSH_REQ(t / u);
                    if ((u128)t * (u128)u <= (u128)vl) PUSH_REQ(t * u);
                }
                uint64_t sq = vb_isqrt64(t);
                if (sq > 1ULL && sq * sq == t) PUSH_REQ(sq);
                if ((t & 1ULL) == 0ULL) PUSH_REQ(t >> 1);
                #undef PUSH_REQ

                if (c1) {
                    base_can1_mask |= (1U << ti);
                    base_d[ti] = 1.0;
                } else {
                    double b_gap = 2.0 + 0.25 * (double)bl64(mg);
                    if (d || u) {
                        double dt = vb_dict_cost(d, u, t);
                        if (dt < b_gap) b_gap = dt;
                        for (int i = 0; i < nv; i++) {
                            double dc = vb_dict_pair_ctg(d, u, t, sv[i], vl);
                            if (dc < b_gap) b_gap = dc;
                        }
                    }
                    base_d[ti] = b_gap;
                }
            }
            qsort(reqs, (size_t)n_req, sizeof(VBReq), vb_cmp_req);

            VBChild top_c[VB_K_CHILD];
            int nt_c = 0, wt_c = 0;

            #define TEST_CAND(XV, OPV, AV, BV) do { \
                uint64_t _x = (XV); \
                if (_x > 0ULL && _x <= vl) { \
                    uint32_t _h = (uint32_t)hash64(_x) & (VB_SEEN_HT - 1U); \
                    int _dup = 0; \
                    for (uint32_t _pr = 0; _pr < 64U; _pr++) { \
                        if (seen_g[_h] != gen) break; \
                        if (seen_k[_h] == _x) { _dup = 1; break; } \
                        _h = (_h + 1U) & (VB_SEEN_HT - 1U); \
                    } \
                    if (!_dup && seen_g[_h] != gen) { \
                        seen_g[_h] = gen; \
                        seen_k[_h] = _x; \
                        uint32_t _dm = 0U; \
                        for (int _k = 0; _k < nr; _k++) { \
                            if (_x == rem_t[_k]) _dm |= (1U << rem_ti[_k]); \
                        } \
                        if ((st->rem_mask & ~_dm) == 0U) { \
                            out_op[step] = (uint8_t)(OPV); \
                            out_a[step] = (AV); \
                            out_b[step] = (BV); \
                            out_r[step] = _x; \
                            int32_t _hid = st->hist_id; \
                            for (int _kk = step - 1; _kk >= 0 && _hid >= 0; _kk--) { \
                                out_op[_kk] = hist[_hid].op; \
                                out_a[_kk] = hist[_hid].a; \
                                out_b[_kk] = hist[_hid].b; \
                                out_r[_kk] = hist[_hid].r; \
                                _hid = hist[_hid].parent; \
                            } \
                            ret = step + 1; \
                            goto cleanup; \
                        } \
                        int _pos = vb_lb_req(reqs, n_req, _x); \
                        uint32_t _um = 0U; \
                        uint64_t _bg = UINT64_MAX; \
                        if (_pos < n_req) { \
                            uint64_t _d = reqs[_pos].need - _x; \
                            if (_d < _bg) _bg = _d; \
                            int _p = _pos; \
                            while (_p < n_req && reqs[_p].need == _x) { \
                                _um |= (1U << reqs[_p].ti); \
                                _p++; \
                            } \
                        } \
                        if (_pos > 0) { \
                            uint64_t _d = _x - reqs[_pos - 1].need; \
                            if (_d < _bg) _bg = _d; \
                        } \
                        uint32_t _new_u = _um & ~base_can1_mask; \
                        int32_t _qsc = (_dm ? 1000000 : 0) \
                                     + 50000 * __builtin_popcount(_new_u) \
                                     - 64 * bl64(_bg + 1ULL) \
                                     - bl64(_x); \
                        if (nt_c < VB_K_CHILD) { \
                            top_c[nt_c] = (VBChild){_x, (AV), (BV), (uint8_t)(OPV), _qsc, _dm, _um}; \
                            if (top_c[nt_c].quick_sc < top_c[wt_c].quick_sc) wt_c = nt_c; \
                            nt_c++; \
                        } else if (_qsc > top_c[wt_c].quick_sc) { \
                            top_c[wt_c] = (VBChild){_x, (AV), (BV), (uint8_t)(OPV), _qsc, _dm, _um}; \
                            wt_c = 0; \
                            for (int _hi = 1; _hi < nt_c; _hi++) { \
                                if (top_c[_hi].quick_sc < top_c[wt_c].quick_sc) wt_c = _hi; \
                            } \
                        } \
                    } \
                } \
            } while (0)

            for (int i = 0; i < nv; i++) {
                uint64_t a = sv[i];
                for (int j = i; j < nv; j++) {
                    uint64_t b = sv[j];
                    if (b <= vl - a) TEST_CAND(a + b, 1, a, b);
                    if (a != b) TEST_CAND(b - a, 2, b, a);
                    if (a > 1ULL && b > 1ULL && (u128)a * (u128)b <= (u128)vl) {
                        TEST_CAND(a * b, 3, a, b);
                    }
                    if (i == j && a > 1ULL) {
                        TEST_CAND(1ULL, 4, a, a);
                    } else if (a > 1ULL && (b % a == 0ULL)) {
                        TEST_CAND(b / a, 4, b, a);
                    }
                }
            }
            #undef TEST_CAND

            for (int ci = 0; ci < nt_c; ci++) {
                const VBChild *c = &top_c[ci];
                uint32_t nrem = st->rem_mask & ~c->dmask;
                double sc = 0.0;
                for (int k = 0; k < nr; k++) {
                    uint8_t ti = rem_ti[k];
                    if ((nrem & (1U << ti)) == 0U) continue;
                    uint64_t t = rem_t[k];
                    if ((base_can1_mask & (1U << ti)) || (c->umask & (1U << ti))
                        || (c->x + c->x == t) || ((u128)c->x * (u128)c->x == (u128)t)) {
                        sc += 1.0;
                    } else {
                        uint64_t mg = vb_eval_gap_sv(t, c->x, sv, nv, vl);
                        double dx = 2.0 + 0.25 * (double)bl64(mg);
                        if (d || u) {
                            double dc = vb_dict_pair_ctg(d, u, t, c->x, vl);
                            if (dc < dx) dx = dc;
                        }
                        sc += (base_d[ti] < dx) ? base_d[ti] : dx;
                    }
                }
                sc += 0.002 * (double)bl64(c->x);
                if (n_cnext < max_cand_next) {
                    cand_next[n_cnext++] = (VBCandNext){c->x, c->a, c->b, c->op, si, nrem, sc};
                }
            }
        }

        if (n_cnext == 0) break;
        qsort(cand_next, (size_t)n_cnext, sizeof(VBCandNext), vb_cmp_next);

        memset(g_ht, 0xFF, VB_SET_HT * sizeof(int32_t));
        int n_nxt = 0;
        for (int ci = 0; ci < n_cnext && n_nxt < beam_w; ci++) {
            const VBCandNext *cn = &cand_next[ci];
            const VBState *pst = &cur[cn->p_idx];
            int nv = pst->nv;
            uint64_t sorted_tmp[VB_MAX_POOL];
            memcpy(sorted_tmp, pst->vals, (size_t)nv * sizeof(uint64_t));
            sorted_tmp[nv] = cn->x;
            qsort(sorted_tmp, (size_t)(nv + 1), sizeof(uint64_t), vb_cmp_u64);

            uint64_t hv = vb_hash_arr(sorted_tmp, nv + 1);
            uint32_t h = (uint32_t)hv & (VB_SET_HT - 1U);
            int dup = 0;
            while (g_ht[h] >= 0) {
                int32_t idx = g_ht[h];
                if (memcmp(&nxt_sorted[(size_t)idx * VB_MAX_POOL], sorted_tmp, (size_t)(nv + 1) * sizeof(uint64_t)) == 0) {
                    dup = 1;
                    break;
                }
                h = (h + 1U) & (VB_SET_HT - 1U);
            }
            if (dup) continue;
            if (n_hist >= max_hist) break;

            memcpy(&nxt_sorted[(size_t)n_nxt * VB_MAX_POOL], sorted_tmp, (size_t)(nv + 1) * sizeof(uint64_t));
            g_ht[h] = (int32_t)n_nxt;

            memcpy(nxt[n_nxt].vals, pst->vals, (size_t)nv * sizeof(uint64_t));
            nxt[n_nxt].vals[nv] = cn->x;
            nxt[n_nxt].nv = nv + 1;
            nxt[n_nxt].rem_mask = cn->new_rem;

            int32_t hid = n_hist++;
            hist[hid] = (VBHist){pst->hist_id, cn->op, cn->a, cn->b, cn->x};
            nxt[n_nxt].hist_id = hid;
            n_nxt++;
        }

        if (n_nxt == 0) break;
        VBState *tmp_ptr = cur; cur = nxt; nxt = tmp_ptr;
        n_cur = n_nxt;
    }

cleanup:
    free(cur); free(nxt); free(hist); free(cand_next); free(reqs);
    free(seen_k); free(seen_g); free(g_ht); free(nxt_sorted);
    return ret;
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
        import zlib
        c_tag = f"v10_{zlib.crc32(_C_SOURCE.encode('utf-8')):08x}"
        ext = ".dll" if os.name == "nt" else ".so"
        tdir = tempfile.gettempdir()
        so_path = os.path.join(tdir, f"santa2026_fast_engine_{c_tag}{ext}")
        if not os.path.exists(so_path):
            pid = os.getpid()
            c_path = os.path.join(tdir, f"santa2026_fast_engine_{c_tag}_{pid}.c")
            tmp_so = os.path.join(tdir, f"santa2026_fast_engine_{c_tag}_{pid}{ext}")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(_C_SOURCE)
            flags = ["gcc", "-O3", "-shared", "-fPIC"] + (["-static-libgcc"] if os.name == "nt" else ["-pthread"])
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

        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            gcc_exe = shutil.which("gcc")
            if gcc_exe:
                try:
                    os.add_dll_directory(os.path.dirname(os.path.abspath(gcc_exe)))
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
        lib.fe_dict_copy_scan_prefix.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32
        ]
        lib.fe_dict_copy_scan_prefix.restype = ctypes.c_uint32

        lib.fe_two_pointer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_two_pointer.restype = ctypes.c_int

        lib.fe_two_pointer_s.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_two_pointer_s.restype = ctypes.c_int

        lib.fe_anchor_scan.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_anchor_scan.restype = ctypes.c_int

        lib.fe_mitm_probe.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_mitm_probe.restype = ctypes.c_int

        lib.fe_find_active_1ops.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.fe_find_active_1ops.restype = ctypes.c_int

        lib.fe_act_pairs_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint64,
        ]
        lib.fe_act_pairs_create.restype = ctypes.c_void_p

        lib.fe_act_pairs_free.argtypes = [ctypes.c_void_p]
        lib.fe_act_pairs_free.restype = None

        lib.fe_act_probe.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_act_probe.restype = ctypes.c_int

        lib.fe_screen_smooth.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]

        lib.fe_value_beam64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.fe_value_beam64.restype = ctypes.c_int

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

        self._caches: dict[bool, tuple[dict[int, Optional[tuple[str, int, int]]], dict[int, bool], dict[int, int]]] = {
            True: ({}, {}, {}),
            False: ({}, {}, {}),
        }
        self._node_cache, self._has_cache, self._cost_cache = self._caches[True]
        self._tp_cache: dict[tuple[int, bool, bool], list[tuple[int, int, int, int, int]]] = {}
        self._mitm_cache: dict[tuple[int, bool], list[tuple[int, int, int, int, int]]] = {}
        self._horner_cache: dict[tuple[int, bool, bool], list[list[tuple[str, int, int, int]]]] = {}
        self._act_buf = (ctypes.c_uint64 * 512)()
        self._oneop_op = (ctypes.c_uint8 * 64)()
        self._oneop_a = (ctypes.c_uint64 * 64)()
        self._oneop_b = (ctypes.c_uint64 * 64)()
        self._act_ptr = None
        self._act_key: Optional[tuple[tuple[int, ...], int]] = None
        self._act_probe_cache: dict[tuple[int, bool, tuple[tuple[int, ...], int]], list[tuple[str, int, int, int, int, int, int, int, int]]] = {}
        self._act_ap_buf = (ctypes.c_uint64 * 220)()
        self._ap_op = (ctypes.c_uint8 * 32)()
        self._ap_a = (ctypes.c_uint64 * 32)()
        self._ap_b = (ctypes.c_uint64 * 32)()
        self._ap_comb = (ctypes.c_uint64 * 32)()
        self._ap_d = (ctypes.c_uint64 * 32)()
        self._ap_prod = (ctypes.c_uint64 * 32)()
        self._ap_gap = (ctypes.c_uint64 * 32)()
        self._ap_sgn = (ctypes.c_int * 32)()
        self._ap_sc = (ctypes.c_int * 32)()
        self.ctx = None
        self.univ_ctx = None
        self.c_scan_size = 0
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
                self._node_cache, self._has_cache, self._cost_cache = self._caches[self.univ_ctx is not None]
                self.c_scan_size = int(self.clib.fe_scan_size(self.ctx))
                py_ns = min(self.c_scan_size, 200_000)
                c_arr = (ctypes.c_uint64 * py_ns)()
                got = int(self.clib.fe_dict_copy_scan_prefix(self.ctx, c_arr, ctypes.c_uint32(py_ns)))
                self.D_sorted = list(c_arr[:got])
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
        cand_hi = [v for v in self.D_sorted if 18 <= v.bit_length() <= 24]
        if cand_hi:
            step = max(1, len(cand_hi) // 24)
            sampled = cand_hi[::step][:24]
            sampled.sort(key=lambda v: (self.lookup_cost(v), -v.bit_length(), v))
            self.split_muls = sampled[:4]
        else:
            self.split_muls = [v for v in self.D_sorted if 16 <= v.bit_length() <= 24][:4]

    def __enter__(self) -> "DictEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._act_ptr is not None and self.clib is not None:
            self.clib.fe_act_pairs_free(self._act_ptr)
            self._act_ptr = None
            self._act_key = None
        self._act_probe_cache.clear()
        if self.ctx is not None and self.clib is not None:
            self.clib.fe_dict_free(self.ctx)
            self.ctx = None

    def set_allow_univ(self, allow: bool) -> None:
        """Toggle visibility of shared `{1}` Universal `U` with zero rebuild or cache-wipe cost."""
        new_u = _UNIV_CTX if (allow and self.ctx is not None) else None
        if new_u == self.univ_ctx:
            return
        self.univ_ctx = new_u
        self._node_cache, self._has_cache, self._cost_cache = self._caches[new_u is not None]
        self.powers.clear()
        pw = 1
        while pw <= self.vl:
            if self.has(pw):
                self.powers.add(pw)
            pw *= 2

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

    def find_active_1ops(
        self, tgt: int, active_vals: list[int], max_hits: int = 48
    ) -> list[tuple[str, int, int, int]]:
        """Batch-probe active_vals x (D_local U U) in one C99 call without Python loops."""
        if not active_vals or tgt <= 0 or tgt > self.vl:
            return []
        if self.py_backend is not None or self.ctx is None:
            out: list[tuple[str, int, int, int]] = []
            vl = self.vl
            for a in active_vals:
                if a < tgt:
                    b = tgt - a
                    if self.has(b):
                        out.append(("+", a, b, tgt))
                    if a > 1 and tgt % a == 0 and self.has(tgt // a):
                        out.append(("*", a, tgt // a, tgt))
                elif a > tgt:
                    b = a - tgt
                    if self.has(b):
                        out.append(("-", a, b, tgt))
                    if a % tgt == 0 and self.has(a // tgt):
                        out.append(("/", a, a // tgt, tgt))
                if a > 1 and tgt * a <= vl and self.has(tgt * a):
                    out.append(("/", tgt * a, a, tgt))
                if len(out) >= max_hits:
                    break
            return out[:max_hits]

        out = []
        n = len(active_vals)
        cap = min(max_hits, 64)
        for off in range(0, n, 512):
            chunk = active_vals[off : off + 512]
            na = len(chunk)
            self._act_buf[:na] = chunk
            nh = self.clib.fe_find_active_1ops(
                self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), ctypes.c_uint64(self.vl),
                self._act_buf, na, cap - len(out),
                self._oneop_op, self._oneop_a, self._oneop_b,
            )
            for i in range(nh):
                out.append((_OP_MAP[self._oneop_op[i]], self._oneop_a[i], self._oneop_b[i], tgt))
            if len(out) >= cap:
                break
        return out

    def two_pointer(
        self, tgt: int, K: int = 16, *, fast: bool = False
    ) -> list[tuple[int, int, int, int, int]]:
        t_key = (tgt, self.univ_ctx is not None, fast)
        cached = self._tp_cache.get(t_key)
        if cached is not None and (len(cached) >= K or K <= 16):
            return cached[:K]
        if self.py_backend is not None:
            K_q = max(K, 16)
            res = self.py_backend.two_pointer(tgt, K=K_q)
            self._tp_cache[t_key] = res
            return res[:K]
        K_q = max(K, 16)
        stride = max(1, self.c_scan_size // 500_000) if fast else 1
        od1 = (ctypes.c_uint64 * K_q)()
        od2 = (ctypes.c_uint64 * K_q)()
        oprod = (ctypes.c_uint64 * K_q)()
        ogap = (ctypes.c_uint64 * K_q)()
        osgn = (ctypes.c_int * K_q)()
        nh = self.clib.fe_two_pointer_s(
            self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), K_q, stride,
            od1, od2, oprod, ogap, osgn
        )
        res = [(od1[i], od2[i], oprod[i], ogap[i], osgn[i]) for i in range(nh)]
        self._tp_cache[t_key] = res
        return res[:K]

    def anchor_scan(
        self, tgt: int, active_vals: list[int], K: int = 16
    ) -> list[tuple[int, int, int, int, int, int]]:
        """
        C99 binary search across ALL active DAG intermediates `a` against full `D_local` scan,
        returning `(est_sc, gap, a, qq, prod, sgn)` sorted by lowest estimated cost.
        """
        if not active_vals or tgt <= 1:
            return []
        if self.py_backend is not None or self.ctx is None:
            import bisect
            D_s = self.D_sorted
            nD = len(D_s)
            hits: list[tuple[int, int, int, int, int, int]] = []
            for a in active_vals:
                if a <= 1 or a >= tgt or (a & (a - 1)) == 0:
                    continue
                qi = tgt // a
                for qq in (qi, qi + 1):
                    if qq >= 1 and a * qq <= self.vl and (qq <= 256 or self.has(qq)):
                        prod = a * qq
                        gap = abs(tgt - prod)
                        sc = (0 if qq == 1 else self.lookup_cost(qq) + 1) + (
                            0 if gap == 0 else (gap.bit_length() // 3 + 4)
                        )
                        hits.append((sc, gap, a, qq, prod, 1 if tgt >= prod else 0))
                pos = bisect.bisect_left(D_s, qi)
                for off in (-1, 0, 1):
                    if 0 <= pos + off < nD:
                        qq = D_s[pos + off]
                        prod = a * qq
                        if 0 < prod <= self.vl:
                            gap = abs(tgt - prod)
                            sc = 5 + (0 if gap == 0 else (gap.bit_length() // 3 + 4))
                            hits.append((sc, gap, a, qq, prod, 1 if tgt >= prod else 0))
            hits.sort(key=lambda x: (x[0], x[1]))
            return hits[:K]

        na = min(len(active_vals), 512)
        self._act_buf[:na] = active_vals[-na:]
        oa = (ctypes.c_uint64 * K)()
        oq = (ctypes.c_uint64 * K)()
        oprod = (ctypes.c_uint64 * K)()
        ogap = (ctypes.c_uint64 * K)()
        osgn = (ctypes.c_int * K)()
        osc = (ctypes.c_int * K)()
        nh = self.clib.fe_anchor_scan(
            self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), ctypes.c_uint64(self.vl),
            self._act_buf, na, K, oa, oq, oprod, ogap, osgn, osc
        )
        res = [(osc[i], ogap[i], oa[i], oq[i], oprod[i], osgn[i]) for i in range(nh)]
        res.sort(key=lambda x: (x[0], x[1]))
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
        u_key = (tgt, self.univ_ctx is not None)

        static_hits = self._mitm_cache.get(u_key)
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
            self._mitm_cache[u_key] = static_hits

        na = min(len(active_vals), 256)
        if na > 0:
            self._act_buf[:na] = active_vals[-na:]
            Ka = min(8, K)
            oa = (ctypes.c_uint64 * Ka)()
            oq = (ctypes.c_uint64 * Ka)()
            or_ = (ctypes.c_uint64 * Ka)()
            osgn = (ctypes.c_int * Ka)()
            osc = (ctypes.c_int * Ka)()
            nh_a = self.clib.fe_mitm_probe(
                self.ctx, self.univ_ctx, ctypes.c_uint64(tgt), ctypes.c_uint64(self.vl),
                self._act_buf, na, ctypes.c_uint32(0), Ka, oa, oq, or_, osgn, osc
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

    def act_probe(
        self, tgt: int, active_sorted: list[int], K: int = 12
    ) -> list[tuple[str, int, int, int, int, int, int, int, int]]:
        """
        C99 Dynamic Active-Pair Virtual Dictionary Probe:
        Builds sorted 1-op virtual hubs `E = a1 <op> a2` (+, -, *) over active nodes
        and probes `E +- gap`, `E * q +- gap` (q in D_local U U), and two-pointer
        `E * d_local +- gap`. Caches the compiled `ActPairSet` across identical active sets.
        """
        if len(active_sorted) < 2 or tgt <= 0 or tgt > self.vl:
            return []
        if len(active_sorted) <= 180:
            act_sub = active_sorted
        else:
            small_pw = [a for a in active_sorted if a <= 262144 or a in self.powers][:80]
            sp_set = set(small_pw)
            tail_hi = [a for a in active_sorted if a not in sp_set][-(180 - len(small_pw)):]
            act_sub = sorted(small_pw + tail_hi)
        if len(act_sub) < 2:
            return []

        key = (tuple(act_sub), self.vl)
        c_key = (tgt, self.univ_ctx is not None, key)
        cached = self._act_probe_cache.get(c_key)
        if cached is not None:
            return cached[:K]

        if self.clib is None or self.ctx is None:
            return []

        if self._act_key != key:
            if self._act_ptr is not None:
                self.clib.fe_act_pairs_free(self._act_ptr)
                self._act_ptr = None
            na = len(act_sub)
            self._act_ap_buf[:na] = act_sub
            self._act_ptr = self.clib.fe_act_pairs_create(
                self._act_ap_buf, na, ctypes.c_uint64(self.vl)
            )
            self._act_key = key

        if not self._act_ptr:
            return []

        act_full = active_sorted[-512:] if len(active_sorted) > 512 else active_sorted
        nf = len(act_full)
        self._act_buf[:nf] = act_full
        Kq = min(32, max(K, 12))
        nh = int(
            self.clib.fe_act_probe(
                self._act_ptr,
                self.ctx,
                self.univ_ctx,
                self._act_buf,
                nf,
                ctypes.c_uint64(tgt),
                ctypes.c_uint64(self.vl),
                Kq,
                self._ap_op,
                self._ap_a,
                self._ap_b,
                self._ap_comb,
                self._ap_d,
                self._ap_prod,
                self._ap_gap,
                self._ap_sgn,
                self._ap_sc,
            )
        )
        res = [
            (
                _OP_MAP[int(self._ap_op[i])],
                int(self._ap_a[i]),
                int(self._ap_b[i]),
                int(self._ap_comb[i]),
                int(self._ap_d[i]),
                int(self._ap_prod[i]),
                int(self._ap_gap[i]),
                int(self._ap_sgn[i]),
                int(self._ap_sc[i]),
            )
            for i in range(nh)
        ]
        if len(self._act_probe_cache) >= 2048:
            self._act_probe_cache.clear()
        self._act_probe_cache[c_key] = res
        return res[:K]


def solve_value_beam(
    puzzle: Puzzle,
    *,
    beam_w: int = 160,
    max_steps: int = 44,
    local_ctx: Optional[int] = None,
    use_univ: bool = True,
) -> Optional[list[Instruction]]:
    """
    Run C99 64-bit 2-step + Dictionary-Guided Value-Set Beam Search on small/medium puzzles
    and compile its acyclic DAG into `Instruction`s.
    """
    seeds = sorted(set(puzzle.seeds))
    targets = sorted(set(puzzle.targets) - set(seeds))
    if not targets:
        return []
    if (
        puzzle.value_limit >= (1 << 63)
        or not seeds
        or len(seeds) >= 118
        or len(targets) > 31
        or any(t <= 0 or t > puzzle.value_limit for t in targets)
    ):
        return None

    clib = get_clib()
    if clib is None:
        return None

    s_arr = (ctypes.c_uint64 * len(seeds))(*seeds)
    t_arr = (ctypes.c_uint64 * len(targets))(*targets)
    out_op = (ctypes.c_uint8 * max_steps)()
    out_a = (ctypes.c_uint64 * max_steps)()
    out_b = (ctypes.c_uint64 * max_steps)()
    out_r = (ctypes.c_uint64 * max_steps)()

    univ_p = _UNIV_CTX if use_univ else None
    k = int(
        clib.fe_value_beam64(
            s_arr, len(seeds),
            t_arr, len(targets),
            ctypes.c_uint64(puzzle.value_limit),
            int(beam_w), int(max_steps),
            local_ctx, univ_p,
            out_op, out_a, out_b, out_r,
        )
    )
    if k <= 0:
        return None

    defs: dict[int, tuple[str, int, int]] = {}
    for i in range(k):
        op_code = int(out_op[i])
        if op_code not in _OP_MAP:
            return None
        defs[int(out_r[i])] = (_OP_MAP[op_code], int(out_a[i]), int(out_b[i]))

    from src.dag_compiler import compile_dag
    from src.postpass import remove_dead_ops

    return remove_dead_ops(puzzle, compile_dag(puzzle, defs, n_trials=24))

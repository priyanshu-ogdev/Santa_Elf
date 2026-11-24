"""
src/solvers/fast_engine.py
128-bit overflow-safe dictionary frontier & two-pointer product engine.

Provides two identical-API backends via `DictEngine`:
  1. C99 accelerated backend (`gcc -O3` via `ctypes`, `unsigned __int128` arithmetic,
     ~220,000 stratified nodes across 1..60 bits, ~0.2ms two-pointer scans).
  2. Pure-Python fallback backend (~25,000 stratified nodes) when `gcc` is absent.

Key invariants:
  - Strictly acyclic DAG: existing nodes' parent pointers are NEVER overwritten once added.
  - Squaring backbone (`2, 4, 16, 256, 65536, 2^32`) + product powers + small constants
    `1..256` keep power-of-2 true topological rank and register footprint tiny.
  - Atomic compile-and-rename ensures multiprocessing workers never collide.
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

typedef struct {
    uint64_t val;
    uint64_t a;
    uint64_t b;
    uint8_t  op;    /* 0=seed, 1='+', 2='-', 3='*', 4='/' */
    uint8_t  cost;  /* saturated additive tree cost */
    uint8_t  rank;  /* strictly monotone topological depth */
} NodeRec;

#define HT_SIZE (1 << 20)
#define HT_MASK (HT_SIZE - 1)

typedef struct {
    NodeRec *nodes;
    int32_t  n_nodes;
    int32_t  max_nodes;
    uint64_t vl;
    uint64_t ht_keys[HT_SIZE];
    int32_t  ht_vals[HT_SIZE];
    int32_t  bucket_cnt[65];
} DictCtx;

static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

static inline int ht_find(const DictCtx *c, uint64_t k) {
    uint32_t h = (uint32_t)hash64(k) & HT_MASK;
    while (c->ht_keys[h]) {
        if (c->ht_keys[h] == k) return c->ht_vals[h];
        h = (h + 1) & HT_MASK;
    }
    return -1;
}

static inline void ht_ins(DictCtx *c, uint64_t k, int32_t v) {
    uint32_t h = (uint32_t)hash64(k) & HT_MASK;
    while (c->ht_keys[h]) {
        if (c->ht_keys[h] == k) return;
        h = (h + 1) & HT_MASK;
    }
    c->ht_keys[h] = k;
    c->ht_vals[h] = v;
}

static inline int bl64(uint64_t v) { return v ? (64 - __builtin_clzll(v)) : 0; }

/* Strict insert-once guarantees insertion-order acyclicity in D. */
static int try_add(
    DictCtx *c, uint64_t v, uint8_t op, uint64_t a, uint64_t b,
    uint8_t cost, uint8_t rank, int cap
) {
    if (v == 0 || v > c->vl || c->n_nodes >= c->max_nodes) return 0;
    if (ht_find(c, v) >= 0) return 0;
    int bl = bl64(v);
    if (cap > 0 && c->bucket_cnt[bl] >= cap) return 0;
    int idx = c->n_nodes++;
    c->nodes[idx].val  = v;
    c->nodes[idx].a    = a;
    c->nodes[idx].b    = b;
    c->nodes[idx].op   = op;
    c->nodes[idx].cost = cost;
    c->nodes[idx].rank = rank;
    c->bucket_cnt[bl]++;
    ht_ins(c, v, idx);
    return 1;
}

static int cmp_node(const void *pa, const void *pb) {
    uint64_t a = ((const NodeRec *)pa)->val, b = ((const NodeRec *)pb)->val;
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

static inline uint8_t sat_cost(uint16_t x) { return x < 250 ? (uint8_t)x : 250; }
static inline uint8_t sat_rank(uint16_t x) { return x < 250 ? (uint8_t)x : 250; }

EXPORT void* fe_dict_create(const uint64_t *seeds, int ns, uint64_t vl, int max_n, int per_b) {
    DictCtx *c = (DictCtx *)calloc(1, sizeof(DictCtx));
    if (!c) return NULL;
    c->vl = vl;
    c->max_nodes = max_n;
    c->nodes = (NodeRec *)malloc(sizeof(NodeRec) * (size_t)max_n);
    if (!c->nodes) { free(c); return NULL; }

    for (int i = 0; i < ns; i++) try_add(c, seeds[i], 0, 0, 0, 0, 0, 0);
    if (ht_find(c, 1) < 0 && ns > 0) try_add(c, 1, 4, seeds[0], seeds[0], 1, 1, 0);

    /* Squaring backbone: 2, 4, 16, 256, 65536, 2^32 */
    if (2 <= vl && ht_find(c, 2) < 0 && ht_find(c, 1) >= 0) {
        int i1 = ht_find(c, 1);
        try_add(c, 2, 1, 1, 1, sat_cost(c->nodes[i1].cost + 1), sat_rank(c->nodes[i1].rank + 1), 0);
    }
    uint64_t sq = 2;
    while (sq <= vl && sq <= vl / sq) {
        int idx = ht_find(c, sq);
        if (idx < 0) break;
        uint8_t r = c->nodes[idx].rank, co = c->nodes[idx].cost;
        sq = sq * sq;
        try_add(c, sq, 3, c->nodes[idx].val, c->nodes[idx].val, sat_cost(co + 1), sat_rank(r + 1), 0);
    }

    /* Small doubling powers 1..1024 */
    uint64_t cur = 1;
    while (cur <= 1024 && cur <= vl / 2) {
        int idx = ht_find(c, cur);
        if (idx < 0) break;
        uint8_t r = c->nodes[idx].rank, co = c->nodes[idx].cost;
        try_add(c, cur * 2, 1, cur, cur, sat_cost(co + 1), sat_rank(r + 1), 0);
        cur *= 2;
    }

    /* Fill remaining powers of 2 via products of existing powers */
    for (int pass = 0; pass < 3; pass++) {
        int lim = c->n_nodes;
        for (int i = 0; i < lim; i++) {
            uint64_t a = c->nodes[i].val;
            if ((a & (a - 1)) != 0) continue;
            for (int j = i; j < lim; j++) {
                uint64_t b = c->nodes[j].val;
                if ((b & (b - 1)) != 0) continue;
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)vl) {
                    uint8_t nc = sat_cost((uint16_t)c->nodes[i].cost + c->nodes[j].cost + 1);
                    uint8_t nr = sat_rank((c->nodes[i].rank > c->nodes[j].rank ? c->nodes[i].rank : c->nodes[j].rank) + 1);
                    try_add(c, (uint64_t)p, 3, a, b, nc, nr, 0);
                }
            }
        }
    }

    /* Small constants 1..256 via short BFS */
    for (int it = 0; it < 3; it++) {
        int lim = c->n_nodes;
        for (int i = 0; i < lim; i++) {
            uint64_t a = c->nodes[i].val; if (a > 256) continue;
            uint8_t ra = c->nodes[i].rank, ca = c->nodes[i].cost;
            for (int j = i; j < lim; j++) {
                uint64_t b = c->nodes[j].val; if (b > 256) continue;
                uint8_t nc = sat_cost((uint16_t)ca + c->nodes[j].cost + 1);
                uint8_t nr = sat_rank((ra > c->nodes[j].rank ? ra : c->nodes[j].rank) + 1);
                if (a + b <= 256) try_add(c, a + b, 1, a, b, nc, nr, 0);
                if (a * b <= 256) try_add(c, a * b, 3, a, b, nc, nr, 0);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    try_add(c, hi - lo, 2, hi, lo, nc, nr, 0);
                }
            }
        }
    }

    /* Layer 1: cross seeds, powers, and low-cost atoms (cost <= 9) */
    int base_end = c->n_nodes;
    for (int i = 0; i < base_end && c->n_nodes < max_n; i++) {
        uint64_t a = c->nodes[i].val; uint8_t ca = c->nodes[i].cost, ra = c->nodes[i].rank;
        if (ca > 9) continue;
        for (int j = i; j < base_end && c->n_nodes < max_n; j++) {
            uint64_t b = c->nodes[j].val; uint8_t cb = c->nodes[j].cost;
            if (cb > 9) continue;
            uint8_t nc = sat_cost((uint16_t)ca + cb + 1);
            uint8_t nr = sat_rank((ra > c->nodes[j].rank ? ra : c->nodes[j].rank) + 1);
            u128 p = (u128)a * (u128)b;
            if (p <= (u128)vl) try_add(c, (uint64_t)p, 3, a, b, nc, nr, per_b);
            u128 s = (u128)a + (u128)b;
            if (s <= (u128)vl) try_add(c, (uint64_t)s, 1, a, b, nc, nr, per_b);
            if (a != b) {
                uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                try_add(c, hi - lo, 2, hi, lo, nc, nr, per_b);
                if (hi % lo == 0) try_add(c, hi / lo, 4, hi, lo, nc, nr, per_b);
            }
        }
    }

    /* Collect balanced reps across bit-lengths 1..40: prioritize lowest cost */
    int *reps = (int *)malloc(sizeof(int) * 45 * 140);
    int nr = 0;
    if (reps) {
        for (int bl = 1; bl <= 40; bl++) {
            int cnt = 0;
            for (int tc = 0; tc <= 12 && cnt < 110; tc++) {
                for (int k = 0; k < c->n_nodes && cnt < 110; k++) {
                    if (bl64(c->nodes[k].val) == bl && c->nodes[k].cost == tc) {
                        reps[nr++] = k;
                        cnt++;
                    }
                }
            }
        }

        /* Layer 2: expand top-down across bit buckets so high buckets stay odd & dense */
        for (int ri = nr - 1; ri >= 0 && c->n_nodes < max_n; ri--) {
            int ia = reps[ri];
            uint64_t a = c->nodes[ia].val;
            uint8_t ca = c->nodes[ia].cost, ra = c->nodes[ia].rank;
            for (int rj = 0; rj <= ri && c->n_nodes < max_n; rj++) {
                int ib = reps[rj];
                uint64_t b = c->nodes[ib].val;
                uint16_t nc16 = (uint16_t)ca + c->nodes[ib].cost + 1;
                if (nc16 > 18) continue;
                uint8_t nc = (uint8_t)nc16;
                uint8_t rnk = sat_rank((ra > c->nodes[ib].rank ? ra : c->nodes[ib].rank) + 1);
                u128 p = (u128)a * (u128)b;
                if (p <= (u128)vl) try_add(c, (uint64_t)p, 3, a, b, nc, rnk, per_b);
                u128 s = (u128)a + (u128)b;
                if (s <= (u128)vl) try_add(c, (uint64_t)s, 1, a, b, nc, rnk, per_b);
                if (a != b) {
                    uint64_t hi = a > b ? a : b, lo = a > b ? b : a;
                    try_add(c, hi - lo, 2, hi, lo, nc, rnk, per_b);
                }
            }
        }
        free(reps);
    }

    qsort(c->nodes, c->n_nodes, sizeof(NodeRec), cmp_node);
    memset(c->ht_keys, 0, sizeof(c->ht_keys));
    for (int i = 0; i < c->n_nodes; i++) ht_ins(c, c->nodes[i].val, i);
    return c;
}

EXPORT void fe_dict_free(void *p) {
    DictCtx *c = (DictCtx *)p;
    if (c) { free(c->nodes); free(c); }
}
EXPORT int fe_dict_size(void *p) { return ((DictCtx *)p)->n_nodes; }
EXPORT int fe_dict_lookup(void *p, uint64_t v) { return ht_find((const DictCtx *)p, v); }

EXPORT void fe_dict_get(
    void *p, int idx, uint64_t *val, uint64_t *a, uint64_t *b, int *op, int *cost
) {
    const NodeRec *r = &((const DictCtx *)p)->nodes[idx];
    *val = r->val; *a = r->a; *b = r->b; *op = r->op; *cost = r->cost;
}

EXPORT int fe_dict_copy_vals(void *p, uint64_t *out) {
    const DictCtx *c = (const DictCtx *)p;
    for (int i = 0; i < c->n_nodes; i++) out[i] = c->nodes[i].val;
    return c->n_nodes;
}

typedef struct { uint64_t d1, d2, prod, gap; int32_t metric; uint8_t sgn; } Hit;

EXPORT int fe_two_pointer(
    void *p, uint64_t tgt, int K,
    uint64_t *od1, uint64_t *od2, uint64_t *oprod, uint64_t *ogap, int *osgn
) {
    const DictCtx *c = (const DictCtx *)p;
    int N = c->n_nodes;
    if (N == 0) return 0;
    if (K > 64) K = 64;

    Hit hits_gap[32]; int ng = 0, wg = 0;
    Hit hits_sc[32]; int ns = 0, ws = 0;
    int Kg = K / 2; if (Kg < 1) Kg = 1; if (Kg > 32) Kg = 32;
    int Ks = K - Kg; if (Ks > 32) Ks = 32;

    int i = 0, j = N - 1;
    u128 t128 = (u128)tgt;
    u128 vl128 = (u128)c->vl;

    while (i <= j) {
        uint64_t a = c->nodes[i].val, b = c->nodes[j].val;
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
            int m2 = (int)c->nodes[i].cost + (int)c->nodes[j].cost
                   + (g == 0 ? 0 : (ht_find(c, g) >= 0 ? 1 : (bl64(g) / 3 + 2)));
            if (Ks > 0) {
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
'''

_CLIB_CACHE = None
_CLIB_FAILED = False
_OP_MAP = {1: "+", 2: "-", 3: "*", 4: "/"}


def get_clib() -> Optional[ctypes.CDLL]:
    """
    Lazily compile and load the C99 acceleration shared library in tempdir.
    Safe under concurrent multiprocessing (atomic temp write + rename).
    Returns None if gcc is unavailable or compilation fails (triggers pure-Python fallback).
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
        so_path = os.path.join(tdir, f"santa2026_fast_engine_v4{ext}")
        if not os.path.exists(so_path):
            pid = os.getpid()
            c_path = os.path.join(tdir, f"santa2026_fast_engine_v4_{pid}.c")
            tmp_so = os.path.join(tdir, f"santa2026_fast_engine_v4_{pid}{ext}")
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
                if os.path.exists(so_path):
                    pass
                else:
                    so_path = tmp_so
            try:
                os.remove(c_path)
            except OSError:
                pass

        lib = ctypes.CDLL(so_path)
        lib.fe_dict_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
            ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
        ]
        lib.fe_dict_create.restype = ctypes.c_void_p
        lib.fe_dict_free.argtypes = [ctypes.c_void_p]
        lib.fe_dict_size.argtypes = [ctypes.c_void_p]
        lib.fe_dict_size.restype = ctypes.c_int
        lib.fe_dict_lookup.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.fe_dict_lookup.restype = ctypes.c_int
        lib.fe_dict_get.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_dict_copy_vals.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.fe_dict_copy_vals.restype = ctypes.c_int
        lib.fe_two_pointer.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.fe_two_pointer.restype = ctypes.c_int
        _CLIB_CACHE = lib
        return lib
    except Exception:
        _CLIB_FAILED = True
        return None


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

        # Squaring powers + doubling powers up to 1024 + power products
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

        # Small constants 1..128
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

        # Layer 1
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

        # Layer 2 reps
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


class DictEngine:
    """
    Unified interface to the C99 (or pure-Python fallback) dictionary & two-pointer engine.
    Always use as `with DictEngine(puzzle) as de:` or call `.close()`.
    """

    def __init__(self, puzzle: Puzzle, *, force_python: bool = False, max_nodes: int = 210000, per_bucket: int = 3600):
        self.puzzle = puzzle
        self.vl = puzzle.value_limit
        self.seeds = list(puzzle.seeds)
        self.seeds_set = set(self.seeds)
        self._node_cache: dict[int, Optional[tuple[str, int, int]]] = {}
        self._tp_cache: dict[int, list[tuple[int, int, int, int, int]]] = {}
        self.ctx = None
        self.py_backend: Optional[_PyDictBackend] = None

        self.clib = None if force_python else get_clib()
        if self.clib is not None and self.vl < (1 << 63) and self.seeds:
            arr = (ctypes.c_uint64 * len(self.seeds))(*self.seeds)
            self.ctx = self.clib.fe_dict_create(
                arr, len(self.seeds), ctypes.c_uint64(self.vl), max_nodes, per_bucket
            )
            if self.ctx:
                n = self.clib.fe_dict_size(self.ctx)
                c_arr = (ctypes.c_uint64 * n)()
                self.clib.fe_dict_copy_vals(self.ctx, c_arr)
                self.D_sorted = list(c_arr)
            else:
                self.clib = None

        if self.ctx is None:
            self.py_backend = _PyDictBackend(self.seeds, self.vl)
            self.D_sorted = self.py_backend.D_sorted

        self.D_set = set(self.D_sorted)
        self.powers: set[int] = set()
        pw = 1
        while pw <= self.vl:
            if pw in self.D_set:
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
        return v in self.D_set

    def get_dict_def(self, v: int) -> Optional[tuple[str, int, int]]:
        if v in self._node_cache:
            return self._node_cache[v]
        if self.py_backend is not None:
            res = self.py_backend.defs.get(v)
            self._node_cache[v] = res
            return res
        idx = self.clib.fe_dict_lookup(self.ctx, ctypes.c_uint64(v))
        if idx < 0:
            self._node_cache[v] = None
            return None
        vp = ctypes.c_uint64()
        ap = ctypes.c_uint64()
        bp = ctypes.c_uint64()
        op = ctypes.c_int()
        cp = ctypes.c_int()
        self.clib.fe_dict_get(
            self.ctx, idx, ctypes.byref(vp), ctypes.byref(ap),
            ctypes.byref(bp), ctypes.byref(op), ctypes.byref(cp)
        )
        if op.value == 0:
            self._node_cache[v] = None
            return None
        res = (_OP_MAP[op.value], ap.value, bp.value)
        self._node_cache[v] = res
        return res

    def two_pointer(self, tgt: int, K: int = 12) -> list[tuple[int, int, int, int, int]]:
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
            self.ctx, ctypes.c_uint64(tgt), K, od1, od2, oprod, ogap, osgn
        )
        res = [(od1[i], od2[i], oprod[i], ogap[i], osgn[i]) for i in range(nh)]
        self._tp_cache[tgt] = res
        return res

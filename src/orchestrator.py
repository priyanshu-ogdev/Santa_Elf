"""
src/orchestrator.py
Main driver: runs the solver portfolio per puzzle, picks the best program,
applies post-processing, and writes the final submission CSV.

Usage (CLI):
    python -m src.orchestrator \
        --puzzles  data/puzzles.csv \
        --out      submissions/sub_v1.csv \
        --cores    8 \
        --sa-time  2.0

Usage (programmatic):
    from src.orchestrator import run
    total_instructions = run("data/puzzles.csv", "submissions/sub_v1.csv")
"""
from __future__ import annotations
import csv
import os
import argparse
import multiprocessing
import time
from collections import Counter
from typing import Optional
from src.utils import (
    Puzzle,
    Instruction,
    apply_op,
    load_puzzles,
    program_from_json,
    write_submission,
)
from src.verifier import verify_program, quick_verify
from src.classifier import classify, PuzzleClass


# ---------------------------------------------------------------------------
# Path auto-detectors for Kaggle / Local environments
# ---------------------------------------------------------------------------

def find_puzzles_path(preferred: Optional[str] = None) -> str:
    """Find puzzles.csv across common Kaggle and local paths."""
    if preferred and os.path.exists(preferred):
        return os.path.abspath(preferred)

    candidates = [
        "data/puzzles.csv",
        "/kaggle/input/santa-2026/puzzles.csv",
        "/kaggle/input/santa-2026-the-elf-calculator/puzzles.csv",
        "puzzles.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # Walk /kaggle/input if present
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "puzzles.csv" in files:
                return os.path.abspath(os.path.join(root, "puzzles.csv"))

    # Walk current directory
    for root, dirs, files in os.walk("."):
        if "puzzles.csv" in files and ".git" not in root:
            return os.path.abspath(os.path.join(root, "puzzles.csv"))

    raise FileNotFoundError(
        "Could not find puzzles.csv! Please verify that the competition data is added under Input."
    )


def find_out_path(preferred: Optional[str] = None) -> str:
    """Determine output submission path."""
    if preferred:
        return preferred
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/submission.csv"
    if os.path.exists("submissions"):
        return "submissions/submission.csv"
    return "submission.csv"


# ---------------------------------------------------------------------------
# Per-puzzle solver dispatcher
# ---------------------------------------------------------------------------

def _attribute_program(
    puzzle: Puzzle, program: list[Instruction], dict_vals: Optional[set[int]] = None
) -> dict[str, int]:
    """Classify each instruction in a valid program into structural attribution buckets."""
    seeds = set(puzzle.seeds)
    targets = set(puzzle.targets) - seeds
    regs: list[Optional[tuple[str, int]]] = [None] * puzzle.registers
    for i, s in enumerate(puzzle.seeds):
        regs[i] = ("SEED", s)

    nodes: list[list] = []
    seen_vals: set[int] = set()
    hits_seen: set[int] = set(seeds)

    for inst in program:
        pa = regs[inst.src1]
        pb = regs[inst.src2]
        va = pa[1] if (pa is not None and pa[0] == "SEED") else (nodes[pa[1]][0] if pa is not None else 0)
        vb = pb[1] if (pb is not None and pb[0] == "SEED") else (nodes[pb[1]][0] if pb is not None else 0)
        val = apply_op(inst.op, va, vb, puzzle.value_limit) or 0
        is_tgt = (val in targets) and (val not in hits_seen)
        if is_tgt:
            hits_seen.add(val)
        is_one = (inst.op == "/" and va == vb and val == 1)
        nid = len(nodes)
        nodes.append([val, pa, pb, is_tgt, 0, is_one])
        seen_vals.add(val)
        regs[inst.dst] = ("STEP", nid)

    for node in nodes:
        for par in (node[1], node[2]):
            if par is not None and par[0] == "STEP":
                nodes[par[1]][4] += 1

    counts = {
        "make_one": 0,
        "target_leaf": 0,
        "shared_hub": 0,
        "dict_anc": 0,
        "glue": 0,
        "remat": max(0, len(nodes) - len(seen_vals)),
        "ops": len(nodes),
        "tgts": len(targets),
    }
    dvals = dict_vals or set()
    for val, _, _, is_tgt, out_deg, is_one in nodes:
        if is_one:
            counts["make_one"] += 1
        elif is_tgt and out_deg == 0:
            counts["target_leaf"] += 1
        elif out_deg >= 2:
            counts["shared_hub"] += 1
        elif val in dvals:
            counts["dict_anc"] += 1
        else:
            counts["glue"] += 1
    return counts


def _solve_puzzle_with_telemetry(
    puzzle: Puzzle,
    *,
    sa_time: float = 0.0,
    n_starts: int = 2,
    ram_gb: str | float = "auto",
    verbose: bool = False,
) -> tuple[list[Instruction], dict]:
    """Run solver portfolio and return `(best_program, telemetry_info)`."""
    from src.solvers import fallback, tiny_bfs, meet_in_middle, delta_cluster, factor_smooth
    from src.solvers import binary_shared, dag_synthesizer
    from src.postpass import postprocess

    label = classify(puzzle)
    candidates: list[tuple[str, list[Instruction]]] = []
    warnings: list[str] = []
    dag_diag: dict = {}

    # --- Fallback (always — guaranteed valid ceiling) ---
    fb_prog = fallback.solve(puzzle)
    r = verify_program(puzzle, fb_prog)
    if r.valid:
        candidates.append(("fallback", fb_prog))

    # --- Tiny exact solver ---
    if puzzle.value_limit < 200 or label == PuzzleClass.TINY:
        try:
            prog = tiny_bfs.solve(puzzle)
            if verify_program(puzzle, prog).valid:
                candidates.append(("tiny_bfs", prog))
        except Exception as e:
            warnings.append(f"tiny_bfs failed on {puzzle.id}: {e!r}")

    # --- dag_synthesizer (PRIMARY: High-RAM Shared U + 6-engine DAG + Pair-Rip + MILP) ---
    if puzzle.value_limit >= 40:
        try:
            prog = dag_synthesizer.solve(puzzle, ram_gb=ram_gb, _diag_out=dag_diag)
            r = verify_program(puzzle, prog)
            if r.valid:
                candidates.append(("dag_synthesizer", prog))
            else:
                warnings.append(f"dag_synthesizer invalid on {puzzle.id}: {r.error}")
        except Exception as e:
            warnings.append(f"dag_synthesizer failed on {puzzle.id}: {e!r}")

    # --- C99 2-step + Dictionary-Guided Value-Set Beam Search (small/medium <=100M, <=20t) ---
    if 40 <= puzzle.value_limit and puzzle.targets and max(puzzle.targets) <= 100_000_000 and len(puzzle.targets) <= 20:
        try:
            from src.solvers.fast_engine import solve_value_beam
            vb_lim = min(puzzle.value_limit, max(25_000_000, max(puzzle.targets) * 4))
            vb_puzzle = puzzle if vb_lim == puzzle.value_limit else Puzzle(puzzle.id, puzzle.seeds, puzzle.targets, vb_lim)
            vb_prog = solve_value_beam(vb_puzzle, beam_w=240, max_steps=72, use_univ=True)
            if vb_prog is not None and verify_program(puzzle, vb_prog).valid:
                candidates.append(("solve_value_beam", vb_prog))
        except Exception as e:
            warnings.append(f"solve_value_beam failed on {puzzle.id}: {e!r}")

    # --- binary_shared (secondary safety net only when primary solvers produced nothing) ---
    if len(candidates) <= 1:
        try:
            prog = binary_shared.solve(puzzle)
            r = verify_program(puzzle, prog)
            if r.valid:
                candidates.append(("binary_shared", prog))
        except Exception as e:
            warnings.append(f"binary_shared failed on {puzzle.id}: {e!r}")

    # --- Specialised small-range solvers (fast path for < 10^7 targets) ---
    if max(puzzle.targets) < 10**7 and label != PuzzleClass.TINY:
        try:
            s_name = "meet_in_middle"
            prog = None
            if label == PuzzleClass.SMOOTH:
                s_name = "factor_smooth"
                prog = factor_smooth.solve(puzzle)
            elif label == PuzzleClass.CLUSTERED:
                s_name = "delta_cluster"
                prog = delta_cluster.solve(puzzle)
            else:
                prog = meet_in_middle.solve(puzzle)
            if prog is not None and verify_program(puzzle, prog).valid:
                candidates.append((s_name, prog))
        except Exception:
            pass

        for seed in range(n_starts):
            try:
                prog = meet_in_middle.solve(puzzle, random_seed=seed)
                if verify_program(puzzle, prog).valid:
                    candidates.append(("meet_in_middle", prog))
            except Exception:
                pass

    if not candidates:
        raise RuntimeError(f"All solvers failed for puzzle {puzzle.id}")

    # Pick shortest so far
    winner, best = min(candidates, key=lambda pair: len(pair[1]))

    # --- Post-processing ---
    try:
        improved = postprocess(puzzle, best, run_annealing=(sa_time > 0),
                               annealing_budget=sa_time)
        ok, count, err = quick_verify(puzzle, improved)
        if ok:
            best = improved
    except Exception as e:
        warnings.append(f"postprocess failed on {puzzle.id}: {e!r}")

    if verbose:
        for w in warnings:
            print(f"  [warn] {w}", flush=True)
        print(f"  {puzzle.id} [{label.name}|{winner}]: {len(best)} instructions")

    attr = _attribute_program(puzzle, best, dag_diag.get("dict_vals"))
    return best, {"winner": winner, "warnings": warnings, "attr": attr}


def solve_puzzle(
    puzzle: Puzzle,
    *,
    sa_time: float = 0.0,
    n_starts: int = 2,
    ram_gb: str | float = "auto",
    verbose: bool = False,
) -> list[Instruction]:
    """
    Run the full solver portfolio for one puzzle. Returns the best valid program.
    """
    best, _ = _solve_puzzle_with_telemetry(
        puzzle, sa_time=sa_time, n_starts=n_starts, ram_gb=ram_gb, verbose=verbose
    )
    return best


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[str, list[list], int, dict]:
    """
    Multiprocessing worker. Returns (puzzle_id, program_as_list_of_lists, count, info).
    """
    puzzle, sa_time, n_starts, ram_gb, verbose = args
    try:
        prog, info = _solve_puzzle_with_telemetry(
            puzzle, sa_time=sa_time, n_starts=n_starts, ram_gb=ram_gb, verbose=verbose
        )
        return puzzle.id, [inst.to_list() for inst in prog], len(prog), info
    except Exception as e:
        from src.solvers import fallback
        prog = fallback.solve(puzzle)
        info = {
            "winner": "fallback",
            "warnings": [f"worker crash on {puzzle.id}: {e!r}"],
            "attr": _attribute_program(puzzle, prog),
        }
        return puzzle.id, [inst.to_list() for inst in prog], len(prog), info


def _load_verified_archive(
    archive_path: Optional[str], puzzle_map: dict[str, Puzzle]
) -> dict[str, list[Instruction]]:
    """Load verified programs from an existing archive CSV (skipping missing/invalid rows)."""
    archived: dict[str, list[Instruction]] = {}
    if not archive_path or not os.path.exists(archive_path):
        return archived
    try:
        with open(archive_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("id", "")
                if pid in puzzle_map:
                    prog = program_from_json(row["program"])
                    if verify_program(puzzle_map[pid], prog).valid:
                        if pid not in archived or len(prog) < len(archived[pid]):
                            archived[pid] = prog
    except Exception:
        pass
    return archived


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    puzzles_path: str,
    out_path: str,
    *,
    cores: int = 1,
    sa_time: float = 0.0,
    n_starts: int = 2,
    ram_gb: str | float = "auto",
    archive_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """
    Solve all puzzles and write submission CSV. Returns total instruction count.
    """
    # Prevent SciPy / OpenBLAS / MKL from spawning oversubscribed threads per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    puzzles = load_puzzles(puzzles_path)
    puzzle_map = {p.id: p for p in puzzles}
    archived = _load_verified_archive(archive_path, puzzle_map)
    total_start = time.monotonic()

    # Pre-warm C99 acceleration + Shared Universal `{1}`-Dict U BEFORE fork()
    backend_name = "Python fallback"
    try:
        from src.solvers.fast_engine import init_universal_dict, get_ram_telemetry, get_ram_profile
        prof0 = get_ram_profile(ram_gb)
        t0_mem = get_ram_telemetry()
        if verbose:
            print(
                f"Host RAM pre-init: Used {t0_mem['used_gb']:.2f} / {t0_mem['total_gb']:.2f} GB "
                f"(Avail {t0_mem['avail_gb']:.2f} GB) | Target profile={prof0['mode']} "
                f"(U_cap={int(prof0['univ_cap']):,}, D_local={int(prof0['local_cap']):,}/worker)",
                flush=True,
            )
        t_u = time.monotonic()
        info = init_universal_dict(ram_gb=ram_gb)
        dt_u = time.monotonic() - t_u
        if info.get("c99"):
            u_n = int(info.get("univ_size", 0))
            u_gb = float(info.get("univ_gb", 0.0))
            mode = info.get("mode", "auto")
            t1_mem = get_ram_telemetry()
            backend_name = f"C99 __int128 [{mode}, U={u_n:,} nodes, {u_gb:.2f} GB CoW]"
            if verbose:
                print(
                    f"Universal U ready in {dt_u:.1f}s: {u_n:,} nodes ({u_gb:.2f} GB shared CoW) | "
                    f"Host RAM Used {t1_mem['used_gb']:.2f} / {t1_mem['total_gb']:.2f} GB "
                    f"(Avail {t1_mem['avail_gb']:.2f} GB, Proc RSS {t1_mem['proc_rss_gb']:.2f} GB)",
                    flush=True,
                )
    except Exception as e:
        if verbose:
            print(f"Warning: C99 universal dict init failed ({e!r}); falling back to Python.", flush=True)

    # Longest-Processing-Time (LPT) dispatch eliminates multi-core tail stragglers
    work_puzzles = sorted(
        puzzles,
        key=lambda p: (
            -len(set(p.targets) - set(p.seeds)),
            -max((t.bit_length() for t in p.targets), default=0),
            -p.value_limit.bit_length(),
            p.id,
        ),
    )

    if verbose:
        arch_note = f" | archive={len(archived)} loaded" if archive_path else ""
        print(f"Loaded {len(puzzles)} puzzles. Solving with {cores} core(s) [{backend_name}]{arch_note}...", flush=True)

    # Build arguments for workers
    worker_args = [(p, sa_time, n_starts, ram_gb, verbose and cores == 1) for p in work_puzzles]

    results: dict[str, list[Instruction]] = {}
    running_total = 0
    win_counts: Counter[str] = Counter()
    attr_tot: Counter[str] = Counter()
    total_warns = 0
    arch_wins = 0

    def _record_result(pid: str, prog: list[Instruction], info_d: dict) -> None:
        nonlocal running_total, total_warns, arch_wins
        prev = archived.get(pid)
        if prev is not None and len(prev) < len(prog):
            prog = prev
            arch_wins += 1
            win_counts["archive_kept"] += 1
        else:
            win_counts[info_d.get("winner", "unknown")] += 1
        results[pid] = prog
        count = len(prog)
        running_total += count

        warns = info_d.get("warnings") or []
        total_warns += len(warns)
        if verbose and cores > 1:
            for w in warns:
                print(f"  [warn] {w}", flush=True)

        attr = info_d.get("attr") or {}
        for k, val in attr.items():
            attr_tot[k] += val

        if not verbose:
            return
        done = len(results)
        if done % 25 == 0 or done == len(puzzles):
            el = time.monotonic() - total_start
            try:
                from src.solvers.fast_engine import get_ram_telemetry
                tm = get_ram_telemetry()
                mem_s = f" | RAM Used {tm['used_gb']:.2f}/{tm['total_gb']:.2f} GB (Avail {tm['avail_gb']:.2f} GB)"
            except Exception:
                mem_s = ""
            print(
                f"  {pid}: {count} instructions  [{done}/{len(puzzles)} | sum={running_total:,} | {el:.1f}s{mem_s}]",
                flush=True,
            )
        else:
            print(f"  {pid}: {count} instructions", flush=True)

    if cores > 1:
        # Force Linux `fork` context for true 1x CoW sharing of Universal `U`; `chunksize=1` streams immediately
        try:
            mp_ctx = multiprocessing.get_context("fork") if os.name != "nt" else multiprocessing.get_context()
            with mp_ctx.Pool(processes=cores) as pool:
                for pid, prog_lists, _, info_d in pool.imap_unordered(_worker, worker_args, chunksize=1):
                    _record_result(pid, [Instruction(*row) for row in prog_lists], info_d)
        except Exception as e:
            if verbose:
                print(f"Worker pool encountered {e}; executing remaining on main process...", flush=True)
            for puzzle in work_puzzles:
                if puzzle.id not in results:
                    pid, prog_lists, _, info_d = _worker((puzzle, sa_time, n_starts, ram_gb, verbose))
                    _record_result(pid, [Instruction(*row) for row in prog_lists], info_d)
    else:
        for puzzle in work_puzzles:
            pid, prog_lists, _, info_d = _worker((puzzle, sa_time, n_starts, ram_gb, verbose))
            _record_result(pid, [Instruction(*row) for row in prog_lists], info_d)

    # Write submission (and update monotonic best-of archive if enabled)
    write_submission(results, out_path)
    if archive_path:
        merged = dict(archived)
        for pid, prog in results.items():
            if pid not in merged or len(prog) < len(merged[pid]):
                merged[pid] = prog
        os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
        write_submission(merged, archive_path)

    total = sum(len(v) for v in results.values())
    elapsed = time.monotonic() - total_start

    if verbose:
        print(f"\nTotal instructions: {total:,}")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Solver wins: {dict(win_counts)} | Warnings: {total_warns}")
        if arch_wins:
            print(f"Best-of archive preserved shorter valid programs for {arch_wins} puzzles.")
        ops_sum = max(1, attr_tot.get("ops", 0))
        tgts_sum = max(1, attr_tot.get("tgts", 0))
        print(
            "Attribution summary "
            f"({tgts_sum:,} targets, {ops_sum:,} ops, {ops_sum / tgts_sum:.2f} ops/tgt): "
            f"dict_anc={attr_tot['dict_anc']:,} ({attr_tot['dict_anc']/ops_sum*100:.1f}%), "
            f"shared_hub={attr_tot['shared_hub']:,} ({attr_tot['shared_hub']/ops_sum*100:.1f}%), "
            f"glue={attr_tot['glue']:,} ({attr_tot['glue']/ops_sum*100:.1f}%), "
            f"target_leaf={attr_tot['target_leaf']:,} ({attr_tot['target_leaf']/ops_sum*100:.1f}%), "
            f"make_one={attr_tot['make_one']:,} ({attr_tot['make_one']/ops_sum*100:.1f}%), "
            f"remat={attr_tot['remat']:,} ({attr_tot['remat']/ops_sum*100:.1f}%)"
        )
        print(f"Submission written to: {out_path}")

    return total


# ---------------------------------------------------------------------------
# CLI / Notebook entry point
# ---------------------------------------------------------------------------

def _cli():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    is_notebook = (
        "ipykernel" in sys.modules
        or any("ipykernel" in a or "jupyter" in a or a.endswith(".json") for a in sys.argv)
    )

    # Determine default worker count:
    # On Linux (Kaggle), fork allows multiprocessing using all available CPU cores.
    # On Windows, spawn requires a physical file so notebook cells default to 1 core.
    if is_notebook and os.name != "nt":
        default_cores = multiprocessing.cpu_count() or 4
    elif not is_notebook:
        default_cores = multiprocessing.cpu_count() or 4
    else:
        default_cores = 1

    parser = argparse.ArgumentParser(description="Santa 2026 Elf Calculator Solver")
    parser.add_argument("--puzzles",  default=None, help="Path to puzzles.csv (auto-detected if omitted)")
    parser.add_argument("--out",      default=None, help="Output CSV path (auto-detected if omitted)")
    parser.add_argument("--cores",    type=int, default=default_cores, help="Number of parallel workers")
    parser.add_argument("--sa-time",  type=float, default=0.0, help="Optional extra SA time budget per puzzle (s)")
    parser.add_argument("--n-starts", type=int, default=2, help="M-i-M random restarts per small puzzle")
    parser.add_argument("--ram-gb",   default="auto", help="RAM budget profile in GB ('auto' uses full Kaggle 20-24 GB)")
    parser.add_argument("--archive",  default=None, help="Optional monotonic best-of archive CSV path")
    parser.add_argument("--verify",   action="store_true", default=True, help="Run full re-verification after solving")

    if is_notebook:
        # In a Jupyter / Kaggle cell, sys.argv contains kernel flags like -f kernel.json
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    puzzles_path = find_puzzles_path(args.puzzles)
    out_path = find_out_path(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print("=" * 60)
    print("Santa 2026: The Elf Calculator Solver")
    print(f"  Puzzles source : {puzzles_path}")
    print(f"  Submission out : {out_path}")
    if args.archive:
        print(f"  Archive CSV    : {args.archive}")
    print(f"  Workers        : {args.cores}")
    print(f"  RAM profile    : {args.ram_gb}")
    print(f"  SA time/puzzle : {args.sa_time}s")
    print("=" * 60)

    total = run(
        puzzles_path, out_path,
        cores=args.cores,
        sa_time=args.sa_time,
        n_starts=args.n_starts,
        ram_gb=args.ram_gb,
        archive_path=args.archive,
        verbose=True,
    )

    if args.verify:
        from src.verifier import replay_submission
        print("\nRunning full re-verification against competition rules...")
        verified = replay_submission(puzzles_path, out_path)
        assert verified == total, f"Mismatch: {verified} vs {total}"
        print(f"Verification PASSED with 0 errors!")
        print(f"Final Score: {verified:,} total instructions.")
        print(f"Ready for submission: {out_path}")


if __name__ == "__main__":
    _cli()

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
import os
import argparse
import multiprocessing
import time
from typing import Optional
from src.utils import Puzzle, Instruction, load_puzzles, write_submission
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

    Portfolio (in priority order):
      1. tiny_bfs        — exact bounded best-first search for value_limit < 100
      2. dag_synthesizer — 5-engine shared DAG + BPE + C99/Py two-pointer + MILP + Sethi-Ullman compiler (PRIMARY)
      3. binary_shared   — fast wNAF ensemble (secondary safety baseline)
      4. fallback        — simple binary construction (guaranteed valid ceiling)
      5. Specialised M-i-M / cluster / smooth solvers for small/medium targets
      6. Post-processing: dead-op removal + live-pool peephole + optional SA
    """
    from src.solvers import fallback, tiny_bfs, meet_in_middle, delta_cluster, factor_smooth
    from src.solvers import binary_shared, dag_synthesizer
    from src.postpass import postprocess

    label = classify(puzzle)
    candidates: list[list[Instruction]] = []

    # --- Fallback (always — guaranteed valid ceiling) ---
    fb_prog = fallback.solve(puzzle)
    r = verify_program(puzzle, fb_prog)
    if r.valid:
        candidates.append(fb_prog)

    # --- Tiny exact solver ---
    if puzzle.value_limit < 200 or label == PuzzleClass.TINY:
        try:
            prog = tiny_bfs.solve(puzzle)
            if verify_program(puzzle, prog).valid:
                candidates.append(prog)
        except Exception:
            pass

    # --- dag_synthesizer (PRIMARY: High-RAM Shared U + 6-engine DAG + Pair-Rip + MILP) ---
    if puzzle.value_limit >= 40:
        try:
            prog = dag_synthesizer.solve(puzzle, ram_gb=ram_gb)
            r = verify_program(puzzle, prog)
            if r.valid:
                candidates.append(prog)
        except Exception:
            pass

    # --- binary_shared (secondary wNAF ensemble — fast baseline) ---
    try:
        prog = binary_shared.solve(puzzle)
        r = verify_program(puzzle, prog)
        if r.valid:
            candidates.append(prog)
    except Exception:
        pass

    # --- Specialised small-range solvers (fast path for < 10^7 targets) ---
    if max(puzzle.targets) < 10**7 and label != PuzzleClass.TINY:
        try:
            prog = None
            if label == PuzzleClass.SMOOTH:
                prog = factor_smooth.solve(puzzle)
            elif label == PuzzleClass.CLUSTERED:
                prog = delta_cluster.solve(puzzle)
            else:
                prog = meet_in_middle.solve(puzzle)
            if prog is not None and verify_program(puzzle, prog).valid:
                candidates.append(prog)
        except Exception:
            pass

        for seed in range(n_starts):
            try:
                prog = meet_in_middle.solve(puzzle, random_seed=seed)
                if verify_program(puzzle, prog).valid:
                    candidates.append(prog)
            except Exception:
                pass

    if not candidates:
        raise RuntimeError(f"All solvers failed for puzzle {puzzle.id}")

    # Pick shortest so far
    best = min(candidates, key=len)

    # --- Post-processing ---
    try:
        improved = postprocess(puzzle, best, run_annealing=(sa_time > 0),
                               annealing_budget=sa_time)
        ok, count, err = quick_verify(puzzle, improved)
        if ok:
            best = improved
    except Exception:
        pass

    if verbose:
        print(f"  {puzzle.id} [{label.name}]: {len(best)} instructions")

    return best


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[str, list[list], int]:
    """
    Multiprocessing worker. Returns (puzzle_id, program_as_list_of_lists, count).
    """
    puzzle, sa_time, n_starts, ram_gb, verbose = args
    try:
        prog = solve_puzzle(puzzle, sa_time=sa_time, n_starts=n_starts, ram_gb=ram_gb, verbose=verbose)
        return puzzle.id, [inst.to_list() for inst in prog], len(prog)
    except Exception as e:
        # Emergency: use fallback
        from src.solvers import fallback
        prog = fallback.solve(puzzle)
        return puzzle.id, [inst.to_list() for inst in prog], len(prog)


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
    except Exception:
        pass

    if verbose:
        print(f"Loaded {len(puzzles)} puzzles. Solving with {cores} core(s) [{backend_name}]...", flush=True)

    # Build arguments for workers
    worker_args = [(p, sa_time, n_starts, ram_gb, verbose and cores == 1) for p in puzzles]

    results: dict[str, list[Instruction]] = {}
    running_total = 0

    def _log_progress(pid: str, count: int) -> None:
        nonlocal running_total
        running_total += count
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
                for pid, prog_lists, count in pool.imap_unordered(_worker, worker_args, chunksize=1):
                    from src.utils import Instruction
                    results[pid] = [Instruction(*row) for row in prog_lists]
                    _log_progress(pid, count)
        except Exception as e:
            if verbose:
                print(f"Worker pool encountered {e}; executing remaining on main process...", flush=True)
            for puzzle in puzzles:
                if puzzle.id not in results:
                    pid, prog_lists, count = _worker((puzzle, sa_time, n_starts, ram_gb, verbose))
                    from src.utils import Instruction
                    results[pid] = [Instruction(*row) for row in prog_lists]
                    _log_progress(pid, count)
    else:
        for puzzle in puzzles:
            pid, prog_lists, count = _worker((puzzle, sa_time, n_starts, ram_gb, verbose))
            from src.utils import Instruction
            results[pid] = [Instruction(*row) for row in prog_lists]
            _log_progress(pid, count)

    # Write submission
    write_submission(results, out_path)
    total = sum(len(v) for v in results.values())
    elapsed = time.monotonic() - total_start

    if verbose:
        print(f"\nTotal instructions: {total:,}")
        print(f"Elapsed: {elapsed:.1f}s")
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

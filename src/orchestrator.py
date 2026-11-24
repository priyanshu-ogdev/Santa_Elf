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

    # --- dag_synthesizer (PRIMARY: 5-engine shared DAG + Rip-Reroute + MILP) ---
    if puzzle.value_limit >= 40:
        try:
            prog = dag_synthesizer.solve(puzzle)
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
    puzzle, sa_time, n_starts, verbose = args
    try:
        prog = solve_puzzle(puzzle, sa_time=sa_time, n_starts=n_starts, verbose=verbose)
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
    verbose: bool = True,
) -> int:
    """
    Solve all puzzles and write submission CSV. Returns total instruction count.
    """
    puzzles = load_puzzles(puzzles_path)
    total_start = time.monotonic()

    # Pre-warm C99 acceleration library in main process before forking workers
    backend_name = "Python fallback"
    try:
        from src.solvers.fast_engine import get_clib
        if get_clib() is not None:
            backend_name = "C99 __int128 + ctypes"
    except Exception:
        pass

    if verbose:
        print(f"Loaded {len(puzzles)} puzzles. Solving with {cores} core(s) [{backend_name}]...")

    # Build arguments for workers
    worker_args = [(p, sa_time, n_starts, verbose and cores == 1) for p in puzzles]

    results: dict[str, list[Instruction]] = {}

    if cores > 1:
        chunksize = max(1, len(puzzles) // (cores * 4))
        try:
            with multiprocessing.Pool(processes=cores) as pool:
                for pid, prog_lists, count in pool.imap_unordered(_worker, worker_args, chunksize=chunksize):
                    from src.utils import Instruction
                    results[pid] = [Instruction(*row) for row in prog_lists]
                    if verbose:
                        print(f"  {pid}: {count} instructions")
        except Exception as e:
            if verbose:
                print(f"Worker pool encountered {e}; executing remaining on main process...")
            for puzzle in puzzles:
                if puzzle.id not in results:
                    pid, prog_lists, count = _worker((puzzle, sa_time, n_starts, verbose))
                    from src.utils import Instruction
                    results[pid] = [Instruction(*row) for row in prog_lists]
                    if verbose:
                        print(f"  {pid}: {count} instructions")
    else:
        for puzzle in puzzles:
            pid, prog_lists, count = _worker((puzzle, sa_time, n_starts, verbose))
            from src.utils import Instruction
            results[pid] = [Instruction(*row) for row in prog_lists]

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
    print(f"  SA time/puzzle : {args.sa_time}s")
    print("=" * 60)

    total = run(
        puzzles_path, out_path,
        cores=args.cores,
        sa_time=args.sa_time,
        n_starts=args.n_starts,
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

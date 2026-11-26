"""
tests/test_solvers.py
Integration tests for all solver modules.
Run: python -m pytest tests/test_solvers.py -v
"""
import pytest
from src.utils import Puzzle, Instruction, load_puzzles
from src.verifier import verify_program


def make_puzzle(seeds, targets, value_limit=10**18, pid="test"):
    return Puzzle(id=pid, seeds=seeds, targets=targets, value_limit=value_limit)


def assert_valid(puzzle, program, label=""):
    r = verify_program(puzzle, program)
    assert r.valid, f"{label} invalid on {puzzle.id}: {r.error}"
    return r.instruction_count


# ---------------------------------------------------------------------------
# Fallback solver
# ---------------------------------------------------------------------------

class TestFallback:
    def test_simple(self):
        from src.solvers.fallback import solve
        p = make_puzzle([1, 2], [3, 4, 5, 6, 7, 8])
        assert_valid(p, solve(p), "fallback")

    def test_all_seeds_are_targets(self):
        from src.solvers.fallback import solve
        p = make_puzzle([3, 5], [3, 5])
        r = verify_program(p, solve(p))
        assert r.valid

    def test_target_1(self):
        from src.solvers.fallback import solve
        p = make_puzzle([2], [1, 4, 8])
        assert_valid(p, solve(p), "fallback target=1")

    def test_first_10_scored(self):
        from src.solvers.fallback import solve
        for p in load_puzzles("data/puzzles.csv")[:10]:
            assert_valid(p, solve(p), f"fallback {p.id}")

    def test_all_tiny_puzzles(self):
        from src.solvers.fallback import solve
        for p in [p for p in load_puzzles("data/puzzles.csv") if p.value_limit < 100]:
            assert_valid(p, solve(p), f"fallback tiny {p.id}")


# ---------------------------------------------------------------------------
# Binary Shared solver
# ---------------------------------------------------------------------------

class TestBinaryShared:
    def test_first_50_scored(self):
        from src.solvers.binary_shared import solve
        for p in load_puzzles("data/puzzles.csv")[:50]:
            assert_valid(p, solve(p), f"binary_shared {p.id}")

    def test_beats_fallback(self):
        from src.solvers.binary_shared import solve
        from src.solvers.fallback import solve as fb_solve
        puzzles = load_puzzles("data/puzzles.csv")[:50]
        assert sum(len(solve(p)) for p in puzzles) < sum(len(fb_solve(p)) for p in puzzles)

    def test_all_tiny(self):
        from src.solvers.binary_shared import solve
        for p in [p for p in load_puzzles("data/puzzles.csv") if p.value_limit < 100]:
            assert_valid(p, solve(p), f"binary_shared tiny {p.id}")


# ---------------------------------------------------------------------------
# Tiny BFS solver
# ---------------------------------------------------------------------------

class TestTinyBFS:
    def test_all_tiny_scored(self):
        from src.solvers.tiny_bfs import solve
        from src.solvers.fallback import solve as fallback_solve
        tiny = [p for p in load_puzzles("data/puzzles.csv") if p.value_limit < 100]
        for p in tiny:
            prog = solve(p)
            count = assert_valid(p, prog, f"tiny_bfs {p.id}")
            assert count <= len(fallback_solve(p))

    def test_zero_instruction_case(self):
        from src.solvers.tiny_bfs import solve
        p = make_puzzle([1, 2, 3], [1, 2], value_limit=10)
        assert_valid(p, solve(p))


# ---------------------------------------------------------------------------
# Meet-in-the-middle solver
# ---------------------------------------------------------------------------

class TestMeetInMiddle:
    def test_first_50_scored(self):
        from src.solvers.meet_in_middle import solve
        for p in load_puzzles("data/puzzles.csv")[:50]:
            assert_valid(p, solve(p), f"mim {p.id}")

    def test_tight_limit_puzzles(self):
        from src.solvers.meet_in_middle import solve
        puzzles = {p.id: p for p in load_puzzles("data/puzzles.csv")}
        for pid in ("scored_0883", "scored_0931"):
            if pid in puzzles:
                assert_valid(puzzles[pid], solve(puzzles[pid]), f"mim tight {pid}")

    def test_random_seed_produces_valid(self):
        from src.solvers.meet_in_middle import solve
        p = load_puzzles("data/puzzles.csv")[0]
        for seed in range(3):
            assert_valid(p, solve(p, random_seed=seed), f"mim random_seed={seed}")


# ---------------------------------------------------------------------------
# Delta cluster solver
# ---------------------------------------------------------------------------

class TestDeltaCluster:
    def test_first_20_scored(self):
        from src.solvers.delta_cluster import solve
        for p in load_puzzles("data/puzzles.csv")[:20]:
            assert_valid(p, solve(p), f"delta_cluster {p.id}")

    def test_practice_clustered(self):
        from src.utils import load_practice_puzzles
        from src.solvers.delta_cluster import solve
        try:
            puzzles = load_practice_puzzles("data/practice_puzzles.csv")
        except Exception:
            pytest.skip("practice_puzzles.csv not available")
        for p in [p for p in puzzles if p.family == "shared"][:5]:
            assert_valid(p, solve(p), f"delta_cluster practice {p.id}")


# ---------------------------------------------------------------------------
# Factor smooth solver
# ---------------------------------------------------------------------------

class TestFactorSmooth:
    def test_first_20_scored(self):
        from src.solvers.factor_smooth import solve
        for p in load_puzzles("data/puzzles.csv")[:20]:
            assert_valid(p, solve(p), f"factor_smooth {p.id}")

    def test_purely_smooth_targets(self):
        from src.solvers.factor_smooth import solve
        p = make_puzzle([2, 3, 5, 7, 11], [40, 77, 288], value_limit=10**6)
        assert_valid(p, solve(p), "factor_smooth smooth targets")


# ---------------------------------------------------------------------------
# DAG Synthesizer + Fast Engine + Register Compiler
# ---------------------------------------------------------------------------

class TestDagSynthesizer:
    def test_first_10_scored(self):
        from src.solvers.dag_synthesizer import solve
        from src.solvers.binary_shared import solve as bs_solve
        puzzles = load_puzzles("data/puzzles.csv")[:10]
        tot_dag = 0
        tot_bs = 0
        for p in puzzles:
            prog = solve(p)
            tot_dag += assert_valid(p, prog, f"dag_synth {p.id}")
            tot_bs += len(bs_solve(p))
        assert tot_dag < tot_bs * 0.5, f"Expected >50% improvement over binary_shared: {tot_dag} vs {tot_bs}"

    def test_python_fallback_backend(self):
        from src.solvers.dag_synthesizer import solve
        puzzles = load_puzzles("data/puzzles.csv")[:3]
        for p in puzzles:
            prog = solve(p, force_python=True)
            assert_valid(p, prog, f"dag_synth py_fallback {p.id}")

    def test_tight_limit_and_tiny(self):
        from src.solvers.dag_synthesizer import solve
        puzzles = {p.id: p for p in load_puzzles("data/puzzles.csv")}
        for pid in ("scored_0883", "scored_0931", "scored_0061", "scored_0348"):
            assert_valid(puzzles[pid], solve(puzzles[pid]), f"dag_synth tight {pid}")

    def test_practice_all_families(self):
        from src.utils import load_practice_puzzles
        from src.solvers.dag_synthesizer import solve
        try:
            prac = load_practice_puzzles("data/practice_puzzles.csv")
        except Exception:
            pytest.skip("practice_puzzles.csv not available")
        seen_fam = set()
        for p in prac:
            if p.family not in seen_fam:
                seen_fam.add(p.family)
                assert_valid(p, solve(p), f"dag_synth {p.family} {p.id}")
            if len(seen_fam) == 5:
                break


# ---------------------------------------------------------------------------
# Orchestrator (integration)
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_solve_puzzle_first_5(self):
        from src.orchestrator import solve_puzzle
        for p in load_puzzles("data/puzzles.csv")[:5]:
            assert_valid(p, solve_puzzle(p, sa_time=0.0, n_starts=1), f"orchestrator {p.id}")

    def test_solve_tiny_puzzle(self):
        from src.orchestrator import solve_puzzle
        tiny = [p for p in load_puzzles("data/puzzles.csv") if p.value_limit < 100][0]
        assert_valid(tiny, solve_puzzle(tiny, sa_time=0.0, n_starts=1), f"orchestrator tiny {tiny.id}")

    def test_solve_empty_targets_edge(self):
        from src.orchestrator import solve_puzzle
        p = make_puzzle([3, 7], [], value_limit=1000, pid="empty_targets")
        prog = solve_puzzle(p, sa_time=0.0, n_starts=1)
        assert_valid(p, prog, "orchestrator empty_targets")
        assert len(prog) == 0

    def test_c99_act_probe_and_spilling_compiler(self):
        import src.solvers.fast_engine as fe
        from src.dag_compiler import compile_dag
        p = make_puzzle([2, 3, 5, 7, 11], [100], value_limit=(1 << 50), pid="act_probe_test")
        with fe.DictEngine(p, max_nodes=200_000, ram_gb="4.0") as de:
            if de.clib is not None:
                act = [2, 3, 5, 7, 11, 1009, 2027, 4099]
                h1 = de.act_probe(1009 * 2027 + 5, act, K=8)
                assert any(h[5] == 1009 * 2027 and h[6] == 5 for h in h1)
                h2 = de.act_probe((1009 + 4099) * 100 + 3, act, K=8)
                assert any(h[5] == (1009 + 4099) * 100 and h[6] == 3 for h in h2)
        # Verify compiler handles multi-target overlapping DAGs safely
        seq_defs = {1: ("/", 2, 2)}
        t_list = []
        prev = 11
        for step in range(2, 65):
            cur_v = prev + step
            if step not in seq_defs and step not in {2, 3, 5, 7, 11}:
                seq_defs[step] = ("+", 1, step - 1)
            seq_defs[cur_v] = ("+", prev, step)
            t_list.append(cur_v)
            prev = cur_v
        p_seq = make_puzzle([2, 3, 5, 7, 11], t_list, value_limit=10**9, pid="seq_compile")
        assert_valid(p_seq, compile_dag(p_seq, seq_defs, n_trials=16), "compile_dag multi-target")

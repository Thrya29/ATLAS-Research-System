from atlas.core.registry import ToolRegistry
from atlas.core.types import ExecutionReport, PlanStep, ToolResult
from benchmarks.harness import run, unnecessary_calls
from benchmarks.simulator.env import default_profiles, drift_schedule
from benchmarks.stats import holm, paired
from tests.conftest import spec


def test_simulation_is_deterministic():
    a = run("ATLAS", 7, 60, drift_schedule())
    b = run("ATLAS", 7, 60, drift_schedule())
    assert [(r.success, r.tools, r.latency_ms) for r in a] == [(r.success, r.tools, r.latency_ms) for r in b]


def test_common_random_numbers_shared_across_routers():
    # The oracle and B4 face the same tasks in the same order.
    a, b = run("Oracle", 3, 40), run("B4", 3, 40)
    assert [r.family for r in a] == [r.family for r in b]


def test_every_router_runs():
    for name in ["B0", "B1", "B2", "B3", "B4", "B5", "ATLAS", "Oracle"]:
        recs = run(name, 1, 30)
        assert len(recs) == 30 and all(r.calls >= 0 for r in recs)


def test_unnecessary_calls_metric():
    reg = ToolRegistry([spec("sql", ["m", "table"]), spec("py", ["n"], ["table"]), spec("web", ["w"]), spec("sql2", ["m"])])
    res = [ToolResult("sql", True, 1), ToolResult("py", True, 1), ToolResult("web", True, 1), ToolResult("sql2", True, 1)]
    rep = ExecutionReport(res, [PlanStep(r.tool, frozenset()) for r in res], {}, 0, 0)
    # web is irrelevant, sql2 duplicates an already-covered type.
    assert unnecessary_calls(frozenset({"m", "n"}), rep, reg) == 2


def test_stats_helpers():
    r = paired([0.9, 0.8, 0.85, 0.95, 0.9], [0.7, 0.6, 0.7, 0.8, 0.75])
    assert r["diff"] > 0 and r["lo"] > 0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_default_profiles_are_consistent():
    reg = ToolRegistry(p.spec for p in default_profiles())
    for s in reg:
        for r in s.requires:
            assert reg.providers(r), f"{s.name} requires unprovided {r}"

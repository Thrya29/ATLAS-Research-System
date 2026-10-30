from dataclasses import replace

import pytest

from atlas.core.types import ExecutionReport, PlanStep, RequestContext, ToolResult
from atlas.persistence import dump_state, load_state
from atlas.router import ATLASConfig, ATLASRouter, _responsibilities
from atlas.understanding.classifier import Understanding


class FixedUnderstanding:
    def __init__(self, types, family="fam"):
        self.types, self.family = frozenset(types), family

    def __call__(self, q):
        return Understanding(self.types, {}, self.family, {self.family: 1.0})


def router(reg, types, **cfg):
    return ATLASRouter(reg, FixedUnderstanding(types), replace(ATLASConfig(explore=False), **cfg), seed=0)


def teach(r, tool, rel=None, q=None, n=200):
    st = r._stats(r.registry.get(tool))
    for _i in range(n):
        if rel is not None:
            st.reliability.update(rel, r.now)
        if q is not None:
            st.quality.update("fam", q, r.now)


def test_picks_cheapest_plan_meeting_constraints(small_registry):
    r = router(small_registry, {"metrics", "number"})
    for t in ["sql", "sql_cheap", "python", "calc"]:
        teach(r, t, rel=1.0, q=1.0)
    d = r.route("q", RequestContext())
    assert set(d.plan.tools) == {"sql_cheap", "calc"} and d.plan.feasible


def test_low_quality_tool_is_avoided(small_registry):
    r = router(small_registry, {"metrics", "number"})
    for t in ["sql", "sql_cheap", "python"]:
        teach(r, t, rel=1.0, q=1.0)
    teach(r, "calc", rel=1.0, q=0.0)
    d = r.route("q", RequestContext(min_quality=0.8))
    assert "calc" not in d.plan.tools and "python" in d.plan.tools


def test_unauthorized_and_open_circuit_filtered(small_registry):
    from atlas.execution.circuit import CircuitBreakers

    cb = CircuitBreakers(clock=lambda: 0.0, failure_threshold=1)
    cb.record("sql_cheap", False)
    r = ATLASRouter(small_registry, FixedUnderstanding({"table"}), ATLASConfig(explore=False), cb)
    d = r.route("q", RequestContext(role="analyst"))
    assert d.rejected["fin"] == "unauthorized" and d.rejected["sql_cheap"] == "circuit_open"
    assert d.plan.tools == ["sql"]


def test_only_relevant_providers_are_scored(small_registry):
    d = router(small_registry, {"number"}).route("q", RequestContext(role="analyst"))
    # number's providers need table or scalars, so SQL tools (and fin, which provides table) are relevant;
    # docs is never considered.
    seen = {n for v in d.candidates.values() for n in v} | set(d.rejected)
    assert "docs" not in seen and {"python", "calc", "sql"} <= seen and d.rejected.get("fin") == "unauthorized"


@pytest.mark.parametrize("impute", [True, False])
def test_superseded_answer_is_not_silently_ignored(small_registry, impute):
    r = router(small_registry, {"metrics"}, impute_superseded=impute)
    for _ in range(200):
        r.calibration[0].update(0.0, r.now)  # very low confidence answers are wrong
    d = r.route("q", RequestContext())
    weak = ToolResult("sql_cheap", True, 10, confidence=0.05)
    strong = ToolResult("sql", True, 10, confidence=0.95)
    rep = ExecutionReport([weak, strong], d.plan.steps, {"metrics": strong}, 10, 0.5)
    before = r._stats(small_registry.get("sql_cheap")).quality.mean("fam", r.now)
    for _ in range(20):
        r.observe(d, rep, True)  # verification rescued every task
    after = r._stats(small_registry.get("sql_cheap")).quality.mean("fam", r.now)
    assert (after < before - 0.2) if impute else after == pytest.approx(before, abs=0.02)


def test_no_tool_query(small_registry):
    d = router(small_registry, set()).route("explain tcp", RequestContext())
    assert d.plan.steps == [] and "no external tool required" in d.notes


def test_missing_capability_is_reported(small_registry):
    d = router(small_registry, {"logs"}).route("q", RequestContext())
    assert not d.plan.steps and not d.plan.feasible


def test_sensitive_tools_are_not_explored(small_registry):
    r = ATLASRouter(small_registry, FixedUnderstanding({"market"}), ATLASConfig(explore=True), seed=1)
    spec = small_registry.get("fin")
    draws = {r._draw(spec, "fam", explore=True).q for _ in range(20)}
    assert len(draws) == 1  # posterior mean, no sampling


def test_value_of_information_stopping(small_registry):
    r = router(small_registry, {"metrics"})
    for t in ["sql", "sql_cheap"]:
        teach(r, t, rel=1.0, q=1.0)
    for _ in range(200):
        r.calibration[1].update(0.0, r.now)  # low-confidence evidence is usually wrong
    d = r.route("q", RequestContext(value_of_success=100.0))  # fixture tools cost 0.5-2 units
    low = ToolResult(d.plan.tools[0], True, 10, confidence=0.25)
    rep = ExecutionReport([low], d.plan.steps, {"metrics": low, "table": low, "scalars": low}, 10, 0.5)
    extra = r.hooks(d).followup(rep)
    assert len(extra) == 1 and extra[0].purpose == "verification"
    high = ToolResult(d.plan.tools[0], True, 10, confidence=0.95)
    for _ in range(200):
        r.calibration[4].update(1.0, r.now)
    rep2 = ExecutionReport([high], d.plan.steps, {"metrics": high}, 10, 0.5)
    assert r.hooks(d).followup(rep2) == []


def test_fallback_picks_alternative_provider(small_registry):
    r = router(small_registry, {"metrics"})
    d = r.route("q", RequestContext())
    failed = d.plan.steps[0]
    alt = r.hooks(d).fallback(failed, [ToolResult(failed.tool, False, 1)], {failed.tool})
    assert alt is not None and alt.tool != failed.tool and alt.purpose == "fallback"


def test_soft_credit_assignment_blames_low_confidence():
    a = ToolResult("a", True, 1, confidence=0.9)
    b = ToolResult("b", True, 1, confidence=0.1)
    rep = ExecutionReport([a, b], [PlanStep("a", frozenset()), PlanStep("b", frozenset())], {"x": a, "y": b}, 1, 0)
    w = _responsibilities(rep, frozenset({"x", "y"}))
    assert w["b"] == pytest.approx(0.9) and w["a"] == pytest.approx(0.1)


def test_state_roundtrip(small_registry):
    r = router(small_registry, {"metrics"})
    teach(r, "sql", rel=0.0, q=1.0, n=50)
    r2 = router(small_registry, {"metrics"})
    load_state(r2, dump_state(r))
    assert r2._stats(small_registry.get("sql")).reliability.mean(r.now) == pytest.approx(
        r._stats(small_registry.get("sql")).reliability.mean(r.now)
    )

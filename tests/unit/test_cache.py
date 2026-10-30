import asyncio
from dataclasses import replace

import pytest

from atlas.core.types import ExecutionReport, Plan, PlanStep, RequestContext, ToolResult
from atlas.execution.cache import ResultCache, age_bucket, serve, source_of, virtual_name
from atlas.execution.circuit import CircuitBreakers
from atlas.execution.executor import Executor
from atlas.router import ATLASConfig, ATLASRouter, _posterior_valid
from tests.unit.test_router import FixedUnderstanding, teach


def test_cache_expiry_lru_and_what_is_stored():
    now = {"t": 0}
    c = ResultCache(clock=lambda: now["t"], max_age=10, max_entries=2)
    c.put("sql", "Q  one", ToolResult("sql", True, 5, output=1))
    c.put("sql", "q two", ToolResult("sql", False, 5))  # failures are never cached
    c.put("sql", "q three", ToolResult("cache:sql", True, 1, meta={"cached": True}))  # nor served hits
    assert len(c) == 1
    now["t"] = 4
    hit = c.get("sql", "q one")  # normalised key
    assert hit.output == 1 and hit.meta["cache_age"] == 4
    now["t"] = 11
    assert c.get("sql", "q one") is None  # older than max_age
    c.put("a", "x", ToolResult("a", True, 1))
    c.put("b", "x", ToolResult("b", True, 1))
    c.put("c", "x", ToolResult("c", True, 1))
    assert c.get("a", "x") is None and c.get("c", "x") is not None  # LRU eviction


def test_age_buckets_and_names():
    assert [age_bucket(a) for a in (0, 31, 32, 200, 600)] == ["<32", "<32", "<128", "<512", ">=512"]
    assert source_of(virtual_name("sql")) == "sql" and source_of("sql") is None


def test_executor_serves_planned_hit_without_calling_tool(small_registry):
    called = []

    async def invoke(spec, inputs, attempt):
        called.append(spec.name)
        return ToolResult(spec.name, True, 50)

    cached = ToolResult("sql", True, 50, output="rows", confidence=0.9, meta={"inputs": [], "correct": True})
    plan = Plan(steps=[PlanStep(virtual_name("sql"), frozenset({"metrics"}), cached=cached)])
    ex = Executor({s.name: s for s in small_registry}, invoke, CircuitBreakers(clock=lambda: 0.0))
    rep = asyncio.run(ex.execute(plan, RequestContext()))
    assert called == [] and rep.cost == 0.0
    res = rep.supplied["metrics"]
    assert res.meta["cached"] and "inputs" not in res.meta and res.output == "rows"


def _router(reg, mode):
    r = ATLASRouter(reg, FixedUnderstanding({"metrics"}), ATLASConfig(explore=False, cache=mode, stopping=False), seed=0)
    for t in ("sql", "sql_cheap"):
        teach(r, t, rel=1.0, q=1.0)  # good sources, so a fresh cached answer clears the quality target
    return r


def _observe(r, d, tool):
    res = ToolResult(tool, True, 40, output="rows", confidence=0.9)
    r.observe(d, ExecutionReport([res], d.plan.steps, {"metrics": res}, 40, 1.0), True)


@pytest.mark.parametrize("mode", ["learned", "fixed"])
def test_repeated_query_is_served_from_cache(small_registry, mode):
    r = _router(small_registry, mode)
    d = r.route("revenue in Q1", RequestContext())
    tool = d.plan.steps[0].tool
    _observe(r, d, tool)
    d2 = r.route("Revenue in  Q1", RequestContext())
    step = d2.plan.steps[0]
    assert step.tool == virtual_name(tool) and step.cached is not None
    if mode == "learned":  # planned as a free candidate; fixed mode substitutes after planning
        assert d2.plan.est_cost == 0.0
    assert r.route("revenue in Q2", RequestContext()).plan.steps[0].tool == tool  # different query: no hit


def test_learned_cache_stops_using_stale_answers(small_registry):
    r = _router(small_registry, "learned")
    d = r.route("revenue today", RequestContext())
    tool = d.plan.steps[0].tool
    _observe(r, d, tool)
    for _ in range(30):  # cached answers to this family keep turning out wrong
        d2 = r.route("revenue today", RequestContext())
        assert d2.plan.steps
        res = serve(d2.plan.steps[0].tool, d2.plan.steps[0].cached) if d2.plan.steps[0].cached else ToolResult(tool, True, 40)
        res = replace(res, confidence=0.9)
        r.observe(d2, ExecutionReport([res], d2.plan.steps, {"metrics": res}, 1, 0), False)
    assert source_of(r.route("revenue today", RequestContext()).plan.steps[0].tool) is None  # back to calling a tool


def test_posterior_valid_splits_blame():
    assert _posterior_valid(0.9, 0.0) == pytest.approx(0.9)  # source always wrong: failure says nothing about staleness
    assert _posterior_valid(0.9, 1.0) == pytest.approx(0.0)  # source always right: it must have gone stale
    assert 0 < _posterior_valid(0.9, 0.8) < 0.9

import pytest

from atlas.core.registry import ToolRegistry
from atlas.planning.graph import build_plan, enumerate_plans
from tests.conftest import spec


def cands(reg: ToolRegistry):
    return {t: reg.providers(t) for t in reg.types}


def test_cheapest_plan_resolves_dependencies(small_registry):
    plans = enumerate_plans(frozenset({"metrics", "number"}), cands(small_registry), lambda s: s.cost)
    assert plans[0] == frozenset({"sql_cheap", "calc"})
    for p in plans:
        provided = frozenset().union(*(small_registry.get(n).provides for n in p))
        required = frozenset().union(*(small_registry.get(n).requires for n in p))
        assert {"metrics", "number"} <= provided and required <= provided


def test_plans_sorted_by_cost(small_registry):
    plans = enumerate_plans(frozenset({"number"}), cands(small_registry), lambda s: s.cost, k=10)
    costs = [sum(small_registry.get(n).cost for n in p) for p in plans]
    assert costs == sorted(costs)


def test_missing_provider_gives_no_plan(small_registry):
    assert enumerate_plans(frozenset({"logs"}), cands(small_registry), lambda s: s.cost) == []


def test_empty_requirement_is_empty_plan(small_registry):
    assert enumerate_plans(frozenset(), cands(small_registry), lambda s: s.cost) == [frozenset()]


def test_build_plan_wires_dag(small_registry):
    tools = {n: small_registry.get(n) for n in ["sql", "python"]}
    plan = build_plan(frozenset(tools), frozenset({"metrics", "number"}), tools)
    assert plan.tools == ["sql", "python"]
    assert plan.steps[1].depends_on == (0,)
    assert plan.levels() == [[0], [1]]


def test_build_plan_rejects_unsatisfied_input(small_registry):
    tools = {"python": small_registry.get("python")}
    with pytest.raises(ValueError):
        build_plan(frozenset(tools), frozenset({"number"}), tools)


def test_registry_rejects_self_dependency():
    with pytest.raises(ValueError):
        ToolRegistry([spec("loop", ["a"], ["a"])])

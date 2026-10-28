import asyncio

from atlas.core.types import FailureKind, Plan, PlanStep, RequestContext, ToolResult
from atlas.execution.circuit import CircuitBreakers
from atlas.execution.executor import Executor
from tests.conftest import spec

TOOLS = {
    s.name: s
    for s in [
        spec("a", ["x", "table"], timeout=500),
        spec("b", ["y"], ["table"]),
        spec("a2", ["x", "table"]),
        spec("secret", ["z"], roles=["admin"]),
    ]
}


def make(behaviour, fallback=None):
    calls = []

    async def invoke(s, inputs, attempt):
        calls.append((s.name, attempt))
        return behaviour(s, attempt)

    class Hooks:
        def fallback(self, failed, results, exclude):
            return fallback(failed, exclude) if fallback else None

        def followup(self, report):
            return []

    ex = Executor(TOOLS, invoke, CircuitBreakers(clock=lambda: 0.0))
    return ex, Hooks(), calls


def ok(dur):
    return lambda s, a: ToolResult(s.name, True, dur, confidence=0.9)


def run(ex, plan, hooks, role="analyst"):
    return asyncio.run(ex.execute(plan, RequestContext(role=role), hooks))


def test_critical_path_latency():
    ex, hooks, _ = make(ok(100))
    plan = Plan([PlanStep("a", frozenset({"x", "table"})), PlanStep("b", frozenset({"y"}), (0,))])
    rep = run(ex, plan, hooks)
    assert rep.latency_ms == 200 and set(rep.supplied) >= {"x", "y"}


def test_retry_then_success():
    ex, hooks, calls = make(lambda s, a: ToolResult(s.name, a > 0, 50, failure=None if a else FailureKind.ERROR))
    rep = run(ex, Plan([PlanStep("a", frozenset({"x"}))]), hooks)
    assert calls == [("a", 0), ("a", 1)] and "x" in rep.supplied


def test_slow_result_becomes_timeout_and_fallback_runs():
    def beh(s, a):
        return ToolResult(s.name, True, 900 if s.name == "a" else 50, confidence=0.8)

    ex, hooks, calls = make(beh, fallback=lambda failed, ex_: PlanStep("a2", failed.supplies, purpose="fallback"))
    rep = run(ex, Plan([PlanStep("a", frozenset({"x"}))]), hooks)
    assert [r.failure for r in rep.results[:2]] == [FailureKind.TIMEOUT, FailureKind.TIMEOUT]
    assert rep.supplied["x"].tool == "a2"
    assert any(s.purpose == "fallback" for s in rep.steps)


def test_dependency_failure_propagates():
    ex, hooks, calls = make(lambda s, a: ToolResult(s.name, False, 10, failure=FailureKind.ERROR))
    plan = Plan([PlanStep("a", frozenset({"table"})), PlanStep("b", frozenset({"y"}), (0,))])
    rep = run(ex, plan, hooks)
    assert rep.results[-1].failure is FailureKind.DEPENDENCY_FAILED
    assert ("b", 0) not in calls


def test_policy_denied_never_invokes_tool():
    ex, hooks, calls = make(ok(10))
    rep = run(ex, Plan([PlanStep("secret", frozenset({"z"}))]), hooks)
    assert calls == [] and rep.results[0].failure is FailureKind.POLICY_DENIED and rep.cost == 0

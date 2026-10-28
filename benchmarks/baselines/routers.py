"""Baseline routers. All share the executor, environment and task stream."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping

import numpy as np

from atlas.bandits.estimators import DecayedBeta
from atlas.core.registry import ToolRegistry
from atlas.core.types import Decision, ExecutionReport, Plan, PlanStep, RequestContext, ToolSpec
from atlas.execution.executor import NoHooks
from atlas.planning.graph import build_plan, enumerate_plans
from atlas.retrieval.hybrid import HybridRetriever
from atlas.understanding.classifier import MultiLabelTextClassifier, TaskUnderstanding
from benchmarks.simulator.env import SimEnvironment
from benchmarks.simulator.tasks import Task


def plan_from_tools(names: set[str], needed: frozenset[str], registry: ToolRegistry) -> Plan:
    """DAG if the tool set is self-sufficient, otherwise run everything in parallel."""
    tools = {n: registry.get(n) for n in names if n in registry}
    try:
        return build_plan(frozenset(tools), needed, tools)
    except ValueError:
        return Plan(steps=[PlanStep(n, tools[n].provides) for n in sorted(tools)])


class _Base:
    name = "base"

    def hooks(self, decision: Decision) -> NoHooks:
        return NoHooks()

    def observe(self, decision: Decision, report: ExecutionReport, success: bool | None) -> None:
        pass

    def warm_start(self, decision: Decision, report: ExecutionReport, success: bool | None) -> None:
        self.observe(decision, report, success)

    def _decision(self, query, ctx, names, needed, t0, family=None) -> Decision:
        plan = plan_from_tools(set(names), needed, self.registry)
        return Decision(
            query, ctx, plan, frozenset(needed), family, router=self.name, planning_ms=(time.perf_counter() - t0) * 1000
        )


class SingleToolRouter(_Base):
    """B0: one fixed tool per (predicted) task category."""

    name = "B0-SingleTool"
    TOOL = {
        "revenue_growth": "sql_primary",
        "cohort_analysis": "sql_primary",
        "kpi_today": "sql_primary",
        "policy_lookup": "vector_docs",
        "incident_triage": "log_search",
        "market_compare": "fin_api",
        "stock_quote": "fin_api",
        "breaking_news": "search_a",
        "claim_check": "search_a",
        "metric_explainer": "vector_docs",
    }

    def __init__(self, registry: ToolRegistry, understanding: TaskUnderstanding) -> None:
        self.registry, self.u = registry, understanding

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        fam = self.u(query).family
        tool = self.TOOL.get(fam)
        return self._decision(query, ctx, [tool] if tool else [], frozenset(), t0, fam)


class RuleRouter(_Base):
    """B1: hand-written keyword rules (authored from training templates), permission-aware."""

    name = "B1-Rules"
    R = {
        "internal": re.compile(
            r"\b(our|we|revenue|sales|gross margin|arr|bookings|signups|churn|order volume|today's|booked|segment|cohort|accounts?|kpis?)\b",
            re.I,
        ),
        "compute": re.compile(
            r"\b(growth|percent(age)?|compute|calculate|ratio|rate|regression|trend|median|variance|retention|change|benchmark|compare)\b",
            re.I,
        ),
        "docs": re.compile(
            r"\b(polic(y|ies)|handbook|guidelines?|document|runbooks?|playbook|memo|docs?|commentary|reports)\b", re.I
        ),
        "logs": re.compile(r"\b(logs?|errors?|exceptions?|outage|incident|spikes?|stack)\b", re.I),
        "market": re.compile(r"\b(stock|share price|market cap|ticker|trading|quote|market)\b", re.I),
        "web": re.compile(
            r"\b(news|headlines|latest|recent|web|online|sources|fact-check|verify|true that|developments)\b", re.I
        ),
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        hit = {k: bool(r.search(query)) for k, r in self.R.items()}
        tools: list[str] = []
        needed: set[str] = set()
        if hit["internal"]:
            tools.append("sql_primary")
            needed.add("internal_metrics")
        if hit["docs"]:
            tools.append("vector_docs")
            needed.add("internal_docs")
        if hit["logs"]:
            tools.append("log_search")
            needed.add("logs")
        if hit["market"]:
            fin = self.registry.get("fin_api")
            tools.append("fin_api" if fin.allowed_for(ctx.role) else "search_a")
            needed.add("market_data")
        if hit["web"]:
            tools.append("search_a")
            needed.add("web_facts")
        if hit["compute"]:
            has_table = any("table" in self.registry.get(t).provides for t in tools)
            tools.append("python_sandbox" if has_table else "calculator")
            needed.add("numeric_result")
            if not has_table and not tools[:-1]:
                tools.insert(0, "search_a")
        return self._decision(query, ctx, dict.fromkeys(tools), frozenset(needed), t0)


class SupervisedRouter(_Base):
    """B2: model-based router — multi-label classifier from query (+role) to tool set.

    Trained on oracle-optimal plans under stationary conditions, i.e. an
    *optimistic* model-based baseline (it is given privileged supervision).
    """

    name = "B2-Supervised"

    def __init__(self, registry: ToolRegistry, texts: list[str], roles: list[str], tool_sets: list[frozenset[str]]) -> None:
        self.registry = registry
        self.clf = MultiLabelTextClassifier().fit([f"{t} role_{r}" for t, r in zip(texts, roles, strict=True)], tool_sets)

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        tools = self.clf.predict([f"{query} role_{ctx.role}"])[0]
        needed = frozenset().union(*(self.registry.get(t).provides for t in tools)) if tools else frozenset()
        return self._decision(query, ctx, tools, needed, t0)


class RetrievalRouter(_Base):
    """B3: hybrid BM25 + vector retrieval over tool descriptions, top-k tools."""

    name = "B3-Retrieval"

    def __init__(self, registry: ToolRegistry, k: int = 2) -> None:
        self.registry, self.k = registry, k
        specs = list(registry)
        self.ret = HybridRetriever([s.name for s in specs], [s.description for s in specs])

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        tools = self.ret.top_k(query, self.k)
        needed = frozenset().union(*(self.registry.get(t).provides for t in tools))
        return self._decision(query, ctx, tools, needed, t0)


class ContextualBanditRouter(_Base):
    """B5: discounted Thompson sampling per (family, tool) on a scalarised reward.

    For each predicted type it samples the provider with the best
    reward = success - cost/V - 0.2 * latency/L_max. No dependency planning,
    no explicit constraints, no fallback or verification.
    """

    name = "B5-ContextualBandit"

    def __init__(self, registry: ToolRegistry, understanding: TaskUnderstanding, seed: int = 0, gamma: float = 0.997) -> None:
        self.registry, self.u = registry, understanding
        self.rng = np.random.default_rng(seed)
        self.arms: dict[tuple[str, str], DecayedBeta] = {}
        self.gamma, self.now = gamma, 0

    def _arm(self, fam: str, tool: str) -> DecayedBeta:
        return self.arms.setdefault((fam, tool), DecayedBeta(0.5, prior_strength=2.0, gamma=self.gamma))

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        self.now += 1
        u = self.u(query)
        chosen: set[str] = set()
        for t in sorted(u.types):
            provs = [p for p in self.registry.providers(t) if p.allowed_for(ctx.role)]
            if provs:
                chosen.add(max(provs, key=lambda p: self._arm(u.family, p.name).sample(self.rng, self.now)).name)
        return self._decision(query, ctx, chosen, u.types, t0, u.family)

    def observe(self, decision: Decision, report: ExecutionReport, success: bool | None) -> None:
        if success is None or decision.family is None:
            return
        ctx = decision.context
        r = (1.0 if success else 0.0) - report.cost / ctx.value_of_success - 0.2 * report.latency_ms / ctx.latency_budget_ms
        r = min(max(r, 0.0), 1.0)
        for s in decision.plan.steps:
            self._arm(decision.family, s.tool).update(r, self.now)


class OracleRouter(_Base):
    """Privileged reference: true task requirements and true *current* tool parameters."""

    name = "Oracle"

    def __init__(self, registry: ToolRegistry, env: SimEnvironment, samples: int = 256, seed: int = 0) -> None:
        self.registry, self.env, self.samples = registry, env, samples
        self.rng = np.random.default_rng(seed)
        self.task: Task | None = None
        self.t = 0

    def set_task(self, task: Task, t: int) -> None:
        self.task, self.t = task, t

    def best_tools(self, task: Task, t: int, ctx: RequestContext) -> frozenset[str]:
        if not task.required:
            return frozenset()
        cands: dict[str, list[ToolSpec]] = {}
        for spec in self.registry:
            if spec.allowed_for(ctx.role):
                for ty in spec.provides:
                    cands.setdefault(ty, []).append(spec)
        prof = {s.name: self.env.profile(s.name, t) for v in cands.values() for s in v}
        sets = enumerate_plans(task.required, cands, lambda s: s.cost * (1 + prof[s.name].failure), k=64)
        best, best_key = frozenset(), None
        for ts in sets:
            plan = build_plan(ts, task.required, {n: self.registry.get(n) for n in ts})
            r = math.prod(1 - prof[n].failure ** 2 for n in ts)
            q = math.prod(prof[n].q(task.family) for n in ts)
            finish = np.zeros((len(plan.steps), self.samples))
            for i, s in enumerate(plan.steps):
                p = prof[s.tool]
                lat = np.exp(self.rng.normal(math.log(p.latency_ms), p.sigma, self.samples))
                finish[i] = (finish[list(s.depends_on)].max(axis=0) if s.depends_on else 0.0) + lat
            tail = float((finish.max(axis=0) > ctx.latency_budget_ms).mean())
            ok = q >= ctx.min_quality and r >= 1 - ctx.max_failure and tail <= ctx.latency_tail
            cost = sum(self.registry.get(n).cost * (1 + prof[n].failure) for n in ts)
            key = (0, cost, -r * q) if ok else (1, -r * q * (1 - tail), cost)
            if best_key is None or key < best_key:
                best, best_key = ts, key
        return best

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        assert self.task is not None, "harness must call set_task()"
        tools = self.best_tools(self.task, self.t, ctx)
        return self._decision(query, ctx, tools, self.task.required, t0, self.task.family)


def registry_of(specs: Mapping[str, ToolSpec] | list[ToolSpec]) -> ToolRegistry:
    return ToolRegistry(specs.values() if isinstance(specs, Mapping) else specs)

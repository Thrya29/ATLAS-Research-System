"""ATLAS: Adaptive Tool Learning and Allocation System.

Decision procedure for one query q with runtime context x:

1. Understanding    — predict required capability types and task family.
2. Candidates       — per type, providers from the capability index, filtered
                      for authorization, circuit health and input
                      satisfiability; top-m kept by estimated utility.
3. Posterior draw   — one Thompson sample per candidate tool of reliability,
                      context-specific quality and log-latency from
                      time-decayed posteriors (sensitive tools use the mean:
                      no optimistic exploration on them).
4. Plan search      — branch-and-bound over the AND-OR capability graph for
                      the k cheapest feasible tool sets.
5. Constrained pick — minimise expected cost subject to
                      Q(P) >= Q_min, R(P) >= 1 - eps, P(L(P) > L_max) <= delta;
                      if nothing satisfies the constraints, maximise expected
                      on-time success.
6. Execute          — DAG executor; on terminal failure, replan the failed
                      step's types with the best remaining provider.
7. Stop or verify   — for each required type, call another provider only if
                      the expected value of the extra evidence exceeds its cost
                      and fits the remaining latency budget.
8. Learn            — semi-bandit updates of reliability/latency per call,
                      credit-assigned quality per (tool, family), calibration
                      of confidence signals.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from atlas.bandits.estimators import DecayedBeta, DecayedLogNormal, HierarchicalQuality
from atlas.core.registry import ToolRegistry
from atlas.core.types import (
    Decision,
    ExecutionReport,
    FailureKind,
    Plan,
    PlanStep,
    RequestContext,
    ToolResult,
    ToolSpec,
)
from atlas.execution.cache import ResultCache, age_bucket, source_of, virtual_name, virtual_spec
from atlas.execution.circuit import CircuitBreakers
from atlas.planning.graph import build_plan, enumerate_plans
from atlas.understanding.classifier import TaskUnderstanding

TOOL_FAULTS = {FailureKind.TIMEOUT, FailureKind.ERROR, FailureKind.RATE_LIMITED}
CONF_BINS = 5


@dataclass(frozen=True)
class ATLASConfig:
    contextual: bool = True  # A1: quality per task family (else per tool only)
    learn_reliability: bool = True  # A2
    cost_aware: bool = True  # A3: minimise cost (else maximise success, fewest tools)
    plan_graph: bool = True  # A4: dependency-aware plan search (else greedy per type)
    stopping: bool = True  # A5: value-of-information verification
    learn_latency: bool = True  # A6
    feasibility_filter: bool = True  # A7: RBAC / circuit / schema filtering
    fallback: bool = True  # A8: replan failed steps
    explore: bool = True  # Thompson sampling (else posterior means)
    online: bool = True  # update statistics after deployment
    impute_superseded: bool = True  # A10: learn from answers superseded by verification
    cache: str = "off"
    """Result cache (E7): "off"; "fixed" = serve any fresh hit; "learned" = hits are planned as virtual tools."""
    cache_max_age: float = 500.0
    """Oldest cache entry ever served, in decisions."""
    uniform_policy: bool = False  # pick uniformly among candidate plans (offline logging policy)
    candidates_per_type: int = 4
    k_plans: int = 32
    latency_samples: int = 64
    gamma: float = 0.997
    gamma_quality: float = 0.998
    max_retries: int = 1
    latency_price: float = 0.2
    """Fraction of the value of success charged for using the whole latency budget."""


@dataclass
class ToolStats:
    reliability: DecayedBeta
    latency: DecayedLogNormal
    quality: HierarchicalQuality


@dataclass
class _Draw:
    rel: float
    q: float
    mu: float
    sigma: float
    cost: float


class ATLASRouter:
    name = "ATLAS"

    def __init__(
        self,
        registry: ToolRegistry,
        understanding: TaskUnderstanding,
        config: ATLASConfig | None = None,
        breakers: CircuitBreakers | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        self.registry = registry
        self.understanding = understanding
        self.cfg = config or ATLASConfig()
        self.breakers = breakers
        self.rng = np.random.default_rng(seed)
        self.now = 0
        self.stats: dict[str, ToolStats] = {}
        self.calibration = [DecayedBeta((i + 0.5) / CONF_BINS, gamma=self.cfg.gamma) for i in range(CONF_BINS)]
        self.cache = ResultCache(clock=lambda: self.now, max_age=self.cfg.cache_max_age) if self.cfg.cache != "off" else None
        self._virtual: dict[str, ToolSpec] = {}
        self._age: dict[str, float] = {}  # virtual tool -> age of its cached answer, for this decision
        self.freshness: dict[str, DecayedBeta] = {}
        self._frozen = False
        if name:
            self.name = name

    # ------------------------------------------------------------------ stats
    def _lookup(self, name: str) -> ToolSpec | None:
        """A registered tool, or the virtual ``cache:<tool>`` for a registered source."""
        if name in self.registry:
            return self.registry.get(name)
        src = source_of(name)
        if src is None or src not in self.registry:
            return None
        if name not in self._virtual:
            self._virtual[name] = virtual_spec(self.registry.get(src))
        return self._virtual[name]

    def _stats(self, spec: ToolSpec) -> ToolStats:
        s = self.stats.get(spec.name)
        if s is None:
            s = ToolStats(
                DecayedBeta(1 - spec.prior_failure, prior_strength=4.0, gamma=self.cfg.gamma),
                DecayedLogNormal(spec.prior_latency_ms, gamma=self.cfg.gamma),
                HierarchicalQuality(spec.prior_quality, gamma=self.cfg.gamma_quality),
            )
            self.stats[spec.name] = s
        return s

    def freeze(self) -> None:
        """Stop online learning (used by the static cost-aware baseline)."""
        self._frozen = True

    def _context(self, family: str | None) -> str | None:
        return family if self.cfg.contextual else None

    def _freshness(self, family: str | None, age: float) -> DecayedBeta:
        """P(a cached answer is still valid | task family, age bucket), shared by every tool's cache.

        Staleness is a property of the question, not of the tool that answered it,
        so all ``cache:*`` tools pool their evidence here.
        """
        key = f"{self._context(family) or '*'}|{age_bucket(age)}"
        if key not in self.freshness:
            self.freshness[key] = DecayedBeta(0.9, prior_strength=2.0, gamma=self.cfg.gamma_quality)
        return self.freshness[key]

    def _draw(self, spec: ToolSpec, family: str | None, explore: bool) -> _Draw:
        st, now, ctx = self._stats(spec), self.now, self._context(family)
        explore = explore and self.cfg.explore and not spec.sensitive
        if self.cfg.learn_reliability:
            rel = st.reliability.sample(self.rng, now) if explore else st.reliability.mean(now)
        else:
            rel = 1 - spec.prior_failure
        src = source_of(spec.name)
        if src is not None and spec.name in self._age:
            # Cached answer: right when produced (source tool's quality) and still valid (freshness).
            qs, fr = self._stats(self.registry.get(src)).quality, self._freshness(family, self._age[spec.name])
            q = qs.sample(self.rng, ctx, now) * fr.sample(self.rng, now) if explore else qs.mean(ctx, now) * fr.mean(now)
        else:
            q = st.quality.sample(self.rng, ctx, now) if explore else st.quality.mean(ctx, now)
        if self.cfg.learn_latency:
            if explore:
                mu, sigma = st.latency.sample_mu(self.rng, now)
            else:
                mu, sigma, _ = st.latency.moments(now)
        else:
            mu, sigma = math.log(spec.prior_latency_ms), st.latency.prior_sigma
        attempts = sum((1 - rel) ** i for i in range(self.cfg.max_retries + 1))
        return _Draw(rel, q, mu, sigma, spec.cost * attempts)

    def expected_utility(self, spec: ToolSpec, family: str | None, value: float) -> float:
        d = self._draw(spec, family, explore=False)
        return d.rel * d.q * value - d.cost

    # -------------------------------------------------------------- candidates
    def _relevant(self, needed: frozenset[str]) -> list[str]:
        """Needed types plus every type their providers (transitively) require."""
        seen, frontier = set(needed), list(needed)
        while frontier:
            for spec in self.registry.providers(frontier.pop()):
                for r in spec.requires - seen:
                    seen.add(r)
                    frontier.append(r)
        return sorted(seen)

    def _rejection(self, spec: ToolSpec, ctx: RequestContext) -> str | None:
        if not self.cfg.feasibility_filter:
            return None
        if not spec.allowed_for(ctx.role):
            return "unauthorized"
        if self.breakers is not None and not self.breakers.allow(spec.name):
            return "circuit_open"
        if any(not self.registry.has_providers(r) for r in spec.requires):
            return "unsatisfiable_input"
        return None

    def _candidates(
        self, ctx: RequestContext, family: str | None, needed: frozenset[str]
    ) -> tuple[dict[str, list[ToolSpec]], dict[str, str]]:
        rejected: dict[str, str] = {}
        by_type: dict[str, list[ToolSpec]] = {}
        verdict: dict[str, str | None] = {}
        for t in self._relevant(needed):
            for spec in self.registry.providers(t):
                if spec.name not in verdict:
                    verdict[spec.name] = self._rejection(spec, ctx)
                    if verdict[spec.name]:
                        rejected[spec.name] = verdict[spec.name]
                if verdict[spec.name] is None:
                    by_type.setdefault(t, []).append(spec)
        m = self.cfg.candidates_per_type
        for t, specs in by_type.items():
            specs.sort(key=lambda s: -self.expected_utility(s, family, ctx.value_of_success))
            for s in specs[m:]:
                rejected.setdefault(s.name, f"pruned_top{m}:{t}")
            by_type[t] = specs[:m]
        return by_type, rejected

    # ------------------------------------------------------------------ route
    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        self.now += 1
        u = self.understanding(query)
        family = u.family
        needed = u.types
        decision = Decision(
            query=query, context=ctx, plan=Plan(steps=[]), predicted_types=needed, family=family, router=self.name
        )
        if not needed:
            decision.notes.append("no external tool required")
            decision.planning_ms = (time.perf_counter() - t0) * 1000
            return decision

        cands, rejected = self._candidates(ctx, family, needed)
        hits = self._cache_candidates(query, cands, ctx) if self.cfg.cache == "learned" else {}
        decision.rejected = rejected
        decision.candidates = {t: [s.name for s in v] for t, v in cands.items()}
        draws = {s.name: self._draw(s, family, explore=True) for v in cands.values() for s in v}

        if self.cfg.plan_graph:
            cost_fn = (lambda s: draws[s.name].cost) if self.cfg.cost_aware else (lambda s: 1.0)
            tool_sets = enumerate_plans(needed, cands, cost_fn, k=self.cfg.k_plans)
        else:
            tool_sets = self._greedy(needed, cands, draws)

        if not tool_sets:
            decision.notes.append("no feasible plan: missing providers for " + ", ".join(sorted(needed - set(cands))))
            decision.plan = Plan(steps=[], feasible=False)
            decision.planning_ms = (time.perf_counter() - t0) * 1000
            return decision

        tools = {s.name: s for v in cands.values() for s in v}
        if self.cfg.uniform_policy:
            tool_sets = [tool_sets[int(self.rng.integers(len(tool_sets)))]]
        best: Plan | None = None
        best_key: tuple[float, ...] | None = None
        for ts in tool_sets:
            try:
                plan = build_plan(ts, needed, tools) if self.cfg.plan_graph else _flat_plan(ts, needed, tools)
            except ValueError:
                continue
            self._score(plan, draws, ctx)
            if plan.feasible:
                key = (0.0, plan.est_cost if self.cfg.cost_aware else -plan.est_success, -plan.est_success, len(ts))
            else:
                key = (1.0, -plan.est_success * (1 - plan.est_latency_tail), plan.est_cost, len(ts))
            if best_key is None or key < best_key:
                best, best_key = plan, key
        decision.plan = best or Plan(steps=[], feasible=False)
        if self.cfg.cache == "fixed":
            hits = self._substitute_cached(query, decision.plan)
        for step in decision.plan.steps:
            step.cached = hits.get(step.tool)
        decision.notes.append(
            "constraints satisfied" if decision.plan.feasible else "no plan met constraints; maximised on-time success"
        )
        decision.planning_ms = (time.perf_counter() - t0) * 1000
        return decision

    def _cache_candidates(self, query: str, cands: dict[str, list[ToolSpec]], ctx: RequestContext) -> dict[str, ToolResult]:
        """Learned caching: each fresh hit joins its types' candidates as a virtual tool."""
        hits: dict[str, ToolResult] = {}
        self._age = {}
        for specs in cands.values():
            for spec in list(specs):
                if source_of(spec.name) is not None:
                    continue
                hit = self.cache.get(spec.name, query)
                v = self._lookup(virtual_name(spec.name))
                if hit is None or not v.allowed_for(ctx.role):
                    continue
                hits[v.name] = hit
                self._age[v.name] = hit.meta["cache_age"]
        for t, specs in cands.items():
            specs.extend(self._lookup(n) for n in hits if t in self._lookup(n).provides and self._lookup(n) not in specs)
        return hits

    def _substitute_cached(self, query: str, plan: Plan) -> dict[str, ToolResult]:
        """Fixed-TTL caching: any planned call with a fresh hit is served from the cache."""
        hits: dict[str, ToolResult] = {}
        for i, step in enumerate(plan.steps):
            hit = self.cache.get(step.tool, query)
            if hit is not None:
                name = virtual_name(step.tool)
                plan.steps[i] = PlanStep(name, step.supplies, (), step.purpose)
                hits[name] = hit
        return hits

    def _greedy(self, needed, cands, draws) -> list[frozenset[str]]:
        """A4 ablation: best provider per predicted type, ignoring dependencies."""
        chosen = set()
        for t in needed:
            if t not in cands:
                return []
            chosen.add(max(cands[t], key=lambda s: draws[s.name].rel * draws[s.name].q - draws[s.name].cost).name)
        return [frozenset(chosen)]

    def _score(self, plan: Plan, draws: Mapping[str, _Draw], ctx: RequestContext) -> None:
        r = q = 1.0
        cost = 0.0
        for s in plan.steps:
            d = draws[s.tool]
            r *= 1 - (1 - d.rel) ** (self.cfg.max_retries + 1)
            q *= d.q
            cost += d.cost
        n = self.cfg.latency_samples
        finish = np.zeros((len(plan.steps), n))
        for i, s in enumerate(plan.steps):
            d = draws[s.tool]
            lat = np.exp(self.rng.normal(d.mu, d.sigma, n))
            start = finish[list(s.depends_on)].max(axis=0) if s.depends_on else 0.0
            finish[i] = start + lat
        tail = float((finish.max(axis=0) > ctx.latency_budget_ms).mean()) if len(plan.steps) else 0.0
        plan.est_reliability, plan.est_quality, plan.est_cost = r, q, cost
        plan.est_success, plan.est_latency_tail = r * q, tail
        plan.feasible = q >= ctx.min_quality and r >= 1 - ctx.max_failure and tail <= ctx.latency_tail

    # ------------------------------------------------------------------ hooks
    def hooks(self, decision: Decision) -> _Hooks:
        return _Hooks(self, decision)

    # ---------------------------------------------------------------- learning
    def observe(self, decision: Decision, report: ExecutionReport, success: bool | None) -> None:
        if self.cache is not None:
            for res in report.results:
                if res.ok and source_of(res.tool) is None:
                    self.cache.put(res.tool, decision.query, res)
        if self._frozen or not self.cfg.online:
            return
        self._learn(decision, report, success)

    def _learn_freshness(self, family: str | None, name: str, age: float, success: bool, w: float) -> None:
        fr = self._freshness(family, age)
        if success:
            fr.update(1.0, self.now)
        elif w > 0:
            # A wrong cached answer was either wrong when produced or has gone stale: split the evidence.
            q = self._stats(self.registry.get(source_of(name))).quality.mean(self._context(family), self.now)
            fr.update(_posterior_valid(fr.mean(self.now), q), self.now, w)

    def warm_start(self, decision: Decision, report: ExecutionReport, success: bool | None) -> None:
        """Learn from offline logs regardless of the online/frozen settings."""
        self._learn(decision, report, success)

    def _learn(self, decision: Decision, report: ExecutionReport, success: bool | None) -> None:
        now = self.now
        for res in report.results:
            spec = self._lookup(res.tool)
            if spec is None:
                continue
            st = self._stats(spec)
            if res.ok:
                st.reliability.update(1.0, now)
                st.latency.update(res.duration_ms, now)
            elif res.failure in TOOL_FAULTS:
                st.reliability.update(0.0, now)
                if res.failure is FailureKind.TIMEOUT:
                    st.latency.update(res.duration_ms, now)
        if success is None or not decision.predicted_types:
            return
        needed = decision.predicted_types
        if not needed <= set(report.supplied):
            return  # execution failure, not an answer-quality signal
        ctx = self._context(decision.family)
        ages = {r.tool: r.meta["cache_age"] for r in report.results if "cache_age" in r.meta}
        blame = _responsibilities(report, needed)
        for name, w in blame.items():
            if self._lookup(name) is None:
                continue
            if name in ages:
                self._learn_freshness(decision.family, name, ages[name], success, w)
                continue
            q = self._stats(self._lookup(name)).quality
            if success:
                q.update(ctx, 1.0, now)
            elif w > 0:
                q.update(ctx, 0.0, now, w)
        if self.cfg.impute_superseded:
            # An answer superseded by a more confident one never reaches the
            # grader, so its correctness is censored. Ignoring it hides exactly
            # the mistakes verification caught and inflates the tool's quality;
            # impute the calibrated probability of being correct instead.
            for res in report.results:
                spec = self._lookup(res.tool)
                if not res.ok or res.confidence is None or res.tool in blame or spec is None or res.tool in ages:
                    continue
                if spec.provides & needed:
                    p = self.calibration[_bin(res.confidence)].mean(now)
                    self._stats(spec).quality.update(ctx, p, now)
        for t in needed:
            res = report.supplied[t]
            if res.confidence is None:
                continue
            w = 1.0 if success else blame.get(res.tool, 0.0)
            if w > 0:
                self.calibration[_bin(res.confidence)].update(1.0 if success else 0.0, now, w)


def _posterior_valid(v: float, q: float) -> float:
    """P(still valid | cached answer judged wrong), with P(valid)=v and P(right when produced)=q."""
    return v * (1 - q) / max(1e-9, 1 - q * v)


def _responsibilities(report: ExecutionReport, needed: frozenset[str]) -> dict[str, float]:
    """Soft credit assignment for a failed answer.

    Only the task-level outcome is observed. Blame is shared among the tools
    whose evidence was used, in proportion to (1 - confidence), so the most
    doubtful evidence absorbs most of it. Tools without a confidence signal get
    an uninformative 0.5. Weights sum to 1.
    """
    raw = {name: 1.0 - (c if c is not None else 0.5) for name, c in _contributors(report, needed).items()}
    z = sum(raw.values())
    if z <= 0:
        return {n: 1.0 / len(raw) for n in raw} if raw else {}
    return {n: v / z for n, v in raw.items()}


def _contributors(report: ExecutionReport, needed: frozenset[str]) -> dict[str, float | None]:
    """Tools whose output was used as evidence for a needed type (plus their inputs) -> confidence."""
    out: dict[str, float | None] = {}
    frontier = [report.supplied[t] for t in needed if t in report.supplied]
    while frontier:
        r = frontier.pop()
        if r.tool in out:
            continue
        out[r.tool] = r.confidence
        frontier.extend(r.meta.get("inputs", []))
    return out


def _bin(c: float) -> int:
    return min(int(c * CONF_BINS), CONF_BINS - 1)


def _flat_plan(ts: frozenset[str], needed: frozenset[str], tools: Mapping[str, ToolSpec]) -> Plan:
    return Plan(steps=[PlanStep(n, tools[n].provides & needed) for n in sorted(ts)])


@dataclass
class _Hooks:
    router: ATLASRouter
    decision: Decision
    verified: set[str] = field(default_factory=set)

    def _alternatives(self, types: frozenset[str], exclude: set[str]) -> list[ToolSpec]:
        r, ctx = self.router, self.decision.context
        alts, seen = [], set(exclude)
        for t in sorted(types):
            for spec in r.registry.providers(t):
                if spec.name in seen:
                    continue
                seen.add(spec.name)
                if r.cfg.feasibility_filter and (
                    not spec.allowed_for(ctx.role) or (r.breakers is not None and not r.breakers.allow(spec.name))
                ):
                    continue
                alts.append(spec)
        return alts

    def fallback(self, failed: PlanStep, results: list[ToolResult], exclude: set[str]) -> PlanStep | None:
        r = self.router
        if not r.cfg.fallback:
            return None
        exclude = exclude | {res.tool for res in results if not res.ok}
        alts = [s for s in self._alternatives(failed.supplies, exclude) if failed.supplies <= s.provides]
        if not alts:
            return None
        v = self.decision.context.value_of_success
        best = max(alts, key=lambda s: r.expected_utility(s, self.decision.family, v))
        return PlanStep(best.name, failed.supplies & best.provides, failed.depends_on, purpose="fallback")

    def followup(self, report: ExecutionReport) -> list[PlanStep]:
        """Value-of-information stopping rule (at most one verification per needed type).

        Evidence whose *calibrated* probability of being correct is already at
        or above the request's quality target is accepted. Otherwise another
        provider is called only if

            (1 - p_now) * rel_alt * q_alt * V  >  cost_alt + price * V * lat_alt / L_max

        and its expected latency fits in the remaining budget.
        """
        r, ctx = self.router, self.decision.context
        if not r.cfg.stopping:
            return []
        remaining = ctx.latency_budget_ms - report.latency_ms
        used = {res.tool for res in report.results}
        steps: list[PlanStep] = []
        for t in sorted(self.decision.predicted_types):
            res = report.supplied.get(t)
            if res is None or res.confidence is None:
                continue
            p_now = r.calibration[_bin(res.confidence)].mean(r.now)
            if p_now >= ctx.min_quality:
                continue
            best_gain, best_spec = 0.0, None
            for spec in self._alternatives(frozenset({t}), used):
                if not spec.requires <= set(report.supplied):
                    continue
                d = r._draw(spec, self.decision.family, explore=False)
                lat = math.exp(d.mu + d.sigma**2 / 2)
                if lat > remaining:
                    continue
                v = ctx.value_of_success
                gain = (1 - p_now) * d.rel * d.q * v - d.cost - r.cfg.latency_price * v * lat / ctx.latency_budget_ms
                if gain > best_gain:
                    best_gain, best_spec = gain, spec
            if best_spec is not None:
                steps.append(PlanStep(best_spec.name, frozenset({t}), purpose="verification"))
                used.add(best_spec.name)
        return steps


def static_router(registry: ToolRegistry, understanding: TaskUnderstanding, breakers=None, seed: int = 0) -> ATLASRouter:
    """B4: same planner, historical averages only (no exploration/adaptation/VoI/replanning)."""
    cfg = replace(ATLASConfig(), explore=False, online=False, stopping=False, fallback=False)
    return ATLASRouter(registry, understanding, cfg, breakers, seed, name="B4-StaticCostAware")

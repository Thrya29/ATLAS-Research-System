"""Controlled tool environment with ground-truth parameters and drift.

Every stochastic outcome is drawn from a generator seeded by
(seed, task id, tool, attempt), i.e. *common random numbers*: two routers that
call the same tool on the same task observe the same latency, failure and
correctness. Differences between routers are therefore due to their decisions,
which makes paired comparisons far more sensitive than independent runs.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from atlas.core.types import FailureKind, ToolResult, ToolSpec
from benchmarks.simulator.tasks import Task

DIRECT_ANSWER_QUALITY = 0.95


@dataclass(frozen=True)
class Profile:
    """Ground truth for one tool. ``spec`` is what the router is told."""

    spec: ToolSpec
    latency_ms: float
    sigma: float
    failure: float
    quality: float
    quality_by_family: Mapping[str, float] = field(default_factory=dict)

    def q(self, family: str) -> float:
        return self.quality_by_family.get(family, self.quality)


@dataclass(frozen=True)
class Drift:
    start: int
    end: int
    tool: str
    latency_mult: float = 1.0
    failure: float | None = None
    quality_mult: float = 1.0


def _spec(name, desc, provides, requires=(), cost=0.0, latency=200.0, roles=(), sensitive=False, timeout=2000.0):
    return ToolSpec(
        name=name,
        description=desc,
        provides=frozenset(provides),
        requires=frozenset(requires),
        cost=cost,
        timeout_ms=timeout,
        roles=frozenset(roles),
        sensitive=sensitive,
        prior_latency_ms=latency,  # advertised latency; true value may differ
        prior_failure=0.05,
        prior_quality=0.8,
    )


def default_profiles() -> list[Profile]:
    P = Profile
    return [
        P(
            _spec(
                "sql_primary",
                "Read-only SQL on the primary warehouse: revenue, sales, KPIs, metrics tables",
                ["internal_metrics", "table", "scalars"],
                cost=0.002,
                latency=120,
            ),
            120,
            0.30,
            0.010,
            0.97,
        ),
        P(
            _spec(
                "sql_replica",
                "Read-only SQL on the analytics replica (asynchronously replicated): metrics tables",
                ["internal_metrics", "table", "scalars"],
                cost=0.0005,
                latency=80,
            ),
            70,
            0.30,
            0.015,
            0.96,
            {"kpi_today": 0.55},
        ),
        P(
            _spec(
                "vector_docs",
                "Semantic search over internal documents, policies, runbooks and memos",
                ["internal_docs"],
                cost=0.001,
                latency=100,
            ),
            90,
            0.30,
            0.010,
            0.93,
        ),
        P(
            _spec(
                "vector_legacy",
                "Legacy keyword index over the old intranet wiki and documents",
                ["internal_docs"],
                cost=0.0003,
                latency=60,
            ),
            60,
            0.30,
            0.030,
            0.72,
            {"incident_triage": 0.60},
        ),
        P(
            _spec(
                "search_a",
                "General web search API: news, facts, public data, market information",
                ["web_facts", "scalars", "market_data"],
                cost=0.004,
                latency=300,
            ),
            300,
            0.40,
            0.020,
            0.88,
            {"breaking_news": 0.90, "stock_quote": 0.82, "market_compare": 0.80},
        ),
        P(
            _spec(
                "search_b",
                "Low-cost web search API: news, facts and public information",
                ["web_facts", "scalars"],
                cost=0.002,
                latency=200,
            ),
            180,
            0.35,
            0.030,
            0.86,
            {"breaking_news": 0.72},
        ),
        P(
            _spec(
                "search_c",
                "Premium web search with publisher-verified results and market information",
                ["web_facts", "scalars", "market_data"],
                cost=0.008,
                latency=450,
            ),
            450,
            0.30,
            0.005,
            0.94,
            {"stock_quote": 0.86, "market_compare": 0.85},
        ),
        P(
            _spec(
                "fin_api",
                "Licensed financial market data API: quotes, prices, market caps",
                ["market_data", "table", "scalars"],
                cost=0.010,
                latency=250,
                roles=("finance", "admin"),
                sensitive=True,
            ),
            250,
            0.30,
            0.020,
            0.98,
        ),
        P(
            _spec(
                "python_sandbox",
                "Sandboxed Python for statistics, regressions, cohorts over tabular data",
                ["numeric_result"],
                ["table"],
                cost=0.003,
                latency=450,
                timeout=2500,
            ),
            450,
            0.25,
            0.020,
            0.985,
        ),
        P(
            _spec(
                "calculator",
                "Arithmetic calculator for percentages, ratios and simple formulas on numbers",
                ["numeric_result"],
                ["scalars"],
                cost=0.0001,
                latency=10,
            ),
            8,
            0.20,
            0.001,
            0.97,
            {"cohort_analysis": 0.25, "market_compare": 0.50},
        ),
        P(
            _spec(
                "log_search", "Search over recent application logs, errors and stack traces", ["logs"], cost=0.002, latency=200
            ),
            200,
            0.30,
            0.020,
            0.92,
        ),
        P(
            _spec("log_archive", "Cold-storage log archive query (slow, cheap)", ["logs"], cost=0.0005, latency=900),
            800,
            0.30,
            0.010,
            0.85,
        ),
    ]


def drift_schedule() -> list[Drift]:
    """Non-stationary scenario used in experiment E2 (stream of 3000 tasks)."""
    return [
        Drift(1000, 2000, "search_b", latency_mult=3.0, failure=0.50),  # overload, then recovery
        Drift(1000, 3000, "sql_replica", latency_mult=4.0, quality_mult=0.6),  # replication lag
        Drift(1800, 2100, "vector_docs", failure=0.95),  # outage
        Drift(2000, 3000, "python_sandbox", latency_mult=2.5),  # noisy neighbour
        Drift(2200, 3000, "search_a", latency_mult=0.5),  # upgrade: becomes faster
    ]


def _key(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


class SimEnvironment:
    def __init__(self, profiles: list[Profile], drifts: list[Drift] = (), seed: int = 0) -> None:
        self.base = {p.spec.name: p for p in profiles}
        self.drifts = list(drifts)
        self.seed = seed

    @property
    def specs(self) -> list[ToolSpec]:
        return [p.spec for p in self.base.values()]

    def profile(self, tool: str, t: int) -> Profile:
        p = self.base[tool]
        for d in self.drifts:
            if d.tool == tool and d.start <= t < d.end:
                p = replace(
                    p,
                    latency_ms=p.latency_ms * d.latency_mult,
                    failure=p.failure if d.failure is None else d.failure,
                    quality=p.quality * d.quality_mult,
                    quality_by_family={k: v * d.quality_mult for k, v in p.quality_by_family.items()},
                )
        return p

    def _rng(self, *parts) -> np.random.Generator:
        return np.random.default_rng([self.seed, _key(*parts)])

    def call(self, task: Task, t: int, spec: ToolSpec, inputs: Mapping[str, ToolResult], attempt: int) -> ToolResult:
        p = self.profile(spec.name, t)
        ex = self._rng("exec", task.id, spec.name, attempt)
        latency = float(math.exp(ex.normal(math.log(p.latency_ms), p.sigma)))
        if ex.random() < p.failure:
            # Failures surface either as errors (fast) or as hangs that time out.
            if ex.random() < 0.5:
                return ToolResult(spec.name, False, latency * 0.3, failure=FailureKind.ERROR)
            return ToolResult(spec.name, False, spec.timeout_ms, failure=FailureKind.TIMEOUT)
        # Correctness is a property of (task, tool), independent of retries.
        qr = self._rng("quality", task.id, spec.name)
        own = qr.random() < p.q(task.family)
        correct = own and all(r.meta.get("correct", False) for r in inputs.values())
        conf = float(qr.beta(6, 2) if correct else qr.beta(2, 4))
        return ToolResult(spec.name, True, latency, output=f"<{spec.name}:{task.id}>", confidence=conf, meta={"correct": correct})

    def grade(self, task: Task, supplied: Mapping[str, ToolResult]) -> bool:
        if not task.required:
            return bool(self._rng("direct", task.id).random() < DIRECT_ANSWER_QUALITY)
        return all(t in supplied and supplied[t].meta.get("correct", False) for t in task.required)


def distractor_profiles(n: int, seed: int = 0) -> list[Profile]:
    """Synthetic extra tools for the registry-scaling experiment.

    Mixes near-duplicates of real tools (similar descriptions, worse true
    quality or reliability) with unrelated tools, so that larger registries
    contain more plausible-but-wrong candidates.
    """
    rng = np.random.default_rng(seed)
    base = default_profiles()
    nouns = ["CRM", "HR", "inventory", "weather", "maps", "translation", "email", "calendar", "ticketing", "IoT"]
    out = []
    for i in range(n):
        if rng.random() < 0.5:
            b = base[int(rng.integers(len(base)))]
            spec = replace(
                b.spec,
                name=f"{b.spec.name}_v{i}",
                description=b.spec.description + f" (mirror {i})",
                cost=b.spec.cost * float(rng.uniform(0.3, 1.5)),
            )
            out.append(
                Profile(
                    spec,
                    b.latency_ms * float(rng.uniform(0.8, 3.0)),
                    b.sigma,
                    min(0.5, b.failure + float(rng.uniform(0.0, 0.2))),
                    b.quality * float(rng.uniform(0.4, 0.95)),
                    {k: v * 0.8 for k, v in b.quality_by_family.items()},
                )
            )
        else:
            noun = nouns[int(rng.integers(len(nouns)))]
            spec = _spec(
                f"{noun.lower()}_tool_{i}",
                f"{noun} system API for {noun} records",
                [f"{noun.lower()}_records"],
                cost=float(rng.uniform(0.0005, 0.01)),
                latency=200,
            )
            out.append(Profile(spec, 200, 0.3, 0.02, 0.9))
    return out

"""Wiring of the production service: registry of real tools, router, executor."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.core.registry import ToolRegistry
from atlas.core.types import Decision, ExecutionReport, RequestContext, ToolResult, ToolSpec
from atlas.execution.circuit import CircuitBreakers
from atlas.execution.executor import Executor
from atlas.persistence import StateStore, dump_state, load_state
from atlas.router import ATLASConfig, ATLASRouter
from atlas.tools.base import ToolAdapter
from atlas.tools.compute import CalculatorTool, PythonSandboxTool
from atlas.tools.search import BM25CorpusTool
from atlas.tools.sql import SQLiteReadOnlyTool, create_demo_db
from atlas.understanding.classifier import TaskUnderstanding


def _spec(name, desc, provides, requires=(), cost=0.0, latency=100.0, roles=(), timeout=2000.0, sensitive=False):
    return ToolSpec(
        name,
        desc,
        frozenset(provides),
        frozenset(requires),
        cost,
        timeout,
        frozenset(roles),
        sensitive,
        # Optimistic prior for vetted first-party tools; refined by /v1/feedback.
        prior_latency_ms=latency,
        prior_failure=0.05,
        prior_quality=0.92,
    )


def build_tools(data_dir: Path) -> list[ToolAdapter]:
    db = data_dir / "warehouse.db"
    if not db.exists():
        create_demo_db(db)
    return [
        SQLiteReadOnlyTool(
            _spec(
                "sql_warehouse",
                "Read-only SQL over the metrics warehouse",
                ["internal_metrics", "table", "scalars"],
                cost=0.002,
                latency=20,
            ),
            db,
        ),
        BM25CorpusTool(
            _spec("docs_search", "Search internal policies, runbooks and memos", ["internal_docs"], cost=0.001, latency=20),
            data_dir / "docs",
        ),
        BM25CorpusTool(
            _spec("web_search_offline", "Offline web/news corpus search", ["web_facts"], cost=0.004, latency=30), data_dir / "web"
        ),
        BM25CorpusTool(
            _spec(
                "log_search",
                "Search recent application logs",
                ["logs"],
                cost=0.002,
                latency=20,
                roles=("sre", "admin", "analyst"),
            ),
            data_dir / "logs",
            k=5,
        ),
        CalculatorTool(
            _spec("calculator", "Arithmetic on upstream scalar values", ["numeric_result"], ["scalars"], cost=0.0001, latency=5)
        ),
        PythonSandboxTool(
            _spec(
                "python_sandbox",
                "Sandboxed Python statistics over upstream tables",
                ["numeric_result"],
                ["table"],
                cost=0.003,
                latency=400,
                timeout=5000,
            )
        ),
    ]


@dataclass
class _Pending:
    decision: Decision
    report: ExecutionReport
    created: float = field(default_factory=time.time)


class AtlasService:
    def __init__(
        self,
        adapters: list[ToolAdapter],
        understanding: TaskUnderstanding,
        store: StateStore | None = None,
        config: ATLASConfig | None = None,
        seed: int = 0,
        max_pending: int = 10_000,
    ) -> None:
        self.adapters = {a.spec.name: a for a in adapters}
        self.registry = ToolRegistry(a.spec for a in adapters)
        self.breakers = CircuitBreakers(clock=time.monotonic, cooldown=30.0)
        self.router = ATLASRouter(self.registry, understanding, config, self.breakers, seed)
        self.store = store
        self.lock = asyncio.Lock()
        self.pending: OrderedDict[str, _Pending] = OrderedDict()
        self.max_pending = max_pending
        if store is not None and (state := store.load()):
            load_state(self.router, state)

    async def route(self, query: str, ctx: RequestContext) -> Decision:
        async with self.lock:  # router state is not thread-safe; routing is ~ms of CPU
            return self.router.route(query, ctx)

    async def execute(self, decision: Decision, args: Mapping[str, Mapping[str, Any]]) -> ExecutionReport:
        async def invoke(spec: ToolSpec, inputs: Mapping[str, ToolResult], attempt: int) -> ToolResult:
            return await self.adapters[spec.name](decision.query, args.get(spec.name, {}), inputs)

        ex = Executor({n: a.spec for n, a in self.adapters.items()}, invoke, self.breakers, real_time=True)
        report = await ex.execute(decision.plan, decision.context, self.router.hooks(decision))
        async with self.lock:
            # Reliability/latency are learned now; answer quality waits for feedback.
            self.router.observe(decision, report, None)
            self.pending[decision.context.request_id] = _Pending(decision, report)
            while len(self.pending) > self.max_pending:
                self.pending.popitem(last=False)
        return report

    async def feedback(self, request_id: str, success: bool) -> bool:
        async with self.lock:
            item = self.pending.pop(request_id, None)
            if item is None:
                return False
            # Execution statistics were already learned in execute(); only the
            # answer-quality signal is new, so replay with tool results stripped.
            stripped = ExecutionReport([], item.report.steps, item.report.supplied, item.report.latency_ms, item.report.cost)
            self.router.observe(item.decision, stripped, success)
            return True

    def snapshot(self) -> None:
        if self.store is not None:
            self.store.save(dump_state(self.router))

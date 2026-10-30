"""Dependency-aware asynchronous plan executor.

Steps in the same topological level run concurrently. Each step gets per-tool
timeouts, bounded retries with exponential backoff and full jitter, a circuit
breaker check and a policy (RBAC) check. When a step finally fails, the router
may supply a fallback step for the same types (replanning); after the plan
completes the router may request verification steps (stopping rule).

Latency is accounted as the critical path through the executed DAG. For
simulated tools durations are virtual; for real tools they are measured wall
time and the steps genuinely run in parallel.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from atlas.core.types import (
    ExecutionReport,
    FailureKind,
    Plan,
    PlanStep,
    RequestContext,
    ToolResult,
    ToolSpec,
)
from atlas.execution.cache import serve, source_of, virtual_spec
from atlas.execution.circuit import CircuitBreakers

Invoke = Callable[[ToolSpec, Mapping[str, ToolResult], int], Awaitable[ToolResult]]
RETRYABLE = {FailureKind.TIMEOUT, FailureKind.ERROR, FailureKind.RATE_LIMITED}


class Hooks(Protocol):
    def fallback(self, failed: PlanStep, results: list[ToolResult], exclude: set[str]) -> PlanStep | None: ...

    def followup(self, report: ExecutionReport) -> list[PlanStep]: ...


class NoHooks:
    def fallback(self, failed: PlanStep, results: list[ToolResult], exclude: set[str]) -> PlanStep | None:
        return None

    def followup(self, report: ExecutionReport) -> list[PlanStep]:
        return []


@dataclass
class RetryPolicy:
    max_retries: int = 1
    base_ms: float = 50.0
    cap_ms: float = 1000.0

    def backoff_ms(self, attempt: int, rng: random.Random) -> float:
        return rng.uniform(0, min(self.cap_ms, self.base_ms * 2**attempt))


@dataclass
class Executor:
    tools: Mapping[str, ToolSpec]
    invoke: Invoke
    breakers: CircuitBreakers
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    real_time: bool = False
    """Sleep for real during backoff and enforce timeouts with wall time."""
    on_result: Callable[[ToolResult], None] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    async def _attempt(self, spec: ToolSpec, inputs: Mapping[str, ToolResult], attempt: int) -> ToolResult:
        start = time.perf_counter()
        try:
            if self.real_time:
                res = await asyncio.wait_for(self.invoke(spec, inputs, attempt), spec.timeout_ms / 1000)
            else:
                res = await self.invoke(spec, inputs, attempt)
        except TimeoutError:
            return ToolResult(spec.name, False, spec.timeout_ms, failure=FailureKind.TIMEOUT, attempt=attempt)
        except Exception as e:  # tool adapters must never crash the executor
            dur = (time.perf_counter() - start) * 1000
            return ToolResult(spec.name, False, dur, failure=FailureKind.ERROR, error=repr(e), attempt=attempt)
        if res.ok and res.duration_ms > spec.timeout_ms:
            res = ToolResult(spec.name, False, spec.timeout_ms, failure=FailureKind.TIMEOUT, attempt=attempt, meta=res.meta)
        res.attempt = attempt
        res.meta.setdefault("inputs", list(inputs.values()))
        return res

    async def _run_tool(
        self, step: PlanStep, inputs: Mapping[str, ToolResult], ctx: RequestContext
    ) -> tuple[list[ToolResult], float]:
        """Run one step with retries. Returns all attempt results and elapsed ms."""
        spec = self._spec(step.tool)
        if not spec.allowed_for(ctx.role):
            return [ToolResult(spec.name, False, 0.0, failure=FailureKind.POLICY_DENIED)], 0.0
        if step.cached is not None:  # planned cache hit: no call, no breaker update
            res = serve(step.tool, step.cached)
            if self.on_result:
                self.on_result(res)
            return [res], res.duration_ms
        out: list[ToolResult] = []
        elapsed = 0.0
        for attempt in range(self.retry.max_retries + 1):
            if not self.breakers.allow(spec.name):
                out.append(ToolResult(spec.name, False, 0.0, failure=FailureKind.CIRCUIT_OPEN, attempt=attempt))
                break
            res = await self._attempt(spec, inputs, attempt)
            self.breakers.record(spec.name, res.ok)
            if self.on_result:
                self.on_result(res)
            out.append(res)
            elapsed += res.duration_ms
            if res.ok or res.failure not in RETRYABLE or attempt == self.retry.max_retries:
                break
            pause = self.retry.backoff_ms(attempt, self._rng)
            elapsed += pause
            if self.real_time:
                await asyncio.sleep(pause / 1000)
        return out, elapsed

    async def execute(self, plan: Plan, ctx: RequestContext, hooks: Hooks | None = None) -> ExecutionReport:
        hooks = hooks or NoHooks()
        steps = list(plan.steps)
        results: list[ToolResult] = []
        # type -> (successful result, virtual time at which it became available)
        avail: dict[str, list[tuple[ToolResult, float]]] = {}
        ends: list[float] = []

        def inputs_for(spec: ToolSpec) -> tuple[dict[str, ToolResult], float] | None:
            got, ready = {}, 0.0
            for r in spec.requires:
                if r not in avail:
                    return None
                res, t = max(avail[r], key=lambda x: (x[0].confidence or 0.0, -x[1]))
                got[r], ready = res, max(ready, t)
            return got, ready

        async def run_branch(step: PlanStep) -> None:
            """Run a step, then fallbacks for the same types until one succeeds."""
            tried: set[str] = set()
            current: PlanStep | None = step
            start = 0.0
            while current is not None:
                spec = self._spec(current.tool)
                tried.add(spec.name)
                if current is not step:
                    steps.append(current)
                inp = inputs_for(spec)
                if inp is None:
                    results.append(ToolResult(spec.name, False, 0.0, failure=FailureKind.DEPENDENCY_FAILED))
                    return
                inputs, ready = inp
                start = max(start, ready)
                attempts, elapsed = await self._run_tool(current, inputs, ctx)
                results.extend(attempts)
                start += elapsed
                ends.append(start)
                last = attempts[-1]
                if last.ok:
                    for t in spec.provides:
                        avail.setdefault(t, []).append((last, start))
                    return
                current = hooks.fallback(current, results, tried)

        for level in plan.levels():
            await asyncio.gather(*(run_branch(plan.steps[i]) for i in level))

        report = self._report(results, steps, avail, ends)
        extra = hooks.followup(report)
        if extra:
            base = report.latency_ms
            await asyncio.gather(*(self._run_followup(s, ctx, base, results, steps, avail, ends) for s in extra))
            report = self._report(results, steps, avail, ends)
        return report

    async def _run_followup(self, step, ctx, base, results, steps, avail, ends) -> None:
        spec = self._spec(step.tool)
        steps.append(step)
        inputs = {}
        for r in spec.requires:
            if r not in avail:
                results.append(ToolResult(spec.name, False, 0.0, failure=FailureKind.DEPENDENCY_FAILED))
                return
            inputs[r] = max(avail[r], key=lambda x: x[0].confidence or 0.0)[0]
        attempts, elapsed = await self._run_tool(step, inputs, ctx)
        results.extend(attempts)
        ends.append(base + elapsed)
        if attempts[-1].ok:
            for t in spec.provides:
                avail.setdefault(t, []).append((attempts[-1], base + elapsed))

    def _spec(self, name: str) -> ToolSpec:
        src = source_of(name)
        return self.tools[name] if src is None else virtual_spec(self.tools[src])

    def _report(self, results, steps, avail, ends) -> ExecutionReport:
        supplied = {t: max(v, key=lambda x: (x[0].confidence or 0.0, -x[1]))[0] for t, v in avail.items()}
        cost = sum(
            self._spec(r.tool).cost
            for r in results
            if r.failure not in (FailureKind.CIRCUIT_OPEN, FailureKind.POLICY_DENIED, FailureKind.DEPENDENCY_FAILED)
        )
        return ExecutionReport(
            results=list(results), steps=list(steps), supplied=supplied, latency_ms=max(ends, default=0.0), cost=cost
        )

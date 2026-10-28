"""Runs a router over a task stream in the simulator and records per-task metrics."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

from atlas.core.registry import ToolRegistry
from atlas.core.types import Decision, ExecutionReport, FailureKind, RequestContext
from atlas.execution.circuit import CircuitBreakers
from atlas.execution.executor import Executor
from atlas.router import ATLASConfig, ATLASRouter, static_router
from atlas.understanding.classifier import TaskUnderstanding
from benchmarks.baselines.routers import (
    ContextualBanditRouter,
    OracleRouter,
    RetrievalRouter,
    RuleRouter,
    SingleToolRouter,
    SupervisedRouter,
)
from benchmarks.simulator.env import Profile, SimEnvironment, default_profiles
from benchmarks.simulator.tasks import Task, generate

BASE_CTX = RequestContext()
WARMUP_TASKS = 300
TRAIN_TASKS = 2000


@dataclass
class Record:
    router: str
    seed: int
    t: int
    family: str
    role: str
    success: bool
    calls: int
    unnecessary: int
    verification: int
    fallback: int
    denied: int
    tool_failures: int
    latency_ms: float
    cost: float
    planning_ms: float
    est_success: float
    plan_feasible: bool
    tools: str


def unnecessary_calls(required: frozenset[str], report: ExecutionReport, registry: ToolRegistry) -> int:
    """Strict count: a call is necessary only if it could supply a still-missing needed type.

    ``needed`` is the task's true requirement closure through the tools that
    were actually called. Retries of the same tool are not counted again;
    verification calls for an already-supplied type count as unnecessary.
    """
    called = [
        r
        for r in report.results
        if r.failure not in (FailureKind.CIRCUIT_OPEN, FailureKind.POLICY_DENIED, FailureKind.DEPENDENCY_FAILED)
    ]
    names = {r.tool for r in called}
    need = set(required)
    changed = True
    while changed:
        changed = False
        for n in names:
            spec = registry.get(n)
            if spec.provides & need and not spec.requires <= need:
                need |= spec.requires
                changed = True
    covered: set[str] = set()
    seen: set[str] = set()
    count = 0
    for r in called:
        if r.tool in seen and r.attempt > 0:
            continue
        seen.add(r.tool)
        useful = registry.get(r.tool).provides & (need - covered)
        if not useful:
            count += 1
        elif r.ok:
            covered |= useful
    return count


@lru_cache(maxsize=4)
def train_understanding(seed: int = 0) -> TaskUnderstanding:
    tasks = generate(TRAIN_TASKS, seed, "train")
    return TaskUnderstanding().fit([t.query for t in tasks], [t.required for t in tasks], [t.family for t in tasks])


def make_router(
    name: str, registry: ToolRegistry, env: SimEnvironment, seed: int, breakers: CircuitBreakers, cfg: ATLASConfig | None = None
):
    u = train_understanding()
    if name == "B0":
        return SingleToolRouter(registry, u)
    if name == "B1":
        return RuleRouter(registry)
    if name == "B2":
        return _supervised(registry, env)
    if name == "B3":
        return RetrievalRouter(registry)
    if name == "B4":
        return static_router(registry, u, None, seed)
    if name == "B5":
        return ContextualBanditRouter(registry, u, seed)
    if name == "Oracle":
        return OracleRouter(registry, env, seed=seed)
    return ATLASRouter(registry, u, cfg or ATLASConfig(), breakers, seed, name=name)


def _supervised(registry: ToolRegistry, env: SimEnvironment) -> SupervisedRouter:
    oracle = OracleRouter(registry, SimEnvironment(list(env.base.values()), [], seed=env.seed))
    tasks = generate(TRAIN_TASKS, 0, "train")
    labels = [oracle.best_tools(t, 0, replace(BASE_CTX, role=t.role)) for t in tasks]
    return SupervisedRouter(registry, [t.query for t in tasks], [t.role for t in tasks], labels)


async def _run(
    router,
    env: SimEnvironment,
    tasks: list[Task],
    registry: ToolRegistry,
    breakers: CircuitBreakers,
    seed: int,
    ctx: RequestContext,
    clock: dict,
    log: list | None = None,
) -> list[Record]:
    holder: dict = {}

    async def invoke(spec, inputs, attempt):
        return env.call(holder["task"], holder["t"], spec, inputs, attempt)

    ex = Executor({s.name: s for s in registry}, invoke, breakers, seed=seed)
    out: list[Record] = []
    for task in tasks:
        t = task.id
        holder["task"], holder["t"] = task, t
        clock["t"] = t
        if isinstance(router, OracleRouter):
            router.set_task(task, t)
        c = replace(ctx, role=task.role, request_id=f"{seed}-{t}")
        d: Decision = router.route(task.query, c)
        rep = await ex.execute(d.plan, c, router.hooks(d))
        ok = env.grade(task, rep.supplied)
        router.observe(d, rep, ok)
        if log is not None:
            log.append((d, rep, ok))
        purposes = [s.purpose for s in rep.steps]
        out.append(
            Record(
                router=getattr(router, "name", type(router).__name__),
                seed=seed,
                t=t,
                family=task.family,
                role=task.role,
                success=ok,
                calls=rep.calls
                - sum(r.failure is FailureKind.POLICY_DENIED or r.failure is FailureKind.DEPENDENCY_FAILED for r in rep.results),
                unnecessary=unnecessary_calls(task.required, rep, registry),
                verification=purposes.count("verification"),
                fallback=purposes.count("fallback"),
                denied=sum(r.failure is FailureKind.POLICY_DENIED for r in rep.results),
                tool_failures=sum((not r.ok) and r.failure in (FailureKind.TIMEOUT, FailureKind.ERROR) for r in rep.results),
                latency_ms=rep.latency_ms,
                cost=rep.cost,
                planning_ms=d.planning_ms,
                est_success=d.plan.est_success,
                plan_feasible=d.plan.feasible,
                tools="+".join(sorted(set(d.plan.tools))),
            )
        )
    return out


def warmup_logs(registry: ToolRegistry, profiles: list[Profile], seed: int, n: int = WARMUP_TASKS):
    """Offline logs from a uniform-random feasible-plan policy on stationary tools."""
    env = SimEnvironment(profiles, [], seed=10_000 + seed)
    clock = {"t": 0}
    breakers = CircuitBreakers(clock=lambda: clock["t"], cooldown=30)
    logger = ATLASRouter(
        registry, train_understanding(), ATLASConfig(uniform_policy=True, stopping=False), breakers, seed, name="logger"
    )
    tasks = [replace(t, id=t.id + 1_000_000) for t in generate(n, seed, "train")]
    log: list = []
    asyncio.run(_run(logger, env, tasks, registry, breakers, seed, BASE_CTX, clock, log=log))
    return log


def run(
    router_name: str,
    seed: int,
    n: int,
    drifts=(),
    profiles: list[Profile] | None = None,
    cfg: ATLASConfig | None = None,
    ctx: RequestContext = BASE_CTX,
    warm: bool = True,
    label: str | None = None,
) -> list[Record]:
    profiles = profiles or default_profiles()
    registry = ToolRegistry(p.spec for p in profiles)
    env = SimEnvironment(profiles, list(drifts), seed=seed)
    clock = {"t": 0}
    breakers = CircuitBreakers(clock=lambda: clock["t"], cooldown=30)
    router = make_router(router_name, registry, env, seed, breakers, cfg)
    if label:
        router.name = label
    if warm and hasattr(router, "warm_start"):
        for d, rep, ok in _cached_logs(seed, tuple(p.spec.name for p in profiles), profiles):
            router.warm_start(d, rep, ok)
        if hasattr(router, "freeze") and router_name == "B4":
            router.freeze()
        if hasattr(router, "now"):
            router.now = 0
    # Breakers use the logical clock (task index) so cooldowns are reproducible.
    return asyncio.run(_run(router, env, generate(n, seed, "test"), registry, breakers, seed, ctx, clock))


_LOG_CACHE: dict = {}


def _cached_logs(seed, key, profiles):
    k = (seed, key)
    if k not in _LOG_CACHE:
        _LOG_CACHE.clear()
        _LOG_CACHE[k] = warmup_logs(ToolRegistry(p.spec for p in profiles), profiles, seed)
    return _LOG_CACHE[k]


def as_dicts(records: list[Record]) -> list[dict]:
    return [asdict(r) for r in records]

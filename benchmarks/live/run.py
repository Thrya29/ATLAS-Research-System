"""Live evaluation: routers drive real tool executions on a generated world.

    python -m benchmarks.live.run --seeds 1-5 --tasks 400

What is real: SQLite queries, BM25/hybrid retrieval, a regex log scan and a
subprocess Python sandbox, their outputs, their errors and their execution
time. Answers are graded against the generator's ground truth.

What is injected (documented, seeded and identical for every router): network
latency in front of the "remote" tools, transient failures, and a drift
schedule (an outage of the primary database, a noisy neighbour slowing the
sandbox, a degraded search cluster). Local tools would otherwise answer in
microseconds and never fail, which is not what deployed tools do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from atlas.core.registry import ToolRegistry
from atlas.core.types import FailureKind, RequestContext, ToolResult, ToolSpec
from atlas.execution.circuit import CircuitBreakers
from atlas.execution.executor import Executor
from atlas.retrieval.hybrid import HybridRetriever
from atlas.router import ATLASConfig, ATLASRouter, static_router
from atlas.tools.base import ToolAdapter, ToolError
from atlas.tools.compute import CalculatorTool, PythonSandboxTool
from atlas.tools.search import BM25CorpusTool
from atlas.tools.sql import SQLiteReadOnlyTool
from atlas.understanding.classifier import TaskUnderstanding
from benchmarks.baselines.routers import ContextualBanditRouter, _Base
from benchmarks.harness import LOGGER_CONFIG
from benchmarks.live import world as W
from benchmarks.stats import bootstrap_ci, holm, paired

OUT = Path(__file__).resolve().parents[1] / "reports"
ROOT = Path(os.environ.get("ATLAS_LIVE_DIR", Path(__file__).resolve().parents[2] / "data" / "live"))
CTX = RequestContext(latency_budget_ms=1500)


# ----------------------------------------------------------------------- tools
class HybridCorpusTool(BM25CorpusTool):
    """BM25 + character n-gram retrieval fused with reciprocal rank fusion."""

    def __init__(self, spec: ToolSpec, corpus_dir: Path, k: int = 3) -> None:
        super().__init__(spec, corpus_dir, k)
        self.hybrid = HybridRetriever([str(i) for i in range(len(self.chunks))], [c for _, c in self.chunks])

    def run(self, query, args, inputs):
        q = str(args.get("query", query))
        fused = self.hybrid.scores(q)
        order = sorted(range(len(self.chunks)), key=lambda i: -fused[str(i)])[: self.k]
        hits = [{"source": self.chunks[i][0], "text": self.chunks[i][1], "score": round(fused[str(i)], 4)} for i in order]
        # Confident when lexical and dense retrieval agree on the top passage.
        agree = int(np.argmax(self.index.scores(q))) == order[0]
        return {"hits": hits}, 0.85 if agree else 0.45


class LogScanTool(ToolAdapter):
    """Exact field filter over the full log archive (a real linear scan)."""

    def __init__(self, spec: ToolSpec, path: Path) -> None:
        self.spec, self.path = spec, path

    def run(self, query, args, inputs):
        svc, day, level = (str(args.get(k, "")) for k in ("service", "day", "level"))
        pat = re.compile(rf"^{re.escape(day)} \S+ {re.escape(svc)} {re.escape(level)} ")
        with self.path.open(encoding="utf-8") as f:
            hits = [{"text": line.strip()} for line in f if pat.match(line)]
        if not hits:
            raise ToolError("no matching log lines")
        return {"hits": hits[::-1]}, 0.95 if len(hits) == 1 else 0.5


def _spec(name, provides, requires=(), cost=0.0, latency=50.0, timeout=2000.0):
    # The router is told the same generic prior for every tool; it must learn the rest.
    return ToolSpec(
        name,
        name.replace("_", " "),
        frozenset(provides),
        frozenset(requires),
        cost,
        timeout,
        prior_latency_ms=latency,
        prior_failure=0.05,
        prior_quality=0.9,
    )


def build_tools(w: W.World) -> list[ToolAdapter]:
    r = w.root
    return [
        SQLiteReadOnlyTool(_spec("sql_primary", ["scalars", "table"], cost=0.002, latency=120), r / "primary.db"),
        SQLiteReadOnlyTool(_spec("sql_replica", ["scalars", "table"], cost=0.0005, latency=40), r / "replica.db"),
        HybridCorpusTool(_spec("docs_hybrid", ["internal_docs"], cost=0.002, latency=80), r / "docs_current"),
        BM25CorpusTool(_spec("docs_archive", ["internal_docs"], cost=0.0005, latency=40), r / "docs_archive"),
        BM25CorpusTool(_spec("log_recent", ["logs"], cost=0.0005, latency=40), r / "logs_recent", k=3),
        LogScanTool(_spec("log_scan", ["logs"], cost=0.003, latency=300), r / "archive.log"),
        CalculatorTool(_spec("calculator", ["numeric_result"], ["scalars"], cost=0.0001, latency=5)),
        PythonSandboxTool(_spec("python_sandbox", ["numeric_result"], ["table"], cost=0.003, latency=400, timeout=5000)),
    ]


# ---------------------------------------------------------------- fault layer
@dataclass(frozen=True)
class Injected:
    latency_ms: float  # median added network latency
    failure: float  # probability of a transient error
    sigma: float = 0.35


BASE = {
    "sql_primary": Injected(120, 0.02),
    "sql_replica": Injected(40, 0.02),
    "docs_hybrid": Injected(60, 0.02),
    "docs_archive": Injected(30, 0.02),
    "log_recent": Injected(30, 0.02),
    "log_scan": Injected(250, 0.03),
    "calculator": Injected(0, 0.0),
    "python_sandbox": Injected(0, 0.01),
}


@dataclass(frozen=True)
class Drift:
    start: int
    end: int
    tool: str
    failure: float | None = None
    latency_mult: float = 1.0
    extra_ms: float = 0.0


def drift_schedule(n: int) -> list[Drift]:
    return [
        Drift(int(0.30 * n), int(0.65 * n), "sql_primary", failure=0.6),  # primary database incident
        Drift(int(0.50 * n), n, "python_sandbox", extra_ms=900),  # noisy neighbour on the sandbox host
        Drift(int(0.65 * n), n, "docs_hybrid", latency_mult=12.0),  # degraded search cluster
    ]


def injected(name: str, t: int, drifts: list[Drift]) -> tuple[Injected, float]:
    inj, extra = BASE[name], 0.0
    for d in drifts:
        if d.tool == name and d.start <= t < d.end:
            inj = Injected(inj.latency_ms * d.latency_mult, inj.failure if d.failure is None else d.failure, inj.sigma)
            extra += d.extra_ms
    return inj, extra


# --------------------------------------------------------------------- routers
class LiveRules(_Base):
    """Keyword rules written from the training templates; ``cheap`` picks the cheapest tool per need."""

    RULES = [
        (r"median|standard deviation", "stats", {"numeric_result"}),
        (r"percent|%|growth|grow|change", "growth", {"numeric_result"}),
        (r"policy|limit|cap|spend|allow|maximum", "policy", {"internal_docs"}),
        (r"error|log|fail|wrong", "incident", {"logs"}),
    ]

    def __init__(self, registry: ToolRegistry, cheap: bool) -> None:
        self.registry, self.cheap = registry, cheap
        self.name = "Rules (cheapest tools)" if cheap else "Rules (best tools)"

    def route(self, query, ctx):
        t0 = time.perf_counter()
        fam, needed = "kpi_lookup", {"scalars"}
        for pat, f, types in self.RULES:
            if re.search(pat, query, re.I):
                fam, needed = f, types
                break
        if self.cheap:
            tools = {
                "kpi_lookup": ["sql_replica"],
                "growth": ["sql_replica", "calculator"],
                "stats": ["sql_replica", "calculator"],
                "policy": ["docs_archive"],
                "incident": ["log_recent"],
            }[fam]
        else:
            tools = {
                "kpi_lookup": ["sql_primary"],
                "growth": ["sql_primary", "calculator"],
                "stats": ["sql_primary", "python_sandbox"],
                "policy": ["docs_hybrid"],
                "incident": ["log_scan"],
            }[fam]
        return self._decision(query, ctx, tools, frozenset(needed), t0, fam)


ROUTERS = ["ATLAS", "ATLAS v1", "B4 Static cost-aware", "B5 Contextual bandit", "Rules (best tools)", "Rules (cheapest tools)"]
V1 = ATLASConfig(impute_superseded=False)


def make(name, registry, u, breakers, seed):
    if name == "ATLAS":
        return ATLASRouter(registry, u, ATLASConfig(), breakers, seed)
    if name == "ATLAS v1":
        return ATLASRouter(registry, u, V1, breakers, seed, name=name)
    if name.startswith("B4"):
        return static_router(registry, u, None, seed)
    if name.startswith("B5"):
        return ContextualBanditRouter(registry, u, seed)
    return LiveRules(registry, cheap="cheapest" in name)


# ----------------------------------------------------------------------- runner
@dataclass
class LiveRecord:
    router: str
    seed: int
    t: int
    family: str
    success: bool
    calls: int
    latency_ms: float
    cost: float
    tools: str


async def run_stream(router, adapters, registry, tasks, seed, drifts, log=None, real_ms=None) -> list[LiveRecord]:
    # Breakers use the task index as their clock, so cooldowns are reproducible.
    clock = {"t": 0}
    breakers = CircuitBreakers(clock=lambda: clock["t"], cooldown=30)
    if getattr(router, "breakers", None) is not None:
        router.breakers = breakers  # routers that read breaker state see the executor's breakers
    cur: dict = {}

    async def invoke(spec: ToolSpec, inputs, attempt: int) -> ToolResult:
        task, t = cur["task"], cur["t"]
        inj, extra = injected(spec.name, t, drifts)
        rng = np.random.default_rng([seed, zlib.crc32(f"{task.id}|{spec.name}|{attempt}".encode())])
        delay = inj.latency_ms * math.exp(rng.normal(0.0, inj.sigma)) + extra if inj.latency_ms or extra else 0.0
        fail = rng.random() < inj.failure
        await asyncio.sleep(delay / 1000)
        if fail:
            return ToolResult(spec.name, False, delay, failure=FailureKind.ERROR, error="injected transient error")
        res = await adapters[spec.name](task.query, task.args.get(spec.name, {}), inputs)
        if real_ms is not None and res.ok:
            real_ms.setdefault(spec.name, []).append(res.duration_ms)
        res.duration_ms += delay
        return res

    ex = Executor({n: a.spec for n, a in adapters.items()}, invoke, breakers, real_time=True, seed=seed)
    out = []
    for t, task in enumerate(tasks):
        cur["task"], cur["t"], clock["t"] = task, t, t
        ctx = replace(CTX, request_id=f"{seed}-{t}")
        d = router.route(task.query, ctx)
        rep = await ex.execute(d.plan, ctx, router.hooks(d))
        ok = W.grade(task, rep.supplied)
        router.observe(d, rep, ok)
        if log is not None:
            log.append((d, rep, ok))
        calls = sum(r.failure not in (FailureKind.POLICY_DENIED, FailureKind.DEPENDENCY_FAILED) for r in rep.results)
        out.append(
            LiveRecord(
                getattr(router, "name", "?"),
                seed,
                t,
                task.family,
                ok,
                calls,
                rep.latency_ms,
                rep.cost,
                "+".join(sorted({r.tool for r in rep.results})),
            )
        )
    return out


def run_seed(seed: int, n: int, w: W.World, routers: list[str]) -> dict:
    train = W.tasks(w, 1500, 0, "train")
    u = TaskUnderstanding().fit([t.query for t in train], [t.required for t in train], [t.family for t in train])
    adapters = {a.spec.name: a for a in build_tools(w)}
    registry = ToolRegistry(a.spec for a in adapters.values())
    # Shared offline log: a uniform-random feasible-plan policy on stationary tools.
    log: list = []
    logger = ATLASRouter(registry, u, LOGGER_CONFIG, CircuitBreakers(clock=lambda: 0), seed, name="logger")
    warm = [replace(t, id=t.id + 1_000_000) for t in W.tasks(w, 150, seed, "train")]
    asyncio.run(run_stream(logger, adapters, registry, warm, seed + 10_000, [], log=log))
    tasks, drifts = W.tasks(w, n, seed, "test"), drift_schedule(n)
    result, real_ms = {}, {}
    for name in routers:
        router = make(name, registry, u, CircuitBreakers(clock=lambda: 0), seed)
        router.name = name
        if hasattr(router, "warm_start"):
            for d, rep, ok in log:
                router.warm_start(d, rep, ok)
            if name.startswith("B4"):
                router.freeze()
            if hasattr(router, "now"):
                router.now = 0
        t0 = time.time()
        recs = asyncio.run(run_stream(router, adapters, registry, tasks, seed, drifts, real_ms=real_ms))
        result[name] = [asdict(r) for r in recs]
        print(f"seed {seed} {name}: success {np.mean([r.success for r in recs]):.3f} in {time.time() - t0:.0f}s", flush=True)
    return {"records": result, "real_ms": real_ms}


# ----------------------------------------------------------------------- report
def summarise(runs: dict[int, dict], n: int) -> str:
    seeds = sorted(runs)
    lo, hi = int(0.30 * n), int(0.65 * n)

    def per_seed(name, f):
        return [f([r for r in runs[s]["records"][name]]) for s in seeds]

    metrics = {
        "success": lambda rs: np.mean([r["success"] for r in rs]),
        "cost/success (m$)": lambda rs: 1000 * sum(r["cost"] for r in rs) / max(1, sum(r["success"] for r in rs)),
        "outage success": lambda rs: np.mean([r["success"] for r in rs if lo <= r["t"] < hi]),
        "p95 latency (ms)": lambda rs: float(np.percentile([r["latency_ms"] for r in rs], 95)),
        "calls/task": lambda rs: np.mean([r["calls"] for r in rs]),
    }
    names = list(runs[seeds[0]]["records"])
    lines = [
        f"Seeds {seeds[0]}–{seeds[-1]} ({len(seeds)}), {n} tasks each. Mean [95% bootstrap CI over seeds].",
        "",
        "| Router | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for name in names:
        cells = []
        for f in metrics.values():
            m, a, b = bootstrap_ci(per_seed(name, f))
            cells.append(f"{m:.3f} [{a:.3f}, {b:.3f}]" if m < 10 else f"{m:.0f} [{a:.0f}, {b:.0f}]")
        lines.append(f"| {'**ATLAS**' if name == 'ATLAS' else name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Paired differences, ATLAS minus baseline (Holm-corrected Wilcoxon p over seeds):",
        "",
        "| Baseline | Δ success | Δ cost/success (m$) |",
        "|---|---|---|",
    ]
    rows, ps = [], []
    for name in names[1:]:
        ds = paired(per_seed("ATLAS", metrics["success"]), per_seed(name, metrics["success"]))
        dc = paired(per_seed("ATLAS", metrics["cost/success (m$)"]), per_seed(name, metrics["cost/success (m$)"]))
        rows.append((name, ds, dc))
        ps += [ds["p"], dc["p"]]
    adj = holm(ps)
    for i, (name, ds, dc) in enumerate(rows):
        lines.append(
            f"| {name} | {ds['diff']:+.3f} [{ds['lo']:+.3f}, {ds['hi']:+.3f}] (p={adj[2 * i]:.3f}) | "
            f"{dc['diff']:+.2f} [{dc['lo']:+.2f}, {dc['hi']:+.2f}] (p={adj[2 * i + 1]:.3f}) |"
        )
    fams = list(W.TEMPLATES)
    lines += [
        "",
        "Success by task family (mean over seeds):",
        "",
        "| Router | " + " | ".join(fams) + " |",
        "|---|" + "---|" * len(fams),
    ]
    for name in names:
        cells = [np.mean(per_seed(name, lambda rs, f=f: np.mean([r["success"] for r in rs if r["family"] == f]))) for f in fams]
        lines.append(f"| {name} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")
    real: dict[str, list] = {}
    for s in seeds:
        for k, v in runs[s]["real_ms"].items():
            real.setdefault(k, []).extend(v)
    lines += [
        "",
        "Measured real execution time per tool (excluding injected latency):",
        "",
        "| Tool | calls | median (ms) | p95 (ms) |",
        "|---|---|---|---|",
    ]
    for k in sorted(real):
        v = np.asarray(real[k])
        lines.append(f"| {k} | {len(v)} | {np.median(v):.2f} | {np.percentile(v, 95):.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--routers", default=",".join(ROUTERS))
    ap.add_argument("--report", action="store_true", help="only rebuild the report from saved runs")
    ap.add_argument("--redo", action="store_true", help="re-run --routers on existing seeds and replace their records")
    args = ap.parse_args()
    a, b = map(int, args.seeds.split("-"))
    w = W.build(ROOT)
    store = OUT / "live"
    store.mkdir(parents=True, exist_ok=True)
    if not args.report:
        for s in range(a, b + 1):
            f = store / f"seed_{s}.json"
            if args.redo and f.exists():  # e.g. a run interrupted by the machine sleeping
                old, new = json.loads(f.read_text()), run_seed(s, args.tasks, w, args.routers.split(","))
                old["records"].update(new["records"])
                f.write_text(json.dumps(old))
            elif not f.exists():  # resumable: one file per finished seed
                f.write_text(json.dumps(run_seed(s, args.tasks, w, args.routers.split(","))))
    runs = {s: json.loads((store / f"seed_{s}.json").read_text()) for s in range(a, b + 1) if (store / f"seed_{s}.json").exists()}
    (OUT / "LIVE.md").write_text(
        "# Live evaluation (real tool execution)\n\n" + summarise(runs, args.tasks) + "\n", encoding="utf-8"
    )
    print("wrote", OUT / "LIVE.md")


if __name__ == "__main__":
    main()

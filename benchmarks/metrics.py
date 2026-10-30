"""Per-run aggregate metrics from per-task records."""

from __future__ import annotations

import numpy as np

from benchmarks.harness import Record

DRIFT_START = 1000
DEGRADED = ("search_b", 1000, 2000)  # tool, start, end of its degradation window


def aggregate(recs: list[Record], budget_ms: float) -> dict[str, float]:
    n = len(recs)
    succ = np.array([r.success for r in recs], float)
    calls = np.array([r.calls for r in recs], float)
    lat = np.array([r.latency_ms for r in recs], float)
    cost = np.array([r.cost for r in recs], float)
    unnec = np.array([r.unnecessary for r in recs], float)
    s = succ.sum()
    out = {
        "success": succ.mean(),
        "calls_per_task": calls.mean(),
        "calls_per_success": calls.sum() / max(s, 1),
        "unnecessary_per_task": unnec.mean(),
        "unnecessary_rate": unnec.sum() / max(calls.sum(), 1),
        "verification_per_task": np.mean([r.verification for r in recs]),
        "fallback_per_task": np.mean([r.fallback for r in recs]),
        "cache_hits_per_task": np.mean([getattr(r, "cache_hits", 0) for r in recs]),
        "denied_per_task": np.mean([r.denied for r in recs]),
        "tool_failure_rate": sum(r.tool_failures for r in recs) / max(calls.sum(), 1),
        "cost_per_task_musd": 1000 * cost.mean(),
        "cost_per_success_musd": 1000 * cost.sum() / max(s, 1),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "latency_p99": float(np.percentile(lat, 99)),
        "budget_violation": float((lat > budget_ms).mean()),
        "planning_ms_p50": float(np.percentile([r.planning_ms for r in recs], 50)),
        "planning_ms_p95": float(np.percentile([r.planning_ms for r in recs], 95)),
    }
    planned = [r for r in recs if r.est_success > 0]
    if planned:
        out["brier"] = float(np.mean([(r.est_success - r.success) ** 2 for r in planned]))
    if n >= 3000:
        out.update(_drift_metrics(recs, succ))
    return out


def _drift_metrics(recs: list[Record], succ: np.ndarray) -> dict[str, float]:
    tool, a, b = DEGRADED
    uses = np.array([tool in r.tools.split("+") for r in recs], float)
    pre = succ[:DRIFT_START].mean()
    roll = np.convolve(succ, np.ones(100) / 100, mode="valid")  # roll[i] = mean succ[i:i+100]
    # Recovery: from drift onset, through the deepest dip in the next 500 tasks,
    # until the rolling success is back within 3 points of the pre-drift level.
    after = roll[DRIFT_START:]
    dip = int(np.argmin(after[:500]))
    hits = np.nonzero(after[dip:] >= pre - 0.03)[0]
    recovered = hits + dip
    return {
        "success_pre_drift": pre,
        "success_drift_1": succ[1000:2000].mean(),
        "success_drift_2": succ[2000:3000].mean(),
        "degraded_tool_share_during": uses[a + 100 : b].mean(),
        "degraded_tool_share_after": uses[b + 300 :].mean(),
        "recovery_tasks": float(recovered[0] + 100) if len(recovered) else float(len(succ) - DRIFT_START),
    }


def rolling(recs: list[Record], field: str = "success", w: int = 100) -> np.ndarray:
    x = np.array([getattr(r, field) for r in recs], float)
    return np.convolve(x, np.ones(w) / w, mode="valid")

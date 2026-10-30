"""HTTP API for the ATLAS routing service.

Endpoints
  GET  /healthz            liveness
  GET  /readyz             readiness (model loaded, tools registered)
  GET  /metrics            Prometheus exposition
  GET  /v1/tools           tools visible to the caller's role
  POST /v1/route           plan only (for agents that execute tools themselves)
  POST /v1/execute         plan + execute + evidence
  POST /v1/feedback        answer-quality feedback for a previous request
  GET  /v1/stats           learned per-tool estimates and circuit states (admin)
  GET  /ui                 interactive demo page (uses the endpoints above with the caller's API key)

Authentication: ``X-API-Key`` mapped to a role via ATLAS_API_KEYS="key:role,...".
Every decision is written to the audit log with its request id.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from atlas.core.types import Decision, ExecutionReport, RequestContext
from atlas.persistence import FileStore, RedisStore
from atlas.router import ATLASConfig
from atlas.service import AtlasService, build_tools
from atlas.understanding.classifier import TaskUnderstanding

audit = logging.getLogger("atlas.audit")

DECISIONS = Counter("atlas_decisions_total", "Routing decisions", ["feasible", "planned"])
TOOL_CALLS = Counter("atlas_tool_calls_total", "Tool invocations", ["tool", "outcome"])
TOOL_LATENCY = Histogram(
    "atlas_tool_latency_seconds", "Tool latency", ["tool"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)
)
ROUTING = Histogram(
    "atlas_routing_seconds", "Routing (planning) time", buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)
)
PLAN_COST = Histogram("atlas_request_cost_usd", "Tool cost per request", buckets=(0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05))
FEEDBACK = Counter("atlas_feedback_total", "Quality feedback", ["success"])


class Constraints(BaseModel):
    latency_budget_ms: float = Field(1500.0, gt=0, le=60_000)
    min_quality: float = Field(0.8, ge=0, le=1)
    max_failure: float = Field(0.05, ge=0, le=1)
    value_of_success: float = Field(0.05, ge=0, le=100)


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    constraints: Constraints = Constraints()


class ExecuteRequest(RouteRequest):
    arguments: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-tool arguments, keyed by tool name")


class FeedbackRequest(BaseModel):
    request_id: str
    success: bool


def _api_keys() -> dict[str, str]:
    raw = os.environ.get("ATLAS_API_KEYS", "dev-key:admin")
    return dict(item.split(":", 1) for item in raw.split(",") if ":" in item)


def default_understanding() -> TaskUnderstanding:
    from benchmarks.simulator.tasks import generate

    tasks = generate(2000, 0, "train")
    return TaskUnderstanding().fit([t.query for t in tasks], [t.required for t in tasks], [t.family for t in tasks])


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = Path(os.environ.get("ATLAS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    redis_url = os.environ.get("ATLAS_REDIS_URL")
    store = RedisStore(redis_url) if redis_url else FileStore(os.environ.get("ATLAS_STATE_PATH", data / "state" / "router.json"))
    app.state.keys = _api_keys()
    # Learned result caching (E7): repeated questions reuse fresh answers unless they are learned to go stale.
    app.state.svc = AtlasService(build_tools(data), default_understanding(), store, config=ATLASConfig(cache="learned"))
    yield
    app.state.svc.snapshot()


app = FastAPI(title="ATLAS tool router", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.rid = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def role_of(request: Request, x_api_key: str = Header(default="")) -> str:
    role = request.app.state.keys.get(x_api_key)
    if role is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return role


def _ctx(req: RouteRequest, role: str, rid: str) -> RequestContext:
    c = req.constraints
    return RequestContext(
        role=role,
        latency_budget_ms=c.latency_budget_ms,
        min_quality=c.min_quality,
        max_failure=c.max_failure,
        value_of_success=c.value_of_success,
        request_id=rid,
    )


def _decision_json(d: Decision) -> dict:
    p = d.plan
    return {
        "request_id": d.context.request_id,
        "family": d.family,
        "required_types": sorted(d.predicted_types),
        "plan": [
            {"tool": s.tool, "supplies": sorted(s.supplies), "depends_on": list(s.depends_on), "purpose": s.purpose}
            for s in p.steps
        ],
        "estimates": {
            "cost_usd": p.est_cost,
            "success": p.est_success,
            "quality": p.est_quality,
            "reliability": p.est_reliability,
            "p_latency_over_budget": p.est_latency_tail,
        },
        "constraints_satisfied": p.feasible,
        "candidates": d.candidates,
        "rejected": d.rejected,
        "notes": d.notes,
        "planning_ms": round(d.planning_ms, 3),
    }


def _report_json(r: ExecutionReport) -> dict:
    return {
        "latency_ms": round(r.latency_ms, 2),
        "cost_usd": r.cost,
        "calls": [
            {
                "tool": x.tool,
                "ok": x.ok,
                "attempt": x.attempt,
                "duration_ms": round(x.duration_ms, 2),
                "failure": x.failure.value if x.failure else None,
                "error": x.error,
                "confidence": x.confidence,
            }
            for x in r.results
        ],
        "steps": [{"tool": s.tool, "purpose": s.purpose, "supplies": sorted(s.supplies)} for s in r.steps],
        "evidence": {t: {"tool": x.tool, "confidence": x.confidence, "output": x.output} for t, x in r.supplied.items()},
    }


def _audit(event: str, d: Decision, role: str, **extra) -> None:
    audit.info(
        json.dumps(
            {
                "event": event,
                "request_id": d.context.request_id,
                "role": role,
                "router": d.router,
                "query_chars": len(d.query),
                "plan": d.plan.tools,
                "rejected": d.rejected,
                "feasible": d.plan.feasible,
                "ts": time.time(),
                **extra,
            }
        )
    )


UI_PAGE = Path(__file__).with_name("static") / "ui.html"


@app.get("/ui", include_in_schema=False)
async def ui() -> HTMLResponse:
    # Static page; every data call it makes is authenticated with the user's API key.
    return HTMLResponse(UI_PAGE.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict:
    svc: AtlasService | None = getattr(request.app.state, "svc", None)
    if svc is None or not len(svc.registry):
        raise HTTPException(503, "not ready")
    return {"status": "ready", "tools": len(svc.registry)}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/tools")
async def tools(request: Request, role: str = Depends(role_of)) -> list[dict]:
    return [
        {
            "name": s.name,
            "description": s.description,
            "provides": sorted(s.provides),
            "requires": sorted(s.requires),
            "cost_usd": s.cost,
        }
        for s in request.app.state.svc.registry
        if s.allowed_for(role)
    ]


@app.post("/v1/route")
async def route(req: RouteRequest, request: Request, role: str = Depends(role_of)) -> dict:
    svc: AtlasService = request.app.state.svc
    d = await svc.route(req.query, _ctx(req, role, request.state.rid))
    ROUTING.observe(d.planning_ms / 1000)
    DECISIONS.labels(str(d.plan.feasible).lower(), str(bool(d.plan.steps)).lower()).inc()
    _audit("route", d, role)
    return _decision_json(d)


@app.post("/v1/execute")
async def execute(req: ExecuteRequest, request: Request, role: str = Depends(role_of)) -> dict:
    svc: AtlasService = request.app.state.svc
    unknown = set(req.arguments) - set(svc.adapters)
    if unknown:
        raise HTTPException(422, f"arguments for unknown tools: {sorted(unknown)}")
    d = await svc.route(req.query, _ctx(req, role, request.state.rid))
    ROUTING.observe(d.planning_ms / 1000)
    DECISIONS.labels(str(d.plan.feasible).lower(), str(bool(d.plan.steps)).lower()).inc()
    rep = await svc.execute(d, req.arguments)
    for x in rep.results:
        TOOL_CALLS.labels(x.tool, "ok" if x.ok else (x.failure.value if x.failure else "error")).inc()
        TOOL_LATENCY.labels(x.tool).observe(x.duration_ms / 1000)
    PLAN_COST.observe(rep.cost)
    _audit(
        "execute", d, role, latency_ms=rep.latency_ms, cost=rep.cost, failures=[x.failure.value for x in rep.results if x.failure]
    )
    return {"decision": _decision_json(d), "execution": _report_json(rep)}


@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest, request: Request, role: str = Depends(role_of)) -> dict:
    ok = await request.app.state.svc.feedback(req.request_id, req.success)
    if not ok:
        raise HTTPException(404, "unknown or expired request_id")
    FEEDBACK.labels(str(req.success).lower()).inc()
    return {"accepted": True}


@app.get("/v1/stats")
async def stats(request: Request, role: str = Depends(role_of)) -> dict:
    if role != "admin":
        raise HTTPException(403, "admin only")
    svc: AtlasService = request.app.state.svc
    r, now = svc.router, svc.router.now
    return {
        "decisions": now,
        "circuits": svc.breakers.snapshot(),
        "tools": {
            n: {
                "reliability": round(st.reliability.mean(now), 4),
                "latency_ms": round(st.latency.mean_ms(now), 1),
                "quality": round(st.quality.mean(None, now), 4),
            }
            for n, st in r.stats.items()
        },
    }

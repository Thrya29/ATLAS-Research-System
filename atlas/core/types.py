"""Core data model shared by the router, executor, simulator and API.

Tools are described by *typed capabilities*: a tool ``provides`` a set of
evidence/artifact types and may ``require`` input types that must be supplied
by another tool earlier in the same plan. Requirements of a task are expressed
in the same type vocabulary, which turns tool-plan construction into search
over an AND-OR graph (types are OR-nodes over providers, tools are AND-nodes
over their required inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    provides: frozenset[str]
    requires: frozenset[str] = frozenset()
    cost: float = 0.0
    """Monetary cost per call (USD)."""
    timeout_ms: float = 2000.0
    roles: frozenset[str] = frozenset()
    """Roles allowed to call the tool; empty means every authenticated role."""
    sensitive: bool = False
    """Sensitive tools are never explored optimistically (safe exploration)."""
    side_effects: bool = False
    prior_latency_ms: float = 250.0
    prior_failure: float = 0.05
    prior_quality: float = 0.8

    def allowed_for(self, role: str) -> bool:
        return not self.roles or role in self.roles


class FailureKind(StrEnum):
    TIMEOUT = "timeout"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    POLICY_DENIED = "policy_denied"
    DEPENDENCY_FAILED = "dependency_failed"


@dataclass
class ToolResult:
    tool: str
    ok: bool
    duration_ms: float
    output: Any = None
    confidence: float | None = None
    failure: FailureKind | None = None
    error: str | None = None
    attempt: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    """Opaque data for the environment (e.g. hidden correctness in simulation)."""


@dataclass
class PlanStep:
    tool: str
    supplies: frozenset[str]
    """Types this step is responsible for (task requirements or inputs of later steps)."""
    depends_on: tuple[int, ...] = ()
    purpose: str = "primary"
    """primary | fallback | verification — used for accounting and audit."""
    cached: ToolResult | None = None
    """For a ``cache:<tool>`` step: the answer to serve instead of calling the tool."""


@dataclass
class Plan:
    steps: list[PlanStep]
    est_cost: float = 0.0
    est_success: float = 0.0
    est_quality: float = 0.0
    est_reliability: float = 0.0
    est_latency_tail: float = 0.0
    """Estimated P(latency > L_max)."""
    feasible: bool = True
    """Whether all constraints were satisfied when the plan was chosen."""

    @property
    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    def levels(self) -> list[list[int]]:
        """Topological levels: steps in the same level can run concurrently."""
        depth: dict[int, int] = {}
        for i, s in enumerate(self.steps):
            depth[i] = 1 + max((depth[d] for d in s.depends_on), default=-1)
        out: dict[int, list[int]] = {}
        for i, d in depth.items():
            out.setdefault(d, []).append(i)
        return [out[k] for k in sorted(out)]


@dataclass(frozen=True)
class RequestContext:
    role: str = "analyst"
    latency_budget_ms: float = 1500.0
    min_quality: float = 0.80
    max_failure: float = 0.05
    latency_tail: float = 0.10
    value_of_success: float = 0.05
    """USD value assigned to a successful answer; prices marginal tool calls."""
    request_id: str = ""


@dataclass
class Decision:
    """A router's output for one query, plus everything needed for learning/audit."""

    query: str
    context: RequestContext
    plan: Plan
    predicted_types: frozenset[str]
    family: str | None = None
    candidates: dict[str, list[str]] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    """tool -> reason it was filtered out (unauthorized, circuit open, ...)."""
    router: str = ""
    planning_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class ExecutionReport:
    results: list[ToolResult]
    steps: list[PlanStep]
    supplied: dict[str, ToolResult]
    """type -> the result chosen as evidence for that type (highest confidence)."""
    latency_ms: float
    cost: float

    @property
    def calls(self) -> int:
        return sum(1 for r in self.results if r.failure is not FailureKind.CIRCUIT_OPEN)

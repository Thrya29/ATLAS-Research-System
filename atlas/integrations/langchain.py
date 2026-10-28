"""LangChain / LangGraph integration.

Two ways to use ATLAS inside a LangChain agent:

1. **Tool narrowing** — ``AtlasToolSelector.select`` returns only the LangChain
   tools in ATLAS's plan, so the agent's LLM is bound to a minimal, authorised,
   healthy tool set (``llm.bind_tools(selector.select(query))``). The agent
   reports outcomes back with ``selector.feedback``.
2. **Planned execution** — ``atlas_node`` is a LangGraph node that routes the
   query, executes the plan (dependencies, retries, circuit breakers, fallback,
   verification) and writes the evidence into the graph state.

ATLAS needs capability metadata that LangChain tools do not carry, so each
tool is registered with a ``ToolSpec`` (what it provides/requires, cost, roles).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from langchain_core.tools import BaseTool

from atlas.core.types import Decision, ExecutionReport, RequestContext, ToolResult, ToolSpec
from atlas.service import AtlasService
from atlas.tools.base import ToolAdapter, ToolError
from atlas.understanding.classifier import TaskUnderstanding

ArgumentFn = Callable[[str, str, Mapping[str, ToolResult]], dict[str, Any]]


def _default_args(query: str, tool: str, inputs: Mapping[str, ToolResult]) -> dict[str, Any]:
    return {"query": query}


class LangChainToolAdapter(ToolAdapter):
    """Runs a LangChain ``BaseTool`` under the ATLAS executor."""

    def __init__(self, tool: BaseTool, spec: ToolSpec, arguments: ArgumentFn = _default_args, confidence: float = 0.8) -> None:
        if spec.name != tool.name:
            raise ValueError(f"spec name {spec.name!r} != tool name {tool.name!r}")
        self.tool, self.spec, self.arguments, self.confidence = tool, spec, arguments, confidence

    def run(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> tuple[Any, float]:
        payload = dict(args) or self.arguments(query, self.spec.name, inputs)
        try:
            return self.tool.invoke(payload), self.confidence
        except Exception as e:  # surface as a tool failure the router can learn from
            raise ToolError(f"{type(e).__name__}: {e}") from e


class AtlasToolSelector:
    def __init__(self, tools: Sequence[tuple[BaseTool, ToolSpec]], understanding: TaskUnderstanding, **service_kwargs) -> None:
        self.by_name = {t.name: t for t, _ in tools}
        self.service = AtlasService([LangChainToolAdapter(t, s) for t, s in tools], understanding, **service_kwargs)

    async def select(self, query: str, ctx: RequestContext | None = None) -> tuple[list[BaseTool], Decision]:
        d = await self.service.route(query, ctx or RequestContext())
        return [self.by_name[n] for n in dict.fromkeys(d.plan.tools)], d

    async def execute(self, decision: Decision, arguments: Mapping[str, Mapping[str, Any]] | None = None) -> ExecutionReport:
        return await self.service.execute(decision, arguments or {})

    async def feedback(self, request_id: str, success: bool) -> bool:
        return await self.service.feedback(request_id, success)


class AtlasState(TypedDict, total=False):
    query: str
    role: str
    request_id: str
    plan: list[str]
    evidence: dict[str, Any]
    notes: list[str]


def atlas_node(selector: AtlasToolSelector, ctx_factory: Callable[[AtlasState], RequestContext] | None = None):
    """Build an async LangGraph node: ``graph.add_node("atlas", atlas_node(selector))``."""

    async def node(state: AtlasState) -> AtlasState:
        ctx = (
            ctx_factory(state)
            if ctx_factory
            else RequestContext(role=state.get("role", "analyst"), request_id=state.get("request_id", ""))
        )
        _, decision = await selector.select(state["query"], ctx)
        report = await selector.execute(decision)
        return {
            "plan": decision.plan.tools,
            "evidence": {t: r.output for t, r in report.supplied.items()},
            "notes": decision.notes,
        }

    return node

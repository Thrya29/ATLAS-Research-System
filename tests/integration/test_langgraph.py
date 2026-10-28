import asyncio

import pytest

pytest.importorskip("langgraph")

from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from atlas.core.types import RequestContext, ToolSpec  # noqa: E402
from atlas.integrations.langchain import AtlasState, AtlasToolSelector, atlas_node  # noqa: E402
from benchmarks.harness import train_understanding  # noqa: E402


@tool
def policy_search(query: str) -> str:
    """Search company policy documents."""
    return "Hotel reimbursement is capped at 180 USD per night."


@tool
def news_search(query: str) -> str:
    """Search recent news."""
    return "Port strike ended after a tentative deal."


@tool
def broken_news(query: str) -> str:
    """A news provider that is down."""
    raise ConnectionError("provider unavailable")


def _spec(name, provides, cost):
    return ToolSpec(name, name, frozenset(provides), cost=cost, prior_latency_ms=10, prior_failure=0.05, prior_quality=0.92)


@pytest.fixture(scope="module")
def selector():
    return AtlasToolSelector(
        [
            (policy_search, _spec("policy_search", ["internal_docs"], 0.001)),
            (news_search, _spec("news_search", ["web_facts"], 0.004)),
            (broken_news, _spec("broken_news", ["web_facts"], 0.001)),
        ],
        train_understanding(),
    )


def test_selector_narrows_tools(selector):
    tools, d = asyncio.run(selector.select("What does our travel policy say about reimbursement limits?"))
    assert [t.name for t in tools] == ["policy_search"]


def test_langgraph_node_executes_with_fallback(selector):
    g = StateGraph(AtlasState)
    g.add_node("atlas", atlas_node(selector))
    g.add_edge(START, "atlas")
    g.add_edge("atlas", END)
    app = g.compile()
    out = asyncio.run(app.ainvoke({"query": "What is the latest news about the port strike?", "role": "analyst"}))
    # The cheaper provider is planned first; when it fails, ATLAS falls back to the healthy one.
    assert "web_facts" in out["evidence"] and "tentative deal" in out["evidence"]["web_facts"]


def test_feedback_roundtrip(selector):
    _, d = asyncio.run(selector.select("Summarise today's headlines on the port strike", RequestContext(request_id="r-1")))
    asyncio.run(selector.execute(d))
    assert asyncio.run(selector.feedback("r-1", True))

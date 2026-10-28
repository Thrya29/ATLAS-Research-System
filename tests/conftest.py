from __future__ import annotations

import pytest

from atlas.core.registry import ToolRegistry
from atlas.core.types import ToolSpec


def spec(name, provides, requires=(), cost=1.0, roles=(), sensitive=False, latency=100.0, timeout=2000.0):
    return ToolSpec(
        name,
        f"{name} tool",
        frozenset(provides),
        frozenset(requires),
        cost,
        timeout,
        frozenset(roles),
        sensitive,
        prior_latency_ms=latency,
    )


@pytest.fixture
def small_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            spec("sql", ["metrics", "table", "scalars"], cost=2.0),
            spec("sql_cheap", ["metrics", "table", "scalars"], cost=0.5),
            spec("python", ["number"], ["table"], cost=3.0),
            spec("calc", ["number"], ["scalars"], cost=0.1),
            spec("docs", ["docs"], cost=1.0),
            spec("fin", ["market", "table"], cost=10.0, roles=["finance"], sensitive=True),
        ]
    )

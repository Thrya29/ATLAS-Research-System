"""Result cache whose hits are planned like tools.

A fresh cached answer for (tool, query) becomes a *virtual tool* named
``cache:<tool>``: same capabilities and access roles as the source tool, no
inputs, zero price and ~1 ms latency. The router decides whether to use it the
same way it decides about any tool, so it can learn per task family whether
cached answers are still correct (stale stock quotes are not; a policy lookup is).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from atlas.core.types import ToolResult, ToolSpec

PREFIX = "cache:"
HIT_LATENCY_MS = 0.5


def virtual_name(tool: str) -> str:
    return PREFIX + tool


def source_of(name: str) -> str | None:
    return name[len(PREFIX) :] if name.startswith(PREFIX) else None


def virtual_spec(spec: ToolSpec) -> ToolSpec:
    return replace(
        spec,
        name=virtual_name(spec.name),
        description=f"cached answer from {spec.name}",
        requires=frozenset(),
        cost=0.0,
        prior_latency_ms=1.0,
        prior_failure=0.01,  # Beta prior needs a non-zero failure mass
    )


def age_bucket(age: float) -> str:
    """Log-spaced age bins, so freshness can be learned per task family."""
    for edge in (32, 128, 512):
        if age < edge:
            return f"<{edge}"
    return ">=512"


def normalise(query: str) -> str:
    return " ".join(query.lower().split())


@dataclass
class ResultCache:
    clock: Callable[[], float]
    max_age: float
    """Entries older than this (in clock units) are never served."""
    max_entries: int = 10_000
    _entries: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, tool: str, query: str) -> ToolResult | None:
        key = (tool, normalise(query))
        item = self._entries.get(key)
        if item is None or self.clock() - item[1] > self.max_age:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return replace(item[0], meta={**item[0].meta, "cache_age": self.clock() - item[1]})

    def put(self, tool: str, query: str, result: ToolResult) -> None:
        if not result.ok or result.meta.get("cached"):
            return  # only fresh, successful answers; a served hit never refreshes its own age
        key = (tool, normalise(query))
        self._entries[key] = (result, self.clock())
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def serve(name: str, cached: ToolResult) -> ToolResult:
    """The result returned for a planned cache step (upstream links are not re-attributed)."""
    meta = {k: v for k, v in cached.meta.items() if k != "inputs"}
    meta["cached"] = True
    return ToolResult(name, True, HIT_LATENCY_MS, output=cached.output, confidence=cached.confidence, meta=meta)

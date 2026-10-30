"""Tool registry with an inverted capability index."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator

from atlas.core.types import ToolSpec


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._by_type: dict[str, set[str]] = defaultdict(set)
        for t in tools:
            self.register(t)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        if tool.provides & tool.requires:
            raise ValueError(f"tool {tool.name!r} requires a type it provides")
        self._tools[tool.name] = tool
        for t in tool.provides:
            self._by_type[t].add(tool.name)

    def unregister(self, name: str) -> None:
        tool = self._tools.pop(name)
        for t in tool.provides:
            self._by_type[t].discard(name)

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def providers(self, type_: str) -> list[ToolSpec]:
        return [self._tools[n] for n in sorted(self._by_type.get(type_, ()))]

    def has_providers(self, type_: str) -> bool:
        return bool(self._by_type.get(type_))

    @property
    def types(self) -> set[str]:
        return {t for t, names in self._by_type.items() if names}

    def min_cost(self, type_: str) -> float:
        return min((t.cost for t in self.providers(type_)), default=float("inf"))

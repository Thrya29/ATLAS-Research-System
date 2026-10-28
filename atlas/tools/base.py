"""Adapter interface between real tools and the executor."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from atlas.core.types import FailureKind, ToolResult, ToolSpec


class ToolError(Exception):
    """Expected, user-facing tool failure (bad input, no data, rejected query)."""


class ToolAdapter(ABC):
    spec: ToolSpec

    @abstractmethod
    def run(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> tuple[Any, float]:
        """Synchronous tool body. Returns (output, confidence in [0, 1])."""

    async def __call__(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> ToolResult:
        start = time.perf_counter()
        try:
            # Tool bodies may block (SQLite, subprocess); keep the event loop free.
            output, conf = await asyncio.to_thread(self.run, query, args, inputs)
        except ToolError as e:
            return ToolResult(
                self.spec.name, False, (time.perf_counter() - start) * 1000, failure=FailureKind.ERROR, error=str(e)
            )
        return ToolResult(self.spec.name, True, (time.perf_counter() - start) * 1000, output=output, confidence=conf)

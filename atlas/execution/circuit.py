"""Per-tool circuit breaker (closed -> open -> half-open)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Breaker:
    state: State = State.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0


@dataclass
class CircuitBreakers:
    """``clock`` is injectable so simulations can use logical time."""

    clock: Callable[[], float]
    failure_threshold: int = 5
    cooldown: float = 30.0
    _b: dict[str, _Breaker] = field(default_factory=dict)

    def _get(self, tool: str) -> _Breaker:
        return self._b.setdefault(tool, _Breaker())

    def state(self, tool: str) -> State:
        b = self._get(tool)
        if b.state is State.OPEN and self.clock() - b.opened_at >= self.cooldown:
            b.state = State.HALF_OPEN
        return b.state

    def allow(self, tool: str) -> bool:
        return self.state(tool) is not State.OPEN

    def record(self, tool: str, ok: bool) -> None:
        b = self._get(tool)
        if ok:
            b.state, b.consecutive_failures = State.CLOSED, 0
            return
        b.consecutive_failures += 1
        if b.state is State.HALF_OPEN or b.consecutive_failures >= self.failure_threshold:
            b.state, b.opened_at = State.OPEN, self.clock()

    def snapshot(self) -> dict[str, str]:
        return {t: self.state(t).value for t in self._b}

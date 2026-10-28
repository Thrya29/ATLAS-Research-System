"""Snapshot/restore of the router's learned statistics (JSON; file or Redis)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Protocol

from atlas.bandits.estimators import DecayedBeta, DecayedLogNormal
from atlas.router import ATLASRouter

SCHEMA_VERSION = 1


def dump_state(router: ATLASRouter) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "now": router.now,
        "calibration": [asdict(b) for b in router.calibration],
        "tools": {
            name: {
                "reliability": asdict(st.reliability),
                "latency": asdict(st.latency),
                "quality_global": asdict(st.quality.global_),
                "quality_ctx": {c: asdict(b) for c, b in st.quality.ctx.items()},
            }
            for name, st in router.stats.items()
        },
    }


def _load(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_state(router: ATLASRouter, state: dict) -> None:
    if state.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported state schema {state.get('schema')}")
    router.now = int(state["now"])
    router.calibration = [_load(DecayedBeta, b) for b in state["calibration"]]
    for name, s in state["tools"].items():
        if name not in router.registry:
            continue  # tool was removed from the registry since the snapshot
        st = router._stats(router.registry.get(name))
        st.reliability = _load(DecayedBeta, s["reliability"])
        st.latency = _load(DecayedLogNormal, s["latency"])
        st.quality.global_ = _load(DecayedBeta, s["quality_global"])
        st.quality.ctx = {c: _load(DecayedBeta, b) for c, b in s["quality_ctx"].items()}


class StateStore(Protocol):
    def save(self, state: dict) -> None: ...

    def load(self) -> dict | None: ...


class FileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a crash never leaves a truncated snapshot.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, self.path)

    def load(self) -> dict | None:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None


class RedisStore:
    def __init__(self, url: str, key: str = "atlas:router_state") -> None:
        import redis

        self.client, self.key = redis.Redis.from_url(url), key

    def save(self, state: dict) -> None:
        self.client.set(self.key, json.dumps(state))

    def load(self) -> dict | None:
        raw = self.client.get(self.key)
        return json.loads(raw) if raw else None

"""E8 preparation: run the local LLM router once over every test query.

    ATLAS_LLM_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf python -m benchmarks.experiments.llm_router
    python -m benchmarks.experiments.run_all --only E8          # then route with the cached choices

The choices are committed (``reports/llm_router_cache.json``), so E8 reproduces without the model.
"""

from __future__ import annotations

import json
import os

import numpy as np

from atlas.core.registry import ToolRegistry
from benchmarks.baselines.llm_router import LATENCY, LLMToolChooser
from benchmarks.simulator.env import default_profiles
from benchmarks.simulator.tasks import generate

SEEDS = range(1, 11)


def main() -> None:
    registry = ToolRegistry(p.spec for p in default_profiles())
    chooser = LLMToolChooser(registry, os.environ["ATLAS_LLM_MODEL"])
    queries = sorted({t.query for s in SEEDS for t in generate(3000, s, "test")})
    todo = [q for q in queries if q not in chooser.cache]
    print(f"{len(queries)} unique queries, {len(todo)} to route", flush=True)
    for i, q in enumerate(todo, 1):
        chooser.choose(q)
        if i % 50 == 0:
            chooser.save()
            print(f"{i}/{len(todo)} median {np.median(chooser.latencies_ms):.0f} ms", flush=True)
    chooser.save()
    if chooser.latencies_ms:
        lat = chooser.latencies_ms
        LATENCY.write_text(
            json.dumps({"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "n": len(lat)})
        )


if __name__ == "__main__":
    main()

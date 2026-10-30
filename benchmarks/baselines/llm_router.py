"""B6: an LLM chooses the tools directly (the common "LLM-powered query router").

The model sees the tool catalogue (name, description, outputs, inputs, role
restrictions) and one worked example per task family, and answers with tool
names. Grammar-constrained decoding guarantees the answer names only real tools.
Answers are cached per query in ``reports/llm_router_cache.json`` so experiment
runs are reproducible without the model.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from pathlib import Path

from atlas.core.registry import ToolRegistry
from atlas.core.types import Decision, RequestContext, ToolSpec
from benchmarks.baselines.routers import OracleRouter, _Base
from benchmarks.simulator.env import SimEnvironment, default_profiles
from benchmarks.simulator.tasks import FAMILIES, TEMPLATES, Task, _fill

CACHE = Path(__file__).resolve().parents[1] / "reports" / "llm_router_cache.json"
LATENCY = CACHE.with_name("llm_router_latency.json")
MAX_TOOLS = 3


def _describe(s: ToolSpec) -> str:
    line = f"- {s.name}: {s.description}. Gives: {', '.join(sorted(s.provides))}."
    if s.requires:
        line += f" Needs input: {', '.join(sorted(s.requires))}."
    if s.roles:
        line += f" Only for roles: {', '.join(sorted(s.roles))}."
    return line


def examples(registry: ToolRegistry) -> list[tuple[str, list[str]]]:
    """One training-phrased query per family, labelled with the oracle's tool set (B2's supervision)."""
    from benchmarks.harness import BASE_CTX

    oracle = OracleRouter(registry, SimEnvironment(default_profiles(), [], seed=0))
    rng = random.Random("llm-router-examples")
    out = []
    for i, fam in enumerate(FAMILIES):
        q = _fill(TEMPLATES[fam]["train"][0], rng)
        tools = sorted(oracle.best_tools(Task(i, q, fam, FAMILIES[fam], "analyst"), 0, BASE_CTX))
        out.append((q, tools))
    return out


def system_prompt(registry: ToolRegistry) -> str:
    catalog = "\n".join(_describe(s) for s in sorted(registry, key=lambda s: s.name))
    shots = "\n".join(f'Question: "{q}" -> {", ".join(t) or "none"}' for q, t in examples(registry))
    return (
        "You are the tool router of an AI agent. Choose the smallest set of tools whose outputs are enough to answer "
        "the question. A tool that needs an input must be paired with a tool that gives it. Reply with tool names "
        'separated by ", ", or "none" if the question needs no tools.\n\n'
        f"Tools:\n{catalog}\n\nExamples:\n{shots}"
    )


class LLMToolChooser:
    def __init__(self, registry: ToolRegistry, model_path: str | Path, n_threads: int | None = None) -> None:
        self.registry, self.model_path, self.n_threads = registry, Path(model_path), n_threads
        self.system = system_prompt(registry)
        names = sorted(s.name for s in registry)
        tool = " | ".join(f'"{n}"' for n in names)
        alts = ['"none"'] + [' ", " '.join(["tool"] * k) for k in range(1, MAX_TOOLS + 1)]
        self.grammar_text = f"root ::= {' | '.join(alts)}\ntool ::= {tool}"
        self.cache: dict[str, list[str]] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.latencies_ms: list[float] = []
        self._llm = self._grammar = None

    def choose(self, query: str) -> list[str]:
        if query in self.cache:
            return self.cache[query]
        if self._llm is None:
            from llama_cpp import Llama, LlamaGrammar

            self._llm = Llama(str(self.model_path), n_ctx=4096, n_threads=self.n_threads, seed=0, verbose=False)
            self._grammar = LlamaGrammar.from_string(self.grammar_text, verbose=False)
        t0 = time.perf_counter()
        out = self._llm.create_chat_completion(
            [{"role": "system", "content": self.system}, {"role": "user", "content": f'Question: "{query}"'}],
            max_tokens=48,
            temperature=0.0,
            grammar=self._grammar,
        )
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        text = out["choices"][0]["message"]["content"].strip()
        tools = [] if text == "none" else sorted({t.strip() for t in text.split(",") if t.strip() in self.registry})
        self.cache[query] = tools
        return tools

    def save(self) -> None:
        CACHE.write_text(json.dumps(self.cache, indent=0, sort_keys=True), encoding="utf-8")


class LLMRouter(_Base):
    name = "B6-LLM"

    def __init__(self, registry: ToolRegistry, choices: Mapping[str, list[str]]) -> None:
        self.registry, self.choices = registry, choices

    def route(self, query: str, ctx: RequestContext) -> Decision:
        t0 = time.perf_counter()
        if query not in self.choices:
            raise KeyError(f"no cached LLM routing decision for {query!r}; run the E8 --prepare step")
        tools = [t for t in self.choices[query] if t in self.registry]
        needed = frozenset().union(*(self.registry.get(t).provides for t in tools)) if tools else frozenset()
        return self._decision(query, ctx, tools, needed, t0)


def load_choices() -> dict[str, list[str]]:
    return json.loads(CACHE.read_text())

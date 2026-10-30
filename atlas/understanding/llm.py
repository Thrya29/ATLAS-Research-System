"""Task understanding with a small local LLM (llama.cpp, GGUF weights; no API).

The model sees each task family's name, a description and training examples,
and must answer with a family name. A GBNF grammar restricts decoding to the
valid names, so the output always parses. The system prompt is identical for
every query, so llama.cpp reuses its cached prefix and only the question is
evaluated per call.

Requires the optional ``llm`` extra (``llama-cpp-python``) and a GGUF file, e.g.
Qwen2.5-0.5B-Instruct (Q4_K_M, ~0.5 GB).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

from atlas.understanding.classifier import Understanding


class LLMUnderstanding:
    def __init__(
        self,
        model_path: str | Path,
        families: Mapping[str, frozenset[str]],
        descriptions: Mapping[str, str],
        examples: Mapping[str, list[str]],
        cache_path: str | Path | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.names = list(families)
        self.families = dict(families)
        options = "\n".join(f"- {n}: {descriptions[n]} Examples: " + "; ".join(f'"{e}"' for e in examples[n]) for n in self.names)
        self.system = (
            "You route questions to the right kind of data source. Reply with the name of the one category that "
            f"best matches the question.\n\nCategories:\n{options}"
        )
        self.grammar_text = "root ::= " + " | ".join(f'"{n}"' for n in self.names)
        self.model_path, self.n_threads = Path(model_path), n_threads
        self._llm = self._grammar = None
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.latencies_ms: list[float] = []

    def _model(self):
        if self._llm is None:
            from llama_cpp import Llama, LlamaGrammar

            self._llm = Llama(str(self.model_path), n_ctx=2048, n_threads=self.n_threads, seed=0, verbose=False)
            self._grammar = LlamaGrammar.from_string(self.grammar_text, verbose=False)
        return self._llm

    def family(self, query: str) -> str:
        if query in self.cache:
            return self.cache[query]
        t0 = time.perf_counter()
        out = self._model().create_chat_completion(
            [{"role": "system", "content": self.system}, {"role": "user", "content": query}],
            max_tokens=16,
            temperature=0.0,
            grammar=self._grammar,
        )
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        text = out["choices"][0]["message"]["content"].strip()
        fam = text if text in self.families else self.names[0]
        self.cache[query] = fam
        return fam

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=0, sort_keys=True), encoding="utf-8")

    def __call__(self, query: str) -> Understanding:
        fam = self.family(query)
        types = self.families[fam]
        return Understanding(types, {t: 1.0 for t in types}, fam, {fam: 1.0})

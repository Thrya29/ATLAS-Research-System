"""Text-retrieval tools: internal documents, an offline web corpus and application logs.

All three are BM25 over local files so the service runs without network access
or API keys. Confidence is the top score squashed relative to the runner-up,
a cheap proxy for "the evidence clearly answers the query".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas.core.types import ToolResult, ToolSpec
from atlas.retrieval.hybrid import BM25
from atlas.tools.base import ToolAdapter, ToolError


def _chunks(path: Path) -> list[tuple[str, str]]:
    out = []
    for f in sorted(path.glob("**/*")):
        if f.suffix in {".md", ".txt", ".log"}:
            text = f.read_text(encoding="utf-8")
            sep = "\n" if f.suffix == ".log" else "\n\n"
            out += [(f.name, c.strip()) for c in text.split(sep) if c.strip()]
    return out


class BM25CorpusTool(ToolAdapter):
    def __init__(self, spec: ToolSpec, corpus_dir: str | Path, k: int = 3) -> None:
        self.spec, self.k = spec, k
        self.chunks = _chunks(Path(corpus_dir))
        if not self.chunks:
            raise ValueError(f"empty corpus at {corpus_dir}")
        self.index = BM25([c for _, c in self.chunks])

    def run(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> tuple[Any, float]:
        scores = self.index.scores(str(args.get("query", query)))
        order = scores.argsort()[::-1][: self.k]
        hits = [
            {"source": self.chunks[i][0], "text": self.chunks[i][1], "score": round(float(scores[i]), 3)}
            for i in order
            if scores[i] > 0
        ]
        if not hits:
            raise ToolError("no matching evidence")
        top = hits[0]["score"]
        second = hits[1]["score"] if len(hits) > 1 else 0.0
        conf = 1 / (1 + math.exp(-(top - second - 1.0)))
        return {"hits": hits}, round(conf, 3)

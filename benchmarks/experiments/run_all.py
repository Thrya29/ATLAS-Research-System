"""Reproduce every experiment in the report.

    python -m benchmarks.experiments.run_all            # full run (seeds 1-20)
    python -m benchmarks.experiments.run_all --quick    # 3 seeds, shorter streams
    python -m benchmarks.experiments.run_all --only E1,E2

Seed 0 was used during development; reported experiments use seeds >= 1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from atlas.router import ATLASConfig
from benchmarks.harness import BASE_CTX, run
from benchmarks.metrics import aggregate, rolling
from benchmarks.simulator.env import default_profiles, distractor_profiles, drift_schedule

OUT = Path(__file__).resolve().parents[1] / "reports"
ROUTERS = ["B0", "B1", "B2", "B3", "B4", "B5", "ATLAS", "Oracle"]
LABEL = {
    "B0": "B0 Single-tool",
    "B1": "B1 Rules",
    "B2": "B2 Supervised model",
    "B3": "B3 Retrieval top-k",
    "B4": "B4 Static cost-aware",
    "B5": "B5 Contextual bandit",
    "ATLAS": "ATLAS",
    "Oracle": "Oracle (privileged)",
}
BASE = ATLASConfig()
ABLATIONS = {
    "ATLAS": BASE,
    "A1 -context": replace(BASE, contextual=False),
    "A2 -reliability": replace(BASE, learn_reliability=False),
    "A3 -cost": replace(BASE, cost_aware=False),
    "A4 -plan graph": replace(BASE, plan_graph=False),
    "A5 -VoI stopping": replace(BASE, stopping=False),
    "A6 -latency": replace(BASE, learn_latency=False),
    "A7 -feasibility": replace(BASE, feasibility_filter=False),
    "A8 -fallback": replace(BASE, fallback=False),
    "A9 -exploration": replace(BASE, explore=False),
}


def _job(kind: str, label: str, router: str, seed: int, n: int, drift: bool, cfg: dict | None, ctx: dict | None, extra: int):
    profiles = default_profiles() + (distractor_profiles(extra, seed) if extra else [])
    c = replace(BASE_CTX, **ctx) if ctx else BASE_CTX
    t0 = time.time()
    recs = run(router, seed, n, drift_schedule() if drift else (), profiles, ATLASConfig(**cfg) if cfg else None, c, label=label)
    agg = aggregate(recs, c.latency_budget_ms)
    agg["wall_s"] = time.time() - t0
    curve = rolling(recs)[::25].tolist() if drift else None
    return kind, label, seed, agg, curve


def jobs(only: set[str], seeds: list[int], quick: bool):
    n1, n2 = (800, 3000) if quick else (2000, 3000)
    out = []
    if "E1" in only:
        out += [("E1", LABEL[r], r, s, n1, False, None, None, 0) for r in ROUTERS for s in seeds]
    if "E2" in only:
        out += [("E2", LABEL[r], r, s, n2, True, None, None, 0) for r in ROUTERS for s in seeds]
    if "E3" in only:
        out += [("E3", k, "ATLAS", s, n2, True, asdict(c), None, 0) for k, c in ABLATIONS.items() for s in seeds]
    if "E4" in only:
        for extra in [0, 25, 100, 250, 500]:
            for r in ["ATLAS", "B5", "B3"]:
                out += [
                    ("E4", f"{LABEL[r]}|{12 + extra}", r, s, 400 if quick else 600, False, None, None, extra) for s in seeds[:5]
                ]
    if "E5" in only:
        for q in [0.6, 0.7, 0.8, 0.9]:
            for v in [0.01, 0.05, 0.2]:
                out += [
                    ("E5", f"ATLAS q>={q} V={v}", "ATLAS", s, n2, True, None, {"min_quality": q, "value_of_success": v}, 0)
                    for s in seeds[:10]
                ]
    return out


def _rows(shard_dir: Path):
    for f in sorted(shard_dir.glob("shard_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _done(shard_dir: Path) -> set[tuple]:
    return {(kind, label, seed) for kind, label, seed, *_ in _rows(shard_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="E1,E2,E3,E4,E5")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--shard", default=None, help="internal: i/N")
    ap.add_argument("--fresh", action="store_true", help="discard results of a previous, interrupted run")
    args = ap.parse_args()
    seeds = list(range(1, 4 if args.quick else 21))
    only = set(args.only.split(","))
    todo = jobs(only, seeds, args.quick)
    shard_dir = OUT / ("shards_quick" if args.quick else "shards")
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        done = _done(shard_dir)
        # Round-robin assignment keeps shards balanced across experiments.
        mine = [j for k, j in enumerate(todo) if k % n == i and (j[0], j[1], j[3]) not in done]
        with (shard_dir / f"shard_{i}.jsonl").open("a", encoding="utf-8") as f:
            for j in mine:
                # One line per finished job: an interrupted run resumes where it stopped.
                f.write(json.dumps(_job(*j), default=float) + "\n")
                f.flush()
        return
    # Independent child processes (not a multiprocessing pool): robust on Windows
    # and in detached shells.
    OUT.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=True)
    if args.fresh:
        for f in shard_dir.glob("shard_*.jsonl"):
            f.unlink()
    print(f"{len(_done(shard_dir))}/{len(todo)} jobs already done", flush=True)
    t0 = time.time()
    base_args = [sys.executable, "-m", "benchmarks.experiments.run_all", "--only", args.only] + (
        ["--quick"] if args.quick else []
    )
    procs = [subprocess.Popen(base_args + ["--shard", f"{i}/{args.workers}"]) for i in range(args.workers)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"shard failures: {codes}")
    results: dict = {}
    wanted = {(j[0], j[1], j[3]) for j in todo}
    for kind, label, seed, agg, curve in _rows(shard_dir):
        if (kind, label, seed) not in wanted:
            continue
        e = results.setdefault(kind, {}).setdefault(label, {"seeds": {}, "curves": {}})
        e["seeds"][str(seed)] = agg
        if curve is not None:
            e["curves"][str(seed)] = curve
    for kind, data in results.items():
        (OUT / f"{kind}.json").write_text(json.dumps(data, default=float, indent=1))
    meta = {
        "seeds": seeds,
        "quick": args.quick,
        "jobs": len(todo),
        "wall_s": round(time.time() - t0, 1),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", ", ".join(f"{k}.json" for k in results), f"in {meta['wall_s']}s")


if __name__ == "__main__":
    main()

"""A generated company "world" for the live evaluation.

The data is synthetic, but every tool that answers questions about it executes
for real: SQLite queries, BM25 and hybrid retrieval, a regex scan over a log
file, and Python in a subprocess sandbox. Ground truth comes from the generator,
independently of any tool, and answers are graded against it.

Real quality differences between alternative tools come from the data, not
from parameters:

* ``replica.db`` holds *preliminary* figures for the latest quarter (a stale copy).
* The document archive also contains superseded policy versions, which compete
  with the current ones in retrieval.
* The recent-log index covers only the last two days; older incidents are only
  in the full archive.
* The calculator computes percentage changes, so it gives a wrong answer to
  median/standard-deviation questions.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REGIONS = ["EMEA", "APAC", "NA", "LATAM"]
METRICS = ["revenue", "orders", "active users", "refunds"]
QUARTERS = [f"{y}Q{q}" for y in (2023, 2024, 2025) for q in range(1, 5)]
LATEST = QUARTERS[-1]
TOPICS = [
    "travel",
    "expense",
    "equipment",
    "training",
    "relocation",
    "remote work",
    "mobile phone",
    "conference",
    "client entertainment",
    "home office",
    "wellness",
    "software purchase",
]
ITEMS = ["hotel", "meal", "taxi", "laptop", "monitor", "course", "visa", "internet", "gift", "subscription"]
SERVICES = ["checkout", "payments", "auth", "search", "inventory", "shipping", "notifications", "billing"]
CODES = ["E1001", "E1002", "E2010", "E2044", "E3100", "E3307", "E4040", "E4291", "E5000", "E5031", "E6120", "E7002"]
TODAY = date(2026, 10, 28)
RECENT_DAYS = 2


@dataclass
class Policy:
    topic: str
    item: str
    limit: int
    old_limit: int


@dataclass
class World:
    root: Path
    values: dict[tuple[str, str, str], float] = field(default_factory=dict)
    policies: list[Policy] = field(default_factory=list)
    incidents: dict[tuple[str, str], str] = field(default_factory=dict)  # (service, day) -> error code


def build(root: Path, seed: int = 7) -> World:
    """Generate the world. Files are written once (marker ``.built``); later calls only
    regenerate the ground truth in memory, so concurrent runs can share the files."""
    rng = np.random.default_rng(seed)
    w = World(root)
    marker = root / ".built"
    write = not (marker.exists() and marker.read_text() == str(seed))
    root.mkdir(parents=True, exist_ok=True)
    _warehouse(w, rng, write)
    _documents(w, rng, write)
    _logs(w, rng, write)
    if write:
        marker.write_text(str(seed))
    return w


def _warehouse(w: World, rng: np.random.Generator, write: bool) -> None:
    for region in REGIONS:
        for metric in METRICS:
            level = float(rng.uniform(2e4, 9e5))
            for q in QUARTERS:
                level *= float(rng.normal(1.03, 0.06))
                w.values[(q, region, metric)] = round(level, 2)
    for name, stale in (("primary.db", False), ("replica.db", True)):
        rows = []
        for (q, region, metric), v in w.values.items():
            if stale and q == LATEST:
                v = round(v * float(rng.uniform(0.85, 0.97)), 2)  # preliminary figure, not yet revised
            rows.append((q, region, metric, v))
        if not write:
            continue  # the random draws above still run, so ground truth matches the files
        path = w.root / name
        path.unlink(missing_ok=True)
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE metrics (period TEXT, region TEXT, metric TEXT, value REAL)")
            c.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?)", rows)
        c.close()


def _documents(w: World, rng: np.random.Generator, write: bool) -> None:
    current, archive = w.root / "docs_current", w.root / "docs_archive"
    for d in (current, archive) if write else ():
        d.mkdir(exist_ok=True)
        for f in d.glob("*.md"):
            f.unlink()
    for topic in TOPICS:
        items = rng.choice(ITEMS, size=3, replace=False)
        now, old = [f"# {topic.title()} policy (effective 2026)"], [f"# {topic.title()} policy (superseded, 2023 edition)"]
        for item in items:
            limit = int(rng.integers(3, 60)) * 25
            prev = limit - int(rng.integers(1, 8)) * 25 if limit > 200 else limit + int(rng.integers(1, 8)) * 25
            w.policies.append(Policy(topic, str(item), limit, prev))
            now.append(f"Under the {topic} policy the {item} limit is {limit} USD; spending above it needs approval.")
            # Old editions keep the same wording, so only the number differs.
            old.append(f"Under the {topic} policy the {item} limit is {prev} USD; spending above it needs approval.")
        if not write:
            continue
        slug = topic.replace(" ", "_")
        (current / f"{slug}.md").write_text("\n\n".join(now) + "\n", encoding="utf-8")
        (archive / f"{slug}.md").write_text("\n\n".join(now) + "\n", encoding="utf-8")
        (archive / f"{slug}_2023.md").write_text("\n\n".join(old) + "\n", encoding="utf-8")


def _logs(w: World, rng: np.random.Generator, write: bool) -> None:
    lines: list[tuple[str, str]] = []
    for back in range(30, -1, -1):
        day = TODAY - timedelta(days=back)
        for svc in SERVICES:
            code = str(rng.choice(CODES))
            w.incidents[(svc, day.isoformat())] = code
            hour = int(rng.integers(0, 24))
            lines.append(
                (day.isoformat(), f"{day} {hour:02d}:{int(rng.integers(60)):02d}:00 {svc} ERROR {code} incident on {svc}")
            )
            for _ in range(6):  # background noise mentioning the service and other codes
                h = int(rng.integers(0, 24))
                level = str(rng.choice(["INFO", "WARN"]))
                lines.append(
                    (
                        day.isoformat(),
                        f"{day} {h:02d}:{int(rng.integers(60)):02d}:00 {svc} {level} {rng.choice(CODES)} handled ok",
                    )
                )
    lines.sort()
    if not write:
        return
    cutoff = (TODAY - timedelta(days=RECENT_DAYS - 1)).isoformat()
    (w.root / "logs_recent").mkdir(exist_ok=True)
    (w.root / "logs_recent" / "recent.log").write_text("\n".join(t for d, t in lines if d >= cutoff) + "\n", encoding="utf-8")
    (w.root / "archive.log").write_text("\n".join(t for _, t in lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- tasks
TEMPLATES = {
    "kpi_lookup": (
        frozenset({"scalars"}),
        [
            "What was {metric} in {region} for {q}?",
            "Give me the {q} {metric} figure for {region}.",
            "How much {metric} did {region} report in {q}?",
            "Look up {region} {metric}, quarter {q}.",
        ],
        ["{region}: {metric} in {q}, please.", "I need the number for {metric} ({region}, {q})."],
    ),
    "growth": (
        frozenset({"numeric_result"}),
        [
            "By what percent did {metric} in {region} change from {q0} to {q}?",
            "Compute the percentage change of {region} {metric} between {q0} and {q}.",
            "How much did {metric} grow in {region} from {q0} to {q}, in percent?",
            "Percent growth of {metric} for {region}, {q0} vs {q}?",
        ],
        [
            "Relative change in {region} {metric} going from {q0} to {q} (as a percentage)?",
            "What is the % increase or decrease in {metric}, {region}, {q0} -> {q}?",
        ],
    ),
    "stats": (
        frozenset({"numeric_result"}),
        [
            "What is the {stat} of quarterly {metric} in {region} during {year}?",
            "Compute the {stat} of {region} {metric} across the quarters of {year}.",
            "Give the {stat} of {metric} for {region} over {year}'s quarters.",
            "Calculate the {stat} of {year} quarterly {metric} in {region}.",
        ],
        [
            "Across {year}, what's the {stat} of {region}'s quarterly {metric}?",
            "{stat} of {metric} by quarter, {region}, year {year}?",
        ],
    ),
    "policy": (
        frozenset({"internal_docs"}),
        [
            "What is the {item} limit under the {topic} policy?",
            "How much can I spend on {item} per the {topic} policy?",
            "What does the {topic} policy allow for {item}?",
            "Find the {item} cap in our {topic} policy.",
        ],
        ["Is there a maximum for {item} in the {topic} rules, and what is it?", "{topic} policy: spending limit on {item}?"],
    ),
    "incident": (
        frozenset({"logs"}),
        [
            "Which error code did {service} raise on {day}?",
            "What error did the {service} service log on {day}?",
            "Find the {service} error code from {day} in the logs.",
            "On {day}, what ERROR code appeared for {service}?",
        ],
        ["Logs check: {service} failure code, {day}?", "What went wrong in {service} on {day} (error code)?"],
    ),
}


@dataclass(frozen=True)
class LiveTask:
    id: int
    family: str
    query: str
    required: frozenset[str]
    truth: object
    args: dict  # per-tool arguments, as an agent would pass them


def tasks(w: World, n: int, seed: int, split: str = "test") -> list[LiveTask]:
    rng = np.random.default_rng([seed, 0 if split == "test" else 1])
    fams = list(TEMPLATES)
    out = []
    for i in range(n):
        fam = fams[int(rng.integers(len(fams)))]
        req, train, test = TEMPLATES[fam]
        tmpl = str(rng.choice(train if split == "train" else test))
        out.append(_task(w, i, fam, req, tmpl, rng))
    return out


def _sql(where: str, params: list) -> dict:
    return {"sql": f"SELECT period, value FROM metrics WHERE {where} ORDER BY period", "params": params}


def _task(w: World, i: int, fam: str, req: frozenset[str], tmpl: str, rng: np.random.Generator) -> LiveTask:
    region, metric = str(rng.choice(REGIONS)), str(rng.choice(METRICS))
    # A third of KPI/growth questions ask about the latest quarter, where the replica is stale.
    qi = len(QUARTERS) - 1 if rng.random() < 0.33 else int(rng.integers(1, len(QUARTERS) - 1))
    q, q0 = QUARTERS[qi], QUARTERS[qi - 1]
    sql_tools = ("sql_primary", "sql_replica")
    if fam == "kpi_lookup":
        sql = _sql("region = ? AND metric = ? AND period = ?", [region, metric, q])
        truth, args = w.values[(q, region, metric)], {t: sql for t in sql_tools}
        query = tmpl.format(metric=metric, region=region, q=q)
    elif fam == "growth":
        sql = _sql("region = ? AND metric = ? AND period IN (?, ?)", [region, metric, q0, q])
        a, b = w.values[(q0, region, metric)], w.values[(q, region, metric)]
        truth = (b - a) / a * 100
        code = 'v = [r["value"] for r in rows]\nresult = (v[-1] - v[0]) / v[0] * 100'
        args = {**{t: sql for t in sql_tools}, "python_sandbox": {"code": code}}
        query = tmpl.format(metric=metric, region=region, q=q, q0=q0)
    elif fam == "stats":
        year = str(rng.choice(["2023", "2024", "2025"]))
        stat = str(rng.choice(["median", "standard deviation"]))
        sql = _sql("region = ? AND metric = ? AND period LIKE ?", [region, metric, f"{year}Q%"])
        vals = [w.values[(f"{year}Q{k}", region, metric)] for k in range(1, 5)]
        truth = statistics.median(vals) if stat == "median" else statistics.stdev(vals)
        fn = "median" if stat == "median" else "stdev"
        code = f'import statistics\nresult = statistics.{fn}([r["value"] for r in rows])'
        args = {**{t: sql for t in sql_tools}, "python_sandbox": {"code": code}}
        query = tmpl.format(stat=stat, metric=metric, region=region, year=year)
    elif fam == "policy":
        p = w.policies[int(rng.integers(len(w.policies)))]
        truth = p
        query = tmpl.format(item=p.item, topic=p.topic)
        args = {"docs_hybrid": {"query": query}, "docs_archive": {"query": query}}
    else:
        svc = str(rng.choice(SERVICES))
        recent = rng.random() < 0.5
        back = int(rng.integers(0, RECENT_DAYS)) if recent else int(rng.integers(RECENT_DAYS, 30))
        day = (TODAY - timedelta(days=back)).isoformat()
        truth = (w.incidents[(svc, day)], day)
        query = tmpl.format(service=svc, day=day)
        args = {"log_recent": {"query": f"{svc} ERROR {day}"}, "log_scan": {"service": svc, "day": day, "level": "ERROR"}}
    return LiveTask(i, fam, query, req, truth, args)


def grade(task: LiveTask, supplied: dict) -> bool:
    """Grade the evidence a plan produced against the generator's ground truth."""
    if not all(t in supplied for t in task.required):
        return False
    if task.family == "kpi_lookup":
        rows = (supplied["scalars"].output or {}).get("rows", [])
        return len(rows) == 1 and abs(float(rows[0]["value"]) - float(task.truth)) < 1e-6
    if task.family in ("growth", "stats"):
        out = supplied["numeric_result"].output
        val = out.get("value") if isinstance(out, dict) else out
        return isinstance(val, int | float) and abs(float(val) - float(task.truth)) <= 1e-6 * max(1.0, abs(float(task.truth)))
    hits = (supplied[next(iter(task.required))].output or {}).get("hits", [])
    if not hits:
        return False
    top = hits[0]["text"]
    if task.family == "policy":
        p: Policy = task.truth
        return f"{p.item} limit is {p.limit} USD" in top
    code, day = task.truth
    return code in top and day in top and "ERROR" in top

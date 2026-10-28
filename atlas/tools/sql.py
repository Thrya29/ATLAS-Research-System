"""Read-only SQL over SQLite, enforced by the engine rather than by string checks.

Defence in depth:
* the database is opened with ``mode=ro`` (the OS-level file handle is read-only);
* an authorizer callback denies every action except reads, SELECT and function calls;
* a progress handler aborts statements that run longer than the time budget;
* only a single statement is accepted and results are capped at ``max_rows``.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas.core.types import ToolResult, ToolSpec
from atlas.tools.base import ToolAdapter, ToolError

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}

_METRICS = ["revenue", "sales", "gross margin", "arr", "net bookings", "signups", "churn", "order volume"]
_PERIOD = re.compile(r"\b(Q[1-4]|H[12])\b", re.I)


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


class SQLiteReadOnlyTool(ToolAdapter):
    def __init__(self, spec: ToolSpec, db_path: str | Path, max_rows: int = 500, time_budget_s: float = 2.0) -> None:
        self.spec, self.db_path = spec, Path(db_path)
        self.max_rows, self.time_budget_s = max_rows, time_budget_s

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        conn.set_authorizer(_authorizer)
        deadline = time.monotonic() + self.time_budget_s
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        return conn

    def resolve(self, query: str) -> tuple[str, tuple]:
        """Deterministic NL->SQL for the demo schema; production agents pass ``sql``."""
        q = query.lower()
        metric = next((m for m in _METRICS if m in q), "revenue")
        periods = [p.upper() for p in _PERIOD.findall(query)]
        if periods:
            marks = ",".join("?" * len(periods))
            return (
                f"SELECT period, metric, value FROM metrics WHERE metric = ? AND period IN ({marks}) ORDER BY period",
                (metric, *periods),
            )
        return "SELECT period, metric, value FROM metrics WHERE metric = ? ORDER BY period DESC LIMIT 2", (metric,)

    def run(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> tuple[Any, float]:
        sql, params = (args["sql"], tuple(args.get("params", ()))) if "sql" in args else self.resolve(query)
        if ";" in sql.strip().rstrip(";"):
            raise ToolError("only a single statement is allowed")
        with self._connect() as conn:
            try:
                cur = conn.execute(sql, params)
                cols = [d[0] for d in cur.description or ()]
                rows = cur.fetchmany(self.max_rows + 1)
            except sqlite3.DatabaseError as e:
                raise ToolError(f"query rejected: {e}") from e
        if len(rows) > self.max_rows:
            raise ToolError(f"result exceeds {self.max_rows} rows; aggregate in SQL")
        table = [dict(zip(cols, r, strict=True)) for r in rows]
        return {"columns": cols, "rows": table, "sql": sql}, (0.9 if table else 0.2)


def create_demo_db(path: str | Path) -> Path:
    """Small, deterministic metrics warehouse used by the demo service and tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metrics (period TEXT, metric TEXT, value REAL, PRIMARY KEY (period, metric))")
    base = {
        "revenue": 1_200_000,
        "sales": 950,
        "gross margin": 0.61,
        "arr": 4_800_000,
        "net bookings": 310_000,
        "signups": 5_400,
        "churn": 0.031,
        "order volume": 18_000,
    }
    growth = {"Q1": 1.00, "Q2": 1.07, "Q3": 1.12, "Q4": 1.21}
    rows = [(p, m, round(v * g, 4)) for m, v in base.items() for p, g in growth.items()]
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path

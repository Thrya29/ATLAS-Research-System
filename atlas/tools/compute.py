"""Computation tools: a safe arithmetic calculator and an isolated Python sandbox."""

from __future__ import annotations

import ast
import json
import operator
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from atlas.core.types import ToolResult, ToolSpec
from atlas.tools.base import ToolAdapter, ToolError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def safe_eval(expr: str) -> float:
    """Evaluate +-*/%** over numeric literals only (no names, calls or attributes)."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ToolError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ToolError(f"unsupported expression element: {type(node).__name__}")

    try:
        return float(ev(ast.parse(expr, mode="eval").body))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ToolError(str(e)) from e


def _values(inputs: Mapping[str, ToolResult]) -> list[float]:
    for key in ("scalars", "table"):
        res = inputs.get(key)
        if res is not None and isinstance(res.output, dict):
            return [float(r["value"]) for r in res.output.get("rows", []) if isinstance(r.get("value"), int | float)]
    return []


class CalculatorTool(ToolAdapter):
    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    def run(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> tuple[Any, float]:
        if "expression" in args:
            return {"expression": args["expression"], "value": safe_eval(str(args["expression"]))}, 0.95
        vals = _values(inputs)
        if len(vals) < 2:
            raise ToolError("need an expression or two upstream values")
        old, new = vals[-2], vals[-1]
        if old == 0:
            raise ToolError("division by zero in percentage change")
        return {"expression": f"({new} - {old}) / {old} * 100", "value": (new - old) / old * 100, "unit": "%"}, 0.9


SANDBOX_IMPORTS = {"math", "statistics", "json"}
_FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "help",
    "memoryview",
}


def check_code(code: str) -> None:
    """Static allow-list check. Defence in depth only: isolation comes from the subprocess/container."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] not in SANDBOX_IMPORTS for m in mods):
                raise ToolError(f"import not allowed: {mods}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ToolError("access to private/dunder attributes is not allowed")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ToolError(f"name not allowed: {node.id}")


_RUNNER = """
import json, sys
data = json.loads(sys.stdin.read())
ns = {"rows": data["rows"], "result": None}
exec(compile(data["code"], "<atlas-sandbox>", "exec"), ns)
print(json.dumps({"result": ns.get("result")}, default=str))
"""

DEFAULT_ANALYSIS = """
import statistics
vals = [r["value"] for r in rows if isinstance(r.get("value"), (int, float))]
result = {"n": len(vals), "mean": statistics.fmean(vals) if vals else None,
          "stdev": statistics.stdev(vals) if len(vals) > 1 else None,
          "pct_change_first_last": (vals[-1] - vals[0]) / vals[0] * 100 if len(vals) > 1 and vals[0] else None}
"""


class PythonSandboxTool(ToolAdapter):
    """Runs allow-listed Python in a separate, isolated interpreter (``-I -S``) with a hard timeout.

    In the container image the process additionally runs as a non-root user
    with no network access (see deployment/docker). Upstream table rows are
    available to the code as ``rows``; the code must assign ``result``.
    """

    def __init__(self, spec: ToolSpec, timeout_s: float = 3.0, max_output: int = 64_000) -> None:
        self.spec, self.timeout_s, self.max_output = spec, timeout_s, max_output

    def run(self, query: str, args: Mapping[str, Any], inputs: Mapping[str, ToolResult]) -> tuple[Any, float]:
        code = str(args.get("code", DEFAULT_ANALYSIS))
        try:
            check_code(code)
        except SyntaxError as e:
            raise ToolError(f"syntax error: {e}") from e
        table = inputs.get("table")
        rows = table.output.get("rows", []) if table is not None and isinstance(table.output, dict) else []
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=json.dumps({"code": code, "rows": rows}),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise ToolError("sandbox timed out") from e
        if proc.returncode != 0:
            raise ToolError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "sandbox failed")
        if len(proc.stdout) > self.max_output:
            raise ToolError("sandbox output too large")
        return json.loads(proc.stdout)["result"], 0.9

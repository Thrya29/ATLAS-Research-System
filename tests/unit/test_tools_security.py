import pytest

from atlas.tools.base import ToolError
from atlas.tools.compute import CalculatorTool, PythonSandboxTool, check_code, safe_eval
from atlas.tools.sql import SQLiteReadOnlyTool, create_demo_db
from tests.conftest import spec


@pytest.fixture
def sql(tmp_path):
    return SQLiteReadOnlyTool(spec("sql", ["table"]), create_demo_db(tmp_path / "w.db"))


def test_sql_select_and_nl_resolution(sql):
    out, conf = sql.run("revenue growth from Q1 to Q2", {}, {})
    assert [r["period"] for r in out["rows"]] == ["Q1", "Q2"] and conf > 0.5


@pytest.mark.parametrize(
    "stmt",
    [
        "DELETE FROM metrics",
        "UPDATE metrics SET value = 0",
        "INSERT INTO metrics VALUES ('Q9','x',1)",
        "DROP TABLE metrics",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA writable_schema = 1",
        "SELECT 1; DELETE FROM metrics",
    ],
)
def test_sql_rejects_writes(sql, stmt):
    with pytest.raises(ToolError):
        sql.run("", {"sql": stmt}, {})
    out, _ = sql.run("", {"sql": "SELECT COUNT(*) AS n FROM metrics"}, {})
    assert out["rows"][0]["n"] == 32


def test_safe_eval():
    assert safe_eval("(1284000 - 1200000) / 1200000 * 100") == pytest.approx(7.0)
    for bad in ["__import__('os')", "open('x')", "a + 1", "2 ** 1000", "(lambda: 1)()"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from subprocess import run",
        "open('/etc/passwd')",
        "x = ().__class__.__bases__",
        "eval('1')",
        "getattr(1, 'real')",
    ],
)
def test_sandbox_static_rejections(code):
    with pytest.raises(ToolError):
        check_code(code)


def test_sandbox_runs_allowed_code_with_timeout():
    tool = PythonSandboxTool(spec("py", ["n"], ["table"]), timeout_s=5)
    out, _ = tool.run("", {"code": "import statistics\nresult = statistics.fmean([1, 2, 3])"}, {})
    assert out == 2
    slow = PythonSandboxTool(spec("py", ["n"], ["table"]), timeout_s=0.5)
    with pytest.raises(ToolError, match="timed out"):
        slow.run("", {"code": "while True:\n    pass"}, {})


def test_calculator_from_upstream_values(sql):
    from atlas.core.types import ToolResult

    rows, _ = sql.run("revenue Q1 Q2", {}, {})
    calc = CalculatorTool(spec("calc", ["n"], ["scalars"]))
    out, _ = calc.run("", {}, {"scalars": ToolResult("sql", True, 1, output=rows)})
    assert out["value"] == pytest.approx(7.0)

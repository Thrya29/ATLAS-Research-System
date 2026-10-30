"""Build the interactive results dashboard (one self-contained HTML file).

    python -m benchmarks.experiments.dashboard      # -> benchmarks/reports/dashboard.html

Reads the committed result files (E1-E6 JSON, live-evaluation seeds) and embeds
summaries in the page; no network access or JS libraries are needed to view it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.stats import bootstrap_ci

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SHORT = {
    "B0 Single-tool": "B0 Single tool",
    "B1 Rules": "B1 Rules",
    "B2 Supervised model": "B2 Supervised",
    "B3 Retrieval top-k": "B3 Retrieval",
    "B4 Static cost-aware": "B4 Static",
    "B5 Contextual bandit": "B5 Bandit",
    "ATLAS": "ATLAS",
    "Oracle (privileged)": "Oracle",
}
METRICS = {
    "success": ("Task success", 1.0),
    "cost_per_success_musd": ("Cost per success (m$)", 1.0),
    "calls_per_task": ("Tool calls per task", 1.0),
    "unnecessary_per_task": ("Unnecessary calls per task", 1.0),
    "latency_p95": ("Latency p95 (ms)", 1.0),
}
DRIFTS = [
    (1000, "search_b overloaded, sql_replica lag"),
    (1800, "vector_docs outage"),
    (2000, "search_b recovers, sandbox slows"),
    (2200, "search_a upgraded"),
]


def _ci(xs) -> dict:
    m, lo, hi = bootstrap_ci([float(x) for x in xs])
    return {"mean": m, "lo": lo, "hi": hi}


def _load(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def comparison() -> dict:
    out = {}
    for exp in ("E1", "E2"):
        data = _load(exp)
        out[exp] = {key: {SHORT[r]: _ci([s[key] for s in v["seeds"].values()]) for r, v in data.items()} for key in METRICS}
    return out


def curves() -> dict:
    data = _load("E2")
    keep = ["ATLAS", "B2 Supervised model", "B4 Static cost-aware", "B5 Contextual bandit", "Oracle (privileged)"]
    return {SHORT[r]: np.mean([c for c in data[r]["curves"].values()], axis=0).round(4).tolist() for r in keep}


def tradeoff() -> dict:
    e5, e2 = _load("E5"), _load("E2")
    pts = []
    for label, v in e5.items():
        q, val = label.split("q>=")[1].split(" V=")
        if val == "0.05":
            seeds = v["seeds"].values()
            pts.append(
                {
                    "q": float(q),
                    "success": float(np.mean([s["success"] for s in seeds])),
                    "cost": float(np.mean([s["cost_per_success_musd"] for s in seeds])),
                }
            )
    base = []
    for r in ("B2 Supervised model", "B4 Static cost-aware", "B5 Contextual bandit"):
        seeds = e2[r]["seeds"].values()
        base.append(
            {
                "name": SHORT[r],
                "success": float(np.mean([s["success"] for s in seeds])),
                "cost": float(np.mean([s["cost_per_success_musd"] for s in seeds])),
            }
        )
    return {"atlas": sorted(pts, key=lambda p: p["q"]), "baselines": base}


def ablations() -> list[dict]:
    e3 = _load("E3")
    full = e3["ATLAS"]["seeds"]
    rows = []
    for label, v in e3.items():
        if label == "ATLAS":
            continue
        d = [full[s]["success"] - v["seeds"][s]["success"] for s in full]
        c = [v["seeds"][s]["cost_per_success_musd"] - full[s]["cost_per_success_musd"] for s in full]
        name = label.split(" -")[1].strip()
        rows.append({"name": f"{name} ({label.split()[0]})", "delta": _ci(d), "cost_saved": float(np.mean(c))})
    return sorted(rows, key=lambda r: -r["delta"]["mean"])


def scaling() -> dict:
    e4 = _load("E4")
    out: dict = {}
    for label, v in e4.items():
        router, n = label.split("|")
        seeds = v["seeds"].values()
        out.setdefault(SHORT[router], []).append(
            {
                "tools": int(n),
                "success": float(np.mean([s["success"] for s in seeds])),
                "p95": float(np.mean([s["planning_ms_p95"] for s in seeds])),
            }
        )
    return {k: sorted(v, key=lambda p: p["tools"]) for k, v in out.items()}


def live() -> dict:
    files = sorted((REPORTS / "live").glob("seed_*.json"), key=lambda f: int(f.stem.split("_")[1]))
    runs = [json.loads(f.read_text())["records"] for f in files]
    n = max(r["t"] for r in runs[0]["ATLAS"]) + 1
    lo, hi = int(0.30 * n), int(0.65 * n)
    names = list(runs[0])
    fams = ["kpi_lookup", "growth", "stats", "policy", "incident"]
    out = {"seeds": len(runs), "routers": {}, "families": fams}
    for name in names:
        per = [run[name] for run in runs]
        out["routers"][name] = {
            "success": _ci([np.mean([r["success"] for r in rs]) for rs in per]),
            "incident": _ci([np.mean([r["success"] for r in rs if lo <= r["t"] < hi]) for rs in per]),
            "cost": float(np.mean([1000 * sum(r["cost"] for r in rs) / max(1, sum(r["success"] for r in rs)) for rs in per])),
            "family": [float(np.mean([np.mean([r["success"] for r in rs if r["family"] == f]) for rs in per])) for f in fams],
        }
    return out


def understanding() -> dict:
    e6 = _load("E6")
    variants = ["oracle", "classifier", "noisy-0.2", "noisy-0.4", "llm"]
    labels = {
        "oracle": "Oracle",
        "classifier": "Classifier",
        "noisy-0.2": "20% corrupted",
        "noisy-0.4": "40% corrupted",
        "llm": "Local LLM 0.5B",
    }
    out = {
        "levels": [labels[v] for v in variants],
        "accuracy": [e6["accuracy"][v]["exact_set_acc"] for v in variants],
        "routers": {},
    }
    for r, short in (("ATLAS", "ATLAS"), ("B4", "B4 Static"), ("B5", "B5 Bandit")):
        out["routers"][short] = [float(np.mean([x["success"] for x in e6["routing"][f"{v}|{r}"]])) for v in variants]
    return out


def bars_of(kind: str, labels: dict[str, str], keys=("success", "cost_per_success_musd", "calls_per_task")) -> dict:
    path = REPORTS / f"{kind}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        short: {k: _ci([s[k] for s in data[lab]["seeds"].values()]) for k in keys} for lab, short in labels.items() if lab in data
    }


def main() -> None:
    meta = json.loads((REPORTS / "run_meta.json").read_text())
    data = {
        "meta": {"seeds": len(meta["seeds"])},
        "comparison": comparison(),
        "curves": curves(),
        "drifts": [{"t": t, "label": lab} for t, lab in DRIFTS],
        "tradeoff": tradeoff(),
        "ablations": ablations(),
        "scaling": scaling(),
        "live": live(),
        "understanding": understanding(),
        "cache": bars_of("E7", {"no cache": "No cache", "fixed-TTL cache": "Fixed-TTL cache", "learned cache": "Learned cache"}),
        "llm_router": bars_of(
            "E8",
            {
                "ATLAS|drift": "ATLAS",
                "B2 Supervised model|drift": "B2 Supervised",
                "B5 Contextual bandit|drift": "B5 Bandit",
                "B6 LLM router (Qwen 0.5B)|drift": "B6 LLM router",
            },
        ),
        "metrics": {k: v[0] for k, v in METRICS.items()},
    }
    template = (Path(__file__).with_name("dashboard_template.html")).read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (REPORTS / "dashboard.html").write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print("wrote", REPORTS / "dashboard.html")


if __name__ == "__main__":
    main()

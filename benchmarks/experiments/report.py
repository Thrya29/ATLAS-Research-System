"""Build benchmarks/reports/RESULTS.md and figures from the experiment JSON files.

python -m benchmarks.experiments.report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.stats import bootstrap_ci, holm, paired  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
FIG = OUT / "figures"

# Validated categorical palette (fixed order, colour follows the router).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDER = [
    "B0 Single-tool",
    "B1 Rules",
    "B2 Supervised model",
    "B3 Retrieval top-k",
    "B4 Static cost-aware",
    "B5 Contextual bandit",
    "ATLAS",
    "Oracle (privileged)",
]
COLOR = dict(zip(ORDER, COLORS, strict=True))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

MAIN = [
    ("success", "Task success", "{:.3f}", +1),
    ("cost_per_success_musd", "Cost / success (m$)", "{:.2f}", -1),
    ("calls_per_task", "Tool calls / task", "{:.2f}", -1),
    ("unnecessary_per_task", "Unnecessary calls / task", "{:.3f}", -1),
    ("latency_p50", "Latency p50 (ms)", "{:.0f}", -1),
    ("latency_p95", "Latency p95 (ms)", "{:.0f}", -1),
    ("budget_violation", "P(latency > budget)", "{:.3f}", -1),
]
DRIFT = [
    ("success_pre_drift", "Success t<1000", "{:.3f}"),
    ("success_drift_1", "Success 1000-2000", "{:.3f}"),
    ("success_drift_2", "Success 2000-3000", "{:.3f}"),
    ("degraded_tool_share_during", "Degraded-tool share (during)", "{:.3f}"),
    ("degraded_tool_share_after", "Same tool share (after recovery)", "{:.3f}"),
    ("recovery_tasks", "Tasks to recover", "{:.0f}"),
]


def load(kind: str) -> dict:
    p = OUT / f"{kind}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def per_seed(data: dict, label: str, metric: str) -> tuple[list[int], list[float]]:
    seeds = sorted(data[label]["seeds"], key=int)
    return seeds, [data[label]["seeds"][s][metric] for s in seeds]


def ci_cell(vals: list[float], fmt: str) -> str:
    m, lo, hi = bootstrap_ci(vals)
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def table(data: dict, labels: list[str], metrics) -> list[str]:
    head = "| Router | " + " | ".join(m[1] for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 1)]
    for lab in labels:
        if lab in data:
            cells = [ci_cell(per_seed(data, lab, m[0])[1], m[2]) for m in metrics]
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return lines


def tests(data: dict, ref: str, others: list[str], metrics) -> list[str]:
    rows, ps = [], []
    for lab in others:
        if lab not in data or lab == ref:
            continue
        for key, name, fmt, _sign in metrics:
            s_ref, a = per_seed(data, ref, key)
            s_oth, b = per_seed(data, lab, key)
            assert s_ref == s_oth, "paired tests need identical seeds"
            r = paired(a, b)
            rows.append((lab, name, fmt, r))
            ps.append(r["p"])
    adj = holm(ps)
    out = [f"| {ref} vs | Metric | Δ (ref − other) [95% CI] | Holm p | d_z |", "|---|---|---|---|---|"]
    for (lab, name, fmt, r), p in zip(rows, adj, strict=True):
        out.append(
            f"| {lab} | {name} | {fmt.format(r['diff'])} [{fmt.format(r['lo'])}, {fmt.format(r['hi'])}] | {p:.2g} | {r['dz']:.2f} |"
        )
    return out


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_drift(e2: dict) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    for lab in ORDER:
        if lab not in e2 or not e2[lab]["curves"]:
            continue
        curves = np.array(list(e2[lab]["curves"].values()))
        x = np.arange(curves.shape[1]) * 25 + 100
        ax.plot(
            x,
            curves.mean(axis=0),
            color=COLOR[lab],
            linewidth=2 if lab == "ATLAS" else 1.4,
            linestyle="--" if lab.startswith("Oracle") else "-",
            label=lab,
        )
    events = [
        (1000, "search_b overload\nreplica lag", 0.05),
        (1800, "docs outage", 0.22),
        (2000, "search_b recovers\npython slows", 0.05),
    ]
    for t, txt, y in events:
        ax.axvline(t, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(t + 15, y, txt, fontsize=7, color=MUTED, va="bottom")
    ax.set_ylim(0, 1)
    _style(ax, "Rolling task success under tool drift (window 100, mean over seeds)", "task index", "success rate")
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.38))
    fig.tight_layout()
    p = FIG / "e2_drift_success.png"
    fig.savefig(p, facecolor="#fcfcfb")
    plt.close(fig)
    return p.name


def fig_pareto(e2: dict, e5: dict) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    if e5:
        pts = []
        for lab in e5:
            pts.append((np.mean(per_seed(e5, lab, "cost_per_success_musd")[1]), np.mean(per_seed(e5, lab, "success")[1]), lab))
        xs, ys, _ = zip(*pts, strict=True)
        ax.scatter(
            xs,
            ys,
            s=30,
            color=COLOR["ATLAS"],
            alpha=0.45,
            edgecolor="#fcfcfb",
            linewidth=1.5,
            label="ATLAS (SLO / value sweep)",
            zorder=3,
        )
    for lab in ORDER:
        if lab not in e2:
            continue
        x = np.mean(per_seed(e2, lab, "cost_per_success_musd")[1])
        y = np.mean(per_seed(e2, lab, "success")[1])
        ax.scatter(
            [x],
            [y],
            s=220 if lab == "ATLAS" else 70,
            color=COLOR[lab],
            edgecolor="#fcfcfb",
            linewidth=1 if lab == "ATLAS" else 2,
            zorder=4,
            marker="*" if lab == "ATLAS" else ("D" if lab.startswith("Oracle") else "o"),
            label=lab,
        )
        ax.annotate(lab.split(" ")[0], (x, y), xytext=(6, -3), textcoords="offset points", fontsize=7, color=INK)
    _style(ax, "Quality–cost trade-off under drift (up and left is better)", "cost per successful task (m$)", "task success")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    p = FIG / "e5_pareto.png"
    fig.savefig(p, facecolor="#fcfcfb")
    plt.close(fig)
    return p.name


def fig_scaling(e4: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    groups: dict[str, list[tuple[int, float, float]]] = {}
    for lab in e4:
        name, size = lab.split("|")
        groups.setdefault(name, []).append(
            (int(size), np.mean(per_seed(e4, lab, "planning_ms_p95")[1]), np.mean(per_seed(e4, lab, "success")[1]))
        )
    for name, rows in groups.items():
        rows.sort()
        s, ms, succ = zip(*rows, strict=True)
        axes[0].plot(s, ms, marker="o", markersize=5, linewidth=2, color=COLOR.get(name, MUTED), label=name)
        axes[1].plot(s, succ, marker="o", markersize=5, linewidth=2, color=COLOR.get(name, MUTED), label=name)
    _style(axes[0], "Routing overhead (p95)", "registered tools", "ms per decision")
    _style(axes[1], "Task success", "registered tools", "success rate")
    for ax in axes:
        ax.set_xscale("log")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = FIG / "e4_scaling.png"
    fig.savefig(p, facecolor="#fcfcfb")
    plt.close(fig)
    return p.name


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    e1, e2, e3, e4, e5 = (load(k) for k in ("E1", "E2", "E3", "E4", "E5"))
    meta = json.loads((OUT / "run_meta.json").read_text()) if (OUT / "run_meta.json").exists() else {}
    md = [
        "# ATLAS — experimental results",
        "",
        f"Generated by `python -m benchmarks.experiments.report`. Seeds: {meta.get('seeds')}"
        f"{' (QUICK RUN — not the reported configuration)' if meta.get('quick') else ''}.",
        "Cells show the mean over seeds with a 95% bootstrap CI over seeds. Paired tests are Wilcoxon",
        "signed-rank on per-seed values (common random numbers), Holm-corrected within each table;",
        "d_z is the paired standardised mean difference.",
        "",
    ]
    if e1:
        md += ["## E1 — Stationary tools (2000 held-out-template tasks per seed)", ""] + table(e1, ORDER, MAIN)
        md += ["", "### ATLAS vs baselines (E1)", ""] + tests(e1, "ATLAS", ORDER[:6], MAIN[:4]) + [""]
    if e2:
        md += ["## E2 — Non-stationary tools (3000 tasks, drift at t=1000/1800/2000/2200)", ""] + table(e2, ORDER, MAIN)
        md += ["", "### Adaptation (E2)", ""] + table(e2, ORDER, DRIFT)
        md += ["", "### ATLAS vs baselines (E2)", ""] + tests(e2, "ATLAS", ORDER[:6], MAIN[:4])
        md += ["", f"![drift]({'figures/' + fig_drift(e2)})", ""]
    if e3:
        labels = list(e3)
        md += ["## E3 — Ablations under drift", ""] + table(e3, labels, MAIN[:6])
        md += ["", "### Full ATLAS vs each ablation (E3)", ""] + tests(e3, "ATLAS", labels, MAIN[:3]) + [""]
    if e4:
        md += [
            "## E4 — Registry scaling (distractor tools added)",
            "",
            "| Router | Tools | Success | Cost / success (m$) | Planning p50 (ms) | Planning p95 (ms) |",
            "|---|---|---|---|---|---|",
        ]
        for lab in sorted(e4, key=lambda x: (x.split("|")[0], int(x.split("|")[1]))):
            name, size = lab.split("|")
            v = {
                k: np.mean(per_seed(e4, lab, k)[1])
                for k in ("success", "cost_per_success_musd", "planning_ms_p50", "planning_ms_p95")
            }
            md.append(
                f"| {name} | {size} | {v['success']:.3f} | {v['cost_per_success_musd']:.2f} | {v['planning_ms_p50']:.2f} | {v['planning_ms_p95']:.2f} |"
            )
        md += ["", f"![scaling]({'figures/' + fig_scaling(e4)})", ""]
    if e5:
        md += [
            "## E5 — SLO / value-of-success sweep under drift",
            "",
            "| Setting | Success | Cost / success (m$) | Calls / task | Latency p95 |",
            "|---|---|---|---|---|",
        ]
        for lab in e5:
            v = {
                k: np.mean(per_seed(e5, lab, k)[1]) for k in ("success", "cost_per_success_musd", "calls_per_task", "latency_p95")
            }
            md.append(
                f"| {lab} | {v['success']:.3f} | {v['cost_per_success_musd']:.2f} | {v['calls_per_task']:.2f} | {v['latency_p95']:.0f} |"
            )
        md += ["", f"![pareto]({'figures/' + fig_pareto(e2, e5)})", ""]
    (OUT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()

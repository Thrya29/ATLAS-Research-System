"""Feasible tool-plan enumeration over the AND-OR capability graph.

Types are OR-nodes (any provider satisfies them); tools are AND-nodes (all of
their required input types must also be satisfied). ``enumerate_plans`` runs a
depth-first branch-and-bound search that returns up to ``k`` cheapest distinct
feasible tool sets. The bound is admissible: the cost of the partial plan plus,
for the still-unsatisfied types, the largest of their cheapest-provider costs.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence

from atlas.core.types import Plan, PlanStep, ToolSpec


def enumerate_plans(
    needed: frozenset[str],
    candidates: Mapping[str, Sequence[ToolSpec]],
    cost: Callable[[ToolSpec], float],
    k: int = 32,
    max_tools: int = 6,
) -> list[frozenset[str]]:
    """Return up to ``k`` cheapest feasible tool-name sets covering ``needed``.

    ``candidates`` maps each type to its admissible providers (already filtered
    for permissions, health and schema compatibility). Types without candidates
    make every plan that needs them infeasible.
    """
    if not needed:
        return [frozenset()]
    min_cost = {t: min((cost(x) for x in c), default=float("inf")) for t, c in candidates.items()}

    def lower_bound(types: frozenset[str]) -> float:
        return max((min_cost.get(t, float("inf")) for t in types), default=0.0)

    best: list[tuple[float, int, frozenset[str]]] = []  # max-heap via negated cost
    seen: set[frozenset[str]] = set()
    counter = 0

    def kth_cost() -> float:
        return -best[0][0] if len(best) >= k else float("inf")

    def dfs(open_types: frozenset[str], chosen: frozenset[str], provided: frozenset[str], acc: float) -> None:
        nonlocal counter
        if acc + lower_bound(open_types) >= kth_cost():
            return
        if not open_types:
            if chosen not in seen:
                seen.add(chosen)
                counter += 1
                heapq.heappush(best, (-acc, counter, chosen))
                if len(best) > k:
                    heapq.heappop(best)
            return
        if len(chosen) >= max_tools:
            return
        # Branch on the most constrained open type (fewest candidate providers).
        t = min(open_types, key=lambda x: (len(candidates.get(x, ())), x))
        for tool in candidates.get(t, ()):
            if tool.name in chosen:
                continue
            new_provided = provided | tool.provides
            new_open = (open_types | (tool.requires - new_provided)) - new_provided
            dfs(new_open, chosen | {tool.name}, new_provided, acc + cost(tool))

    dfs(needed, frozenset(), frozenset(), 0.0)
    return [s for _, _, s in sorted(best, key=lambda x: (-x[0], x[1]))]


def build_plan(tool_names: frozenset[str], needed: frozenset[str], tools: Mapping[str, ToolSpec]) -> Plan:
    """Wire a tool set into a DAG: each required input depends on a provider in the set."""
    names = sorted(tool_names, key=lambda n: (len(tools[n].requires), n))
    index = {n: i for i, n in enumerate(names)}
    steps: list[PlanStep] = []
    for n in names:
        spec = tools[n]
        deps = []
        for r in sorted(spec.requires):
            prov = next((m for m in names if r in tools[m].provides and m != n), None)
            if prov is None:
                raise ValueError(f"plan {sorted(tool_names)} leaves input {r!r} of {n!r} unsatisfied")
            deps.append(index[prov])
        consumed = needed | frozenset().union(*(tools[m].requires for m in names))
        steps.append(PlanStep(tool=n, supplies=spec.provides & consumed, depends_on=tuple(sorted(set(deps)))))
    # Dependencies must point backwards for a valid topological order.
    for i, s in enumerate(steps):
        if any(d >= i for d in s.depends_on):
            return _toposort(steps)
    return Plan(steps=steps)


def _toposort(steps: list[PlanStep]) -> Plan:
    order: list[int] = []
    state: dict[int, int] = {}

    def visit(i: int) -> None:
        if state.get(i) == 2:
            return
        if state.get(i) == 1:
            raise ValueError("cyclic tool dependencies")
        state[i] = 1
        for d in steps[i].depends_on:
            visit(d)
        state[i] = 2
        order.append(i)

    for i in range(len(steps)):
        visit(i)
    remap = {old: new for new, old in enumerate(order)}
    return Plan(
        steps=[
            PlanStep(steps[o].tool, steps[o].supplies, tuple(sorted(remap[d] for d in steps[o].depends_on)), steps[o].purpose)
            for o in order
        ]
    )

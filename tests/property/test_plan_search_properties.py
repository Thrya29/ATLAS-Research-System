"""Branch-and-bound must agree with brute force on random small registries."""

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from atlas.core.registry import ToolRegistry
from atlas.planning.graph import enumerate_plans
from tests.conftest import spec

TYPES = ["a", "b", "c", "d"]


@st.composite
def registries(draw):
    n = draw(st.integers(2, 7))
    tools = []
    for i in range(n):
        provides = draw(st.sets(st.sampled_from(TYPES), min_size=1, max_size=2))
        requires = draw(st.sets(st.sampled_from([t for t in TYPES if t not in provides]), max_size=1))
        tools.append(spec(f"t{i}", provides, requires, cost=draw(st.integers(1, 20)) / 2))
    needed = frozenset(draw(st.sets(st.sampled_from(TYPES), min_size=1, max_size=2)))
    return ToolRegistry(tools), needed


def feasible(reg, names, needed):
    provided = frozenset().union(*(reg.get(n).provides for n in names))
    required = frozenset().union(*(reg.get(n).requires for n in names))
    return needed <= provided and required <= provided


@settings(max_examples=200, deadline=None)
@given(registries())
def test_matches_brute_force_minimum(case):
    reg, needed = case
    names = [s.name for s in reg]
    brute = [frozenset(c) for r in range(1, len(names) + 1) for c in itertools.combinations(names, r) if feasible(reg, c, needed)]
    plans = enumerate_plans(needed, {t: reg.providers(t) for t in reg.types}, lambda s: s.cost, k=64, max_tools=len(names))
    assert all(feasible(reg, p, needed) for p in plans)
    if not brute:
        assert plans == []
    else:
        best = min(sum(reg.get(n).cost for n in p) for p in brute)
        assert plans and sum(reg.get(n).cost for n in plans[0]) == best

import numpy as np

from atlas.bandits.estimators import DecayedBeta, DecayedLogNormal, HierarchicalQuality
from atlas.execution.circuit import CircuitBreakers, State


def test_decayed_beta_tracks_change():
    b = DecayedBeta(0.5, gamma=0.98)
    for t in range(200):
        b.update(1.0, t)
    assert b.mean(200) > 0.95
    for t in range(200, 400):
        b.update(0.0, t)
    assert b.mean(400) < 0.1


def test_decay_restores_uncertainty_for_unused_arm():
    b = DecayedBeta(0.5, gamma=0.99)
    for t in range(300):
        b.update(0.0, t)
    n_then = b.n(300)
    assert b.n(800) < 0.01 * n_then  # forgotten -> Thompson sampling will re-explore
    assert abs(b.mean(1500) - 0.5) < 0.01  # back to the prior


def test_lognormal_latency_estimate():
    rng = np.random.default_rng(0)
    est = DecayedLogNormal(prior_ms=100)
    for t, x in enumerate(np.exp(rng.normal(np.log(400), 0.2, 500))):
        est.update(float(x), t)
    assert 380 < est.mean_ms(500) < 450


def test_hierarchical_quality_pools_then_specialises():
    q = HierarchicalQuality(prior_mean=0.8)
    for t in range(300):
        q.update("fam_a", 1.0, t)
        q.update("fam_b", 0.0 if t % 2 else 1.0, t)
    assert q.mean("fam_a", 300) > q.mean("fam_b", 300)
    # An unseen context starts from the tool-level estimate, not the uninformed prior.
    assert abs(q.mean("unseen", 300) - q.global_.mean(300)) < 1e-9


def test_circuit_breaker_cycle():
    now = {"t": 0.0}
    cb = CircuitBreakers(clock=lambda: now["t"], failure_threshold=3, cooldown=10)
    for _ in range(3):
        cb.record("x", False)
    assert cb.state("x") is State.OPEN and not cb.allow("x")
    now["t"] = 11
    assert cb.state("x") is State.HALF_OPEN
    cb.record("x", False)
    assert cb.state("x") is State.OPEN
    now["t"] = 22
    cb.record("x", True)
    assert cb.state("x") is State.CLOSED

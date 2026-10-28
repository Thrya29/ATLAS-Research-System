"""Time-decayed Bayesian estimators for non-stationary tool statistics.

All sufficient statistics decay geometrically with *global* time (number of
routing decisions), not with the number of observations of a tool. A tool that
stops being selected therefore regains posterior variance and is eventually
re-explored by Thompson sampling, which is what lets the router notice that a
previously degraded tool has recovered (discounted Thompson sampling, cf. Raj &
Kalyani 2017; discounted UCB, Garivier & Moulines 2011).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DecayedBeta:
    """Beta posterior on a Bernoulli rate with decayed pseudo-counts."""

    prior_mean: float
    prior_strength: float = 4.0
    gamma: float = 0.997
    a: float = 0.0
    b: float = 0.0
    t: int = 0

    def _decay(self, now: int) -> None:
        if now > self.t:
            f = self.gamma ** (now - self.t)
            self.a *= f
            self.b *= f
            self.t = now

    def update(self, success: float, now: int, weight: float = 1.0) -> None:
        self._decay(now)
        self.a += weight * success
        self.b += weight * (1.0 - success)

    def params(self, now: int, prior_mean: float | None = None) -> tuple[float, float]:
        self._decay(now)
        m = self.prior_mean if prior_mean is None else prior_mean
        return self.a + self.prior_strength * m, self.b + self.prior_strength * (1 - m)

    def mean(self, now: int, prior_mean: float | None = None) -> float:
        a, b = self.params(now, prior_mean)
        return a / (a + b)

    def sample(self, rng: np.random.Generator, now: int, prior_mean: float | None = None) -> float:
        a, b = self.params(now, prior_mean)
        return float(rng.beta(a, b))

    def n(self, now: int) -> float:
        self._decay(now)
        return self.a + self.b


@dataclass
class DecayedLogNormal:
    """Normal-Inverse-Gamma-style estimate of log-latency with decayed moments."""

    prior_ms: float
    prior_sigma: float = 0.5
    prior_strength: float = 3.0
    gamma: float = 0.997
    w: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    t: int = 0

    def _decay(self, now: int) -> None:
        if now > self.t:
            f = self.gamma ** (now - self.t)
            self.w *= f
            self.s1 *= f
            self.s2 *= f
            self.t = now

    def update(self, latency_ms: float, now: int) -> None:
        self._decay(now)
        x = math.log(max(latency_ms, 1e-3))
        self.w += 1.0
        self.s1 += x
        self.s2 += x * x

    def moments(self, now: int) -> tuple[float, float, float]:
        """Posterior (mu, sigma, effective n) of log-latency, shrunk to the prior."""
        self._decay(now)
        m0, k0 = math.log(self.prior_ms), self.prior_strength
        n = self.w
        mu = (k0 * m0 + self.s1) / (k0 + n)
        var_data = (self.s2 - self.s1 * self.s1 / n) / n if n > 1e-9 else 0.0
        var = (k0 * self.prior_sigma**2 + n * max(var_data, 0.0)) / (k0 + n)
        return mu, math.sqrt(max(var, 1e-6)), n

    def sample_mu(self, rng: np.random.Generator, now: int) -> tuple[float, float]:
        """Thompson draw of the log-mean (uncertainty shrinks with effective n)."""
        mu, sigma, n = self.moments(now)
        return float(rng.normal(mu, sigma / math.sqrt(self.prior_strength + n))), sigma

    def mean_ms(self, now: int) -> float:
        mu, sigma, _ = self.moments(now)
        return math.exp(mu + sigma**2 / 2)


class HierarchicalQuality:
    """Per-(tool, context) Beta posteriors shrunk toward the tool's global rate.

    Contexts with little data borrow strength from the tool-level estimate
    (empirical-Bayes partial pooling), so a new task family starts from a
    sensible prior instead of an uninformative one.
    """

    def __init__(self, prior_mean: float, gamma: float = 0.998, pool_strength: float = 4.0) -> None:
        self.global_ = DecayedBeta(prior_mean, prior_strength=4.0, gamma=gamma)
        self.ctx: dict[str, DecayedBeta] = {}
        self.gamma, self.pool = gamma, pool_strength

    def _ctx(self, c: str) -> DecayedBeta:
        if c not in self.ctx:
            self.ctx[c] = DecayedBeta(self.global_.prior_mean, prior_strength=self.pool, gamma=self.gamma)
        return self.ctx[c]

    def update(self, context: str | None, success: float, now: int, weight: float = 1.0) -> None:
        self.global_.update(success, now, weight)
        if context is not None:
            self._ctx(context).update(success, now, weight)

    def mean(self, context: str | None, now: int) -> float:
        g = self.global_.mean(now)
        return g if context is None else self._ctx(context).mean(now, prior_mean=g)

    def sample(self, rng: np.random.Generator, context: str | None, now: int) -> float:
        g = self.global_.mean(now)
        if context is None:
            return self.global_.sample(rng, now)
        return self._ctx(context).sample(rng, now, prior_mean=g)

"""The treatment group means experiment.

Shared by the coverage animation (``treatment-group-means/``) and the
Berry-Esseen diagnostic (``diagnostics/``). A :class:`Population` of fixed
potential outcomes is turned into observed data by :func:`simulate_experiment`;
the estimator itself lives in ``ate.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    """A fixed finite population of potential outcomes."""

    y0: np.ndarray
    y1: np.ndarray

    @property
    def n(self) -> int:
        return self.y0.size

    @property
    def ate(self) -> float:
        """Average treatment effect, y1_bar - y0_bar."""
        return float(self.y1.mean() - self.y0.mean())


@dataclass
class ExperimentConfig:
    treatment_probability: float
    population: Population
    rng: np.random.Generator


@dataclass
class ExperimentData:
    """Observed data from a single experiment."""

    d: np.ndarray
    y: np.ndarray


def simulate_experiment(config: ExperimentConfig) -> ExperimentData:
    """Draw a single treatment assignment and return the observed outcomes."""
    d = config.rng.binomial(1, config.treatment_probability, size=config.population.n)
    y = d * config.population.y1 + (1 - d) * config.population.y0
    return ExperimentData(d=d, y=y)

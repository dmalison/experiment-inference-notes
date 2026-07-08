"""Monte Carlo simulation of the studentized ATE statistic.

Repeatedly simulates experiments on a fixed :class:`~experiment.Population` and
returns the studentized statistic sqrt(n)(psi_hat - psi)/sigma_hat for each
draw, dropping all-treated/all-control assignments. Shared by the coverage
animation (``treatment-group-means/``) and the Berry-Esseen diagnostic
(``diagnostics/``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.ate import compute_ate
from lib.experiment import ExperimentConfig, Population, simulate_experiment


@dataclass
class SimulationConfig:
    population: Population
    treatment_probability: float
    ci_level: float
    rng: np.random.Generator
    n_draws: int


def simulate_studentized_ate(config: SimulationConfig) -> np.ndarray:
    """Studentized ATE sqrt(n)(psi_hat - psi)/sigma_hat over many experiment draws."""
    population = config.population
    experiment_config = ExperimentConfig(
        config.treatment_probability, population, config.rng
    )
    studentized = []
    for _ in range(config.n_draws):
        experiment = simulate_experiment(experiment_config)
        if experiment.d.sum() in (0, population.n):
            continue
        result = compute_ate(experiment, config.ci_level)
        studentized.append((result.ate_hat - population.ate) / result.se)
    return np.array(studentized)

"""Monte Carlo simulation of experiment results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.ate import compute_ate
from lib.diagnostics import generate_diagnostics
from lib.experiment import ExperimentData, ExperimentSummaryStats, summarize_experiment


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
class SimulationConfig:
    population: Population
    treatment_probability: float
    ci_level: float
    rng: np.random.Generator
    n_draws: int


@dataclass
class SimulationResult:
    ate_hat: float
    treatment_mean_hat: float
    control_mean_hat: float
    lift_hat: float
    berry_esseen_bound: float
    max_variance_share: float
    ate_se_hat: float
    ate_ci_lo: float
    ate_ci_hi: float
    lift_ci_lo: float
    lift_ci_hi: float

    ate_covered: bool
    lift_covered: bool
    studentized_ate: float


def simulate_experiments(config: SimulationConfig) -> list[ExperimentData]:
    """Draw treatment assignments and return observed outcomes."""
    experiment_data = []
    for _ in range(config.n_draws):
        d = config.rng.binomial(1, config.treatment_probability, size=config.population.n)
        y = d * config.population.y1 + (1 - d) * config.population.y0
        experiment_data.append(ExperimentData(treatment_indicator=d, outcome=y))
    return experiment_data


def simulate_summary_stats(config: SimulationConfig) -> list[ExperimentSummaryStats]:
    """Experiment summary statistics over many simulated experiment draws."""
    summary_stats = []
    for experiment_data in simulate_experiments(config):
        if experiment_data.treatment_indicator.sum() in (0, config.population.n):
            continue
        summary_stats.append(summarize_experiment(experiment_data))
    return summary_stats


def simulate_results(config: SimulationConfig) -> list[SimulationResult]:
    """ATE simulation results over many experiment draws."""
    population = config.population
    results = []
    for summary_stats in simulate_summary_stats(config):
        ate_result = compute_ate(summary_stats, config.ci_level)
        diagnostics = generate_diagnostics(summary_stats)
        lift_hat = (
            ate_result.ate_hat / abs(summary_stats.control_mean)
            if summary_stats.control_mean != 0
            else np.nan
        )
        results.append(
            SimulationResult(
                ate_hat=ate_result.ate_hat,
                treatment_mean_hat=summary_stats.treatment_mean,
                control_mean_hat=summary_stats.control_mean,
                lift_hat=float(lift_hat),
                berry_esseen_bound=diagnostics.berry_esseen_bound,
                max_variance_share=diagnostics.max_variance_share,
                ate_se_hat=ate_result.se,
                ate_ci_lo=ate_result.ci_lo,
                ate_ci_hi=ate_result.ci_hi,
                lift_ci_lo=np.nan,
                lift_ci_hi=np.nan,
                ate_covered=(
                    ate_result.ci_lo
                    <= population.ate
                    <= ate_result.ci_hi
                ),
                lift_covered=False,
                studentized_ate=(
                    ate_result.ate_hat - population.ate
                ) / ate_result.se,
            )
        )
    return results

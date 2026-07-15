"""Observed experiment data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentData:
    """Observed data from a single experiment."""

    treatment_indicator: np.ndarray
    outcome: np.ndarray


@dataclass
class ExperimentSummaryStats:
    count: int
    outcome_mean: float
    outcome_std: float
    outcome_third_abs_central_moment: float

    treatment_count: int
    treatment_mean: float
    treatment_std: float
    treatment_max_variance_share: float

    control_count: int
    control_mean: float
    control_std: float
    control_max_variance_share: float


def summarize_experiment(experiment_data: ExperimentData) -> ExperimentSummaryStats:
    """Compute treatment/control means and counts for one experiment."""
    d, y = experiment_data.treatment_indicator, experiment_data.outcome

    outcome_mean = y.mean()
    outcome_centered = y - outcome_mean
    outcome_third_abs_central_moment = float((np.abs(outcome_centered) ** 3).mean())

    treatment_count = int(d.sum())
    treatment_mean = (d * y).sum() / treatment_count
    treatment_std = np.sqrt(max((d * (y - treatment_mean) ** 2).sum() / treatment_count, 0.0))

    control_count = int(d.size - treatment_count)
    control_mean = ((1 - d) * y).sum() / control_count
    control_std = np.sqrt(max(((1 - d) * (y - control_mean) ** 2).sum() / control_count, 0.0))

    return ExperimentSummaryStats(
        count=int(d.size),
        treatment_mean=float(treatment_mean),
        control_mean=float(control_mean),
        treatment_std=float(treatment_std),
        control_std=float(control_std),
        outcome_mean=float(outcome_mean),
        outcome_std=float(np.sqrt((outcome_centered**2).mean())),
        outcome_third_abs_central_moment=float(outcome_third_abs_central_moment),
        treatment_max_variance_share=_max_variance_share(y[d == 1]),
        control_max_variance_share=_max_variance_share(y[d == 0]),
        treatment_count=treatment_count,
        control_count=control_count,
    )


def _max_variance_share(y_group: np.ndarray) -> float:
    """Largest share of within-group variation from a single unit."""
    residuals_sq = (y_group - y_group.mean()) ** 2
    return float(residuals_sq.max() / residuals_sq.sum())

"""Berry-Esseen and variance-share diagnostics for the studentized ATE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.experiment import ExperimentData

BERRY_ESSEEN_CONSTANT = 0.5583


@dataclass
class Diagnostics:
    berry_esseen_bound: float
    max_variance_share: float


def generate_diagnostics(experiment: ExperimentData) -> Diagnostics:
    d, y = experiment.d, experiment.y
    n = y.size
    d_bar = float(d.mean())

    y_bar = float(y.mean())
    Sn = float(np.sqrt(((y - y_bar) ** 2).mean()))
    lambda_n = (np.abs(y - y_bar) ** 3).sum() / (n ** 1.5 * Sn ** 3)
    design_factor = (d_bar ** 2 + (1 - d_bar) ** 2) / np.sqrt(d_bar * (1 - d_bar))
    berry_esseen_bound = float(BERRY_ESSEEN_CONSTANT * design_factor * lambda_n)

    max_variance_share = max(_variance_share(y[d == 1]), _variance_share(y[d == 0]))

    return Diagnostics(
        berry_esseen_bound=berry_esseen_bound,
        max_variance_share=max_variance_share,
    )


def _variance_share(y_arm: np.ndarray) -> float:
    """Largest share of within-arm variation from a single unit (eq-dominant-unit-diagnostic)."""
    residuals_sq = (y_arm - y_arm.mean()) ** 2
    return float(residuals_sq.max() / residuals_sq.sum())

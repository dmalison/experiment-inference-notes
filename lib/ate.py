from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from lib.experiment import ExperimentData


@dataclass
class ATEResult:
    y0_bar_hat: float
    y1_bar_hat: float
    ate_hat: float
    se: float
    ci_level: float
    ci_lo: float
    ci_hi: float


def compute_ate(
    experiment: ExperimentData,
    ci_level: float,
) -> ATEResult:
    """Estimate the ATE and its Wald interval for a single experiment."""
    d, y, n = experiment.d, experiment.y, experiment.d.size
    n_treated, n_control = d.sum(), n - d.sum()
    d_bar = n_treated / n
    y1_bar_hat = (d * y).sum() / n_treated
    y0_bar_hat = ((1 - d) * y).sum() / n_control
    ate_hat = y1_bar_hat - y0_bar_hat
    s1_hat = np.sqrt(max((d * (y - y1_bar_hat) ** 2).sum() / n_treated, 0.0))
    s0_hat = np.sqrt(max(((1 - d) * (y - y0_bar_hat) ** 2).sum() / n_control, 0.0))
    sigma_hat_sq = _conservative_sigma_sq(s0_hat, s1_hat, d_bar)
    se = np.sqrt(sigma_hat_sq / n)
    z_critical = stats.norm.ppf((1 + ci_level) / 2)
    ci_lo, ci_hi = ate_hat - z_critical * se, ate_hat + z_critical * se
    return ATEResult(
        y0_bar_hat=float(y0_bar_hat),
        y1_bar_hat=float(y1_bar_hat),
        ate_hat=float(ate_hat),
        se=float(se),
        ci_level=ci_level,
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
    )


def _conservative_sigma_sq(s0_hat, s1_hat, d_bar):
    """Conservative variance estimator (eq-sigma-hat) for the ATE."""
    return d_bar * (1 - d_bar) * (s0_hat / (1 - d_bar) + s1_hat / d_bar) ** 2

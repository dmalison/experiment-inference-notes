from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from lib.experiment import ExperimentSummaryStats


@dataclass
class ATEResult:
    ate_hat: float
    se: float
    ci_level: float
    ci_lo: float
    ci_hi: float


def compute_ate(
    summary_stats: ExperimentSummaryStats,
    ci_level: float,
) -> ATEResult:
    """Estimate the ATE and its Wald interval from experiment summary stats."""
    ate_hat = summary_stats.treatment_mean - summary_stats.control_mean
    d_bar = summary_stats.treatment_count / summary_stats.count
    se = (
        summary_stats.control_std * (d_bar / (1 - d_bar) / summary_stats.count) ** 0.5
        + summary_stats.treatment_std * ((1 - d_bar) / d_bar / summary_stats.count) ** 0.5
    )
    z_critical = stats.norm.ppf((1 + ci_level) / 2)
    ci_lo, ci_hi = ate_hat - z_critical * se, ate_hat + z_critical * se
    return ATEResult(
        ate_hat=float(ate_hat),
        se=float(se),
        ci_level=ci_level,
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
    )

"""Berry-Esseen and variance-share diagnostics for the studentized ATE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.experiment import ExperimentSummaryStats

BERRY_ESSEEN_CONSTANT = 0.5583


@dataclass
class Diagnostics:
    berry_esseen_bound: float
    max_variance_share: float


def generate_diagnostics(summary_stats: ExperimentSummaryStats) -> Diagnostics:
    d_bar = summary_stats.treatment_count / summary_stats.count

    lambda_n = (
        summary_stats.outcome_third_abs_central_moment
        / (np.sqrt(summary_stats.count) * summary_stats.outcome_std**3)
    )
    design_factor = (d_bar ** 2 + (1 - d_bar) ** 2) / np.sqrt(d_bar * (1 - d_bar))
    berry_esseen_bound = float(BERRY_ESSEEN_CONSTANT * design_factor * lambda_n)

    max_variance_share = max(
        summary_stats.treatment_max_variance_share,
        summary_stats.control_max_variance_share,
    )

    return Diagnostics(
        berry_esseen_bound=berry_esseen_bound,
        max_variance_share=max_variance_share,
    )

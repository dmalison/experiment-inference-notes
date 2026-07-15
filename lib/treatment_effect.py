"""Treatment effect inference results."""
from __future__ import annotations

from dataclasses import dataclass

from lib.ate import compute_ate
from lib.diagnostics import generate_diagnostics
from lib.experiment import ExperimentData, summarize_experiment
from lib.lift import compute_lift_result


@dataclass
class TreatmentEffectResult:
    treatment_mean_hat: float
    control_mean_hat: float

    ate_hat: float
    ate_se_hat: float
    ate_ci_lo: float
    ate_ci_hi: float

    lift_hat: float
    lift_ci_lo: float
    lift_ci_hi: float

    berry_esseen_bound: float
    max_variance_share: float

    ci_level: float


def compute_treatment_effect_result(
    experiment_data: ExperimentData,
    ci_level: float,
) -> TreatmentEffectResult:
    """Estimate treatment effects and ATE uncertainty for a single experiment."""
    summary_stats = summarize_experiment(experiment_data)
    ate_result = compute_ate(summary_stats, ci_level)
    diagnostics = generate_diagnostics(summary_stats)
    lift_result = compute_lift_result(summary_stats, ci_level)
    return TreatmentEffectResult(
        treatment_mean_hat=summary_stats.treatment_mean,
        control_mean_hat=summary_stats.control_mean,
        ci_level=ci_level,
        berry_esseen_bound=diagnostics.berry_esseen_bound,
        max_variance_share=diagnostics.max_variance_share,
        ate_hat=ate_result.ate_hat,
        ate_se_hat=ate_result.se,
        ate_ci_lo=ate_result.ci_lo,
        ate_ci_hi=ate_result.ci_hi,
        lift_hat=lift_result.lift_hat,
        lift_ci_lo=lift_result.lift_ci_lo,
        lift_ci_hi=lift_result.lift_ci_hi,
    )

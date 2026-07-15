"""Lift inference results."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

from lib.experiment import ExperimentSummaryStats


class BaselineSign(Enum):
	POSITIVE = "positive"
	NEGATIVE = "negative"
	UNKNOWN = "unknown"


@dataclass
class LiftResult:
	lift_hat: float
	lift_ci_lo: float
	lift_ci_hi: float

	ci_level: float


def compute_lift_result(
	summary_stats: ExperimentSummaryStats,
	ci_level: float,
	fieller_share: float = 0.98,
	baseline_sign: BaselineSign = BaselineSign.UNKNOWN,
) -> LiftResult:
	"""Compute the uniformly valid Fieller lift interval from summary stats."""
	lift_hat = _compute_point_estimate(summary_stats)
	lift_ci_lo, lift_ci_hi = _compute_ci(
		summary_stats,
		ci_level=ci_level,
		fieller_share=fieller_share,
		baseline_sign=baseline_sign,
	)

	return LiftResult(
		lift_hat=float(lift_hat),
		lift_ci_lo=float(lift_ci_lo),
		lift_ci_hi=float(lift_ci_hi),
		ci_level=ci_level,
	)


def _compute_point_estimate(summary_stats: ExperimentSummaryStats) -> float:
	ate_hat = summary_stats.treatment_mean - summary_stats.control_mean
	return (
		ate_hat / abs(summary_stats.control_mean)
		if summary_stats.control_mean != 0
		else np.nan
	)


def _compute_ci(
	summary_stats: ExperimentSummaryStats,
	ci_level: float,
	fieller_share: float,
	baseline_sign: BaselineSign,
) -> tuple[float, float]:
	alpha_fieller, alpha_baseline = _split_alpha(ci_level, fieller_share)
	if baseline_sign == BaselineSign.UNKNOWN:
		return _unknown_sign_interval(
			summary_stats,
			alpha_fieller=alpha_fieller,
			alpha_baseline=alpha_baseline,
		)
	return _known_sign_interval(
		summary_stats,
		baseline_sign=baseline_sign,
		alpha_fieller=alpha_fieller,
		alpha_baseline=alpha_baseline,
	)


def _split_alpha(ci_level: float, fieller_share: float) -> tuple[float, float]:
	if not 0 < fieller_share < 1:
		raise ValueError("fieller_share must be between 0 and 1")
	if fieller_share <= 0.5:
		raise ValueError("fieller_share must exceed .5")
	alpha = 1 - ci_level
	alpha_fieller = fieller_share * alpha
	alpha_baseline = alpha - alpha_fieller
	return alpha_fieller, alpha_baseline


def _unknown_sign_interval(
	summary_stats: ExperimentSummaryStats,
	alpha_fieller: float,
	alpha_baseline: float,
) -> tuple[float, float]:
	control_mean_se, _ = _group_mean_standard_errors(summary_stats)
	baseline_t_stat = _safe_divide(
		summary_stats.control_mean,
		control_mean_se,
	)
	baseline_sign_threshold = stats.norm.ppf(1 - alpha_baseline / 2)
	if baseline_t_stat > baseline_sign_threshold:
		return _interval_hull(sign_specific_fieller_set(
			summary_stats,
			denominator_sign=1,
			alpha=alpha_fieller,
		))
	if baseline_t_stat < -baseline_sign_threshold:
		return _interval_hull(sign_specific_fieller_set(
			summary_stats,
			denominator_sign=-1,
			alpha=alpha_fieller,
		))
	return -np.inf, np.inf


def _known_sign_interval(
	summary_stats: ExperimentSummaryStats,
	baseline_sign: BaselineSign,
	alpha_fieller: float,
	alpha_baseline: float,
) -> tuple[float, float]:
	denominator_sign = 1 if baseline_sign == BaselineSign.POSITIVE else -1
	control_mean_se, treatment_mean_se = _group_mean_standard_errors(summary_stats)
	baseline_t_stat = abs(_safe_divide(
		summary_stats.control_mean,
		control_mean_se,
	))
	z_fieller = stats.norm.ppf(1 - alpha_fieller / 2)
	fieller_set = sign_specific_fieller_set(
		summary_stats,
		denominator_sign=denominator_sign,
		alpha=alpha_fieller,
	)
	if baseline_t_stat > z_fieller:
		return _interval_hull(fieller_set)

	ate_stat = _safe_divide(
		summary_stats.treatment_mean - summary_stats.control_mean,
		control_mean_se + treatment_mean_se,
	)
	z_baseline = stats.norm.ppf(1 - alpha_baseline / 2)
	if ate_stat > z_baseline:
		return _interval_hull([
			(max(lo, 0), hi)
			for lo, hi in fieller_set
			if max(lo, 0) <= hi
		])
	if ate_stat < -z_baseline:
		return _interval_hull([
			(lo, min(hi, 0))
			for lo, hi in fieller_set
			if lo <= min(hi, 0)
		])
	return -np.inf, np.inf


def _group_mean_standard_errors(summary_stats: ExperimentSummaryStats) -> tuple[float, float]:
	d_bar = summary_stats.treatment_count / summary_stats.count
	control_mean_se = summary_stats.control_std * np.sqrt(d_bar / (1 - d_bar) / summary_stats.count)
	treatment_mean_se = summary_stats.treatment_std * np.sqrt((1 - d_bar) / d_bar / summary_stats.count)
	return float(control_mean_se), float(treatment_mean_se)


def _safe_divide(numerator: float, denominator: float) -> float:
	if denominator != 0:
		return numerator / denominator
	return float(np.sign(numerator) * np.inf) if numerator != 0 else 0.0


def sign_specific_fieller_set(
	summary_stats: ExperimentSummaryStats,
	denominator_sign: int,
	alpha: float,
) -> list[tuple[float, float]]:
	threshold = float(stats.norm.ppf(1 - alpha / 2))
	intervals = []
	for abs_sign in (1, -1):
		coefficients = _fieller_quadratic_coefficients(
			summary_stats=summary_stats,
			threshold=threshold,
			denominator_sign=denominator_sign,
			abs_sign=abs_sign,
		)
		intervals.extend(
			_quadratic_sublevel_intervals(*coefficients)
		)
	return intervals


def _interval_hull(intervals: list[tuple[float, float]]) -> tuple[float, float]:
	if not intervals:
		return np.nan, np.nan
	return min(lo for lo, _ in intervals), max(hi for _, hi in intervals)


def _fieller_quadratic_coefficients(
	summary_stats: ExperimentSummaryStats,
	threshold: float,
	denominator_sign: int,
	abs_sign: int,
) -> tuple[float, float, float]:
	ate_hat = summary_stats.treatment_mean - summary_stats.control_mean
	control_mean_se, treatment_mean_se = _group_mean_standard_errors(summary_stats)
	ate_se = control_mean_se + treatment_mean_se

	a = summary_stats.control_mean**2 - threshold**2 * control_mean_se**2
	b = -2 * denominator_sign * (
		summary_stats.control_mean * ate_hat
		+ threshold**2
		* (
			control_mean_se**2
			+ abs_sign * control_mean_se * treatment_mean_se
		)
	)
	c = ate_hat**2 - threshold**2 * ate_se**2
	return a, b, c


def _quadratic_sublevel_intervals(a: float, b: float, c: float) -> list[tuple[float, float]]:
	tolerance = 1e-12
	if abs(a) < tolerance:
		if abs(b) < tolerance:
			return [(-np.inf, np.inf)] if c <= 0 else []
		root = -c / b
		return [(-np.inf, root)] if b > 0 else [(root, np.inf)]

	discriminant = b**2 - 4 * a * c
	if discriminant < -tolerance:
		return [(-np.inf, np.inf)] if a < 0 else []
	root_offset = np.sqrt(max(discriminant, 0.0))
	roots = sorted(((-b - root_offset) / (2 * a), (-b + root_offset) / (2 * a)))
	if a > 0:
		return [(roots[0], roots[1])]
	return [(-np.inf, roots[0]), (roots[1], np.inf)]

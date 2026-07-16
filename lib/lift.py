"""Lift inference results."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy import stats

from lib.experiment import ExperimentSummaryStats


class BaselineSign(IntEnum):
	POSITIVE = 1
	NEGATIVE = -1
	UNKNOWN = 0


@dataclass
class LiftResult:
	lift_hat: float
	lift_ci_lo: float
	lift_ci_hi: float

	ci_level: float


def compute_lift_result(
	summary_stats: ExperimentSummaryStats,
	ci_level: float = 0.95,
	sign_test_level: float = 0.999,
	baseline_sign: BaselineSign = BaselineSign.UNKNOWN,
) -> LiftResult:
	"""Compute the uniformly valid Fieller lift interval from summary stats."""
	lift_hat = _compute_point_estimate(summary_stats)
	lift_ci_lo, lift_ci_hi = _compute_ci(
		summary_stats,
		ci_level=ci_level,
		sign_test_level=sign_test_level,
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
	return _safe_divide(ate_hat, abs(summary_stats.control_mean))


def _compute_ci(
	summary_stats: ExperimentSummaryStats,
	ci_level: float,
	sign_test_level: float,
	baseline_sign: BaselineSign,
) -> tuple[float, float]:
	alpha_fieller, alpha_sign_test = _split_alpha(ci_level, sign_test_level)
	if baseline_sign == BaselineSign.UNKNOWN:
		return _unknown_sign_interval(
			summary_stats,
			alpha_fieller=alpha_fieller,
			alpha_baseline=alpha_sign_test,
		)
	return _known_sign_interval(
		summary_stats,
		baseline_sign=baseline_sign,
		alpha_fieller=alpha_fieller,
		alpha_ate=alpha_sign_test,
	)


def _split_alpha(ci_level: float, sign_test_level: float) -> tuple[float, float]:
	if not 0 < ci_level < 1:
		raise ValueError("ci_level must be between 0 and 1")
	if not (1 + ci_level) / 2 < sign_test_level < 1:
		raise ValueError("sign_test_level must exceed (1 + ci_level) / 2 and be less than 1")
	alpha = 1 - ci_level
	alpha_sign_test = 1 - sign_test_level
	alpha_fieller = alpha - alpha_sign_test
	return alpha_fieller, alpha_sign_test


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
		return _intervals_hull(_sign_specific_fieller_set(
			summary_stats,
			baseline_sign=BaselineSign.POSITIVE,
			alpha=alpha_fieller,
		))
	if baseline_t_stat < -baseline_sign_threshold:
		return _intervals_hull(_sign_specific_fieller_set(
			summary_stats,
			baseline_sign=BaselineSign.NEGATIVE,
			alpha=alpha_fieller,
		))
	return -np.inf, np.inf


def _known_sign_interval(
	summary_stats: ExperimentSummaryStats,
	baseline_sign: BaselineSign,
	alpha_fieller: float,
	alpha_ate: float,
) -> tuple[float, float]:
	control_mean_se, treatment_mean_se = _group_mean_standard_errors(summary_stats)
	baseline_t_stat = _safe_divide(
		summary_stats.control_mean,
		control_mean_se,
	)
	z_fieller = stats.norm.ppf(1 - alpha_fieller / 2)
	fieller_set = _sign_specific_fieller_set(
		summary_stats,
		baseline_sign=baseline_sign,
		alpha=alpha_fieller,
	)
	if abs(baseline_t_stat) > z_fieller:
		return _intervals_hull(fieller_set)

	ate_stat = _safe_divide(
		summary_stats.treatment_mean - summary_stats.control_mean,
		control_mean_se + treatment_mean_se,
	)
	z_ate = stats.norm.ppf(1 - alpha_ate / 2)
	if ate_stat > z_ate:
		nonnegative_fieller_set = _truncate_intervals(fieller_set, lower=0)
		return _intervals_hull(nonnegative_fieller_set)
	if ate_stat < -z_ate:
		nonpositive_fieller_set = _truncate_intervals(fieller_set, upper=0)
		return _intervals_hull(nonpositive_fieller_set)
	return -np.inf, np.inf


def _sign_specific_fieller_set(
	summary_stats: ExperimentSummaryStats,
	baseline_sign: BaselineSign,
	alpha: float,
) -> list[tuple[float, float]]:
	threshold = float(stats.norm.ppf(1 - alpha / 2))
	intervals = []
	baseline_sign_value = baseline_sign.value
	for abs_sign_value in (1, -1):
		coefficients = _fieller_quadratic_coefficients(
			summary_stats=summary_stats,
			threshold=threshold,
			baseline_sign_value=baseline_sign_value,
			abs_sign_value=abs_sign_value,
		)
		intervals.extend(
			_fieller_subintervals(
				coefficients,
				baseline_sign_value=baseline_sign_value,
				abs_sign_value=abs_sign_value,
			)
		)
	return _union_intervals(intervals)


def _fieller_quadratic_coefficients(
	summary_stats: ExperimentSummaryStats,
	threshold: float,
	baseline_sign_value: int,
	abs_sign_value: int,
) -> tuple[float, float, float]:
	ate_hat = summary_stats.treatment_mean - summary_stats.control_mean
	control_mean_se, treatment_mean_se = _group_mean_standard_errors(summary_stats)
	ate_se = control_mean_se + treatment_mean_se

	a = summary_stats.control_mean**2 - threshold**2 * control_mean_se**2
	b = (
		-2
		* baseline_sign_value
		* (
			summary_stats.control_mean * ate_hat
			+ threshold**2 * control_mean_se**2
		)
		- 2
		* (abs_sign_value * baseline_sign_value)
		* threshold**2
		* control_mean_se
		* treatment_mean_se
	)
	c = (
		ate_hat**2
		- threshold**2 * ate_se**2
		- 2
		* threshold**2
		* control_mean_se
		* treatment_mean_se
		* (abs_sign_value - 1)
	)
	return a, b, c


def _fieller_subintervals(
	coefficients: tuple[float, float, float],
	baseline_sign_value: int,
	abs_sign_value: int,
) -> list[tuple[float, float]]:
	quadratic_intervals = _quadratic_nonpositive_intervals(*coefficients)
	if abs_sign_value * baseline_sign_value > 0:
		return _truncate_intervals(quadratic_intervals, lower=-baseline_sign_value)
	return _truncate_intervals(quadratic_intervals, upper=-baseline_sign_value)


def _quadratic_nonpositive_intervals(a: float, b: float, c: float) -> list[tuple[float, float]]:
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


def _intervals_hull(intervals: list[tuple[float, float]]) -> tuple[float, float]:
	if not intervals:
		return np.nan, np.nan
	return min(lo for lo, _ in intervals), max(hi for _, hi in intervals)


def _truncate_intervals(
	intervals: list[tuple[float, float]],
	lower: float = -np.inf,
	upper: float = np.inf,
) -> list[tuple[float, float]]:
	return _union_intervals([
		(max(lo, lower), min(hi, upper))
		for lo, hi in intervals
		if max(lo, lower) <= min(hi, upper)
	])


def _union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
	if not intervals:
		return []
	tolerance = 1e-12
	sorted_intervals = sorted(intervals)
	union_intervals = [sorted_intervals[0]]
	for lo, hi in sorted_intervals[1:]:
		last_lo, last_hi = union_intervals[-1]
		if lo <= last_hi + tolerance:
			union_intervals[-1] = (last_lo, max(last_hi, hi))
		else:
			union_intervals.append((lo, hi))
	return union_intervals


def _group_mean_standard_errors(summary_stats: ExperimentSummaryStats) -> tuple[float, float]:
	d_bar = summary_stats.treatment_count / summary_stats.count
	control_mean_se = summary_stats.control_std * np.sqrt(d_bar / (1 - d_bar) / summary_stats.count)
	treatment_mean_se = summary_stats.treatment_std * np.sqrt((1 - d_bar) / d_bar / summary_stats.count)
	return float(control_mean_se), float(treatment_mean_se)


def _safe_divide(numerator: float, denominator: float) -> float:
	if denominator != 0:
		return numerator / denominator
	return float(np.sign(numerator) * np.inf) if numerator != 0 else np.nan

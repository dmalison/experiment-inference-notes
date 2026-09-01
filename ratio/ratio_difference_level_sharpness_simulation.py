"""Sharpness of the level-(1 - alpha) Fieller difference interval.

We compare two ways of combining treatment-group-specific Fieller sets into a
confidence interval for the ratio difference r_n = r_{1,n} - r_{0,n}:

  * Bonferroni:   each arm at level 1 - alpha/2  (quantile z_{1 - alpha/4}),
  * Uncorrected:  each arm at level 1 - alpha    (quantile z_{1 - alpha/2}).

Both denominators are held well separated from zero, so both arm intervals are
bounded and Fieller coincides with the delta-method (Wald) approximation. A
single family of n = 10,000 populations is generated in which the projected
potential outcomes of the two arms,

    v_{i,d} = y_{i,d} - r_{d,n} z_{i,d},

have a controllable cross-arm correlation c = Corr(v_{i,1}, v_{i,0}). Only the
Bernoulli(1/2) treatment assignment is rerandomized; the same assignments are
reused at every c.

As c -> 1 the two arm ratio estimators become perfectly negatively correlated
(the assignment design flips the sign of the cross-arm covariance). In this
regular-denominator configuration, the uncorrected interval's non-coverage
approaches alpha, while the Bonferroni interval remains conservative. The
simulation illustrates the analytic sharpness result; it does not establish
uniform validity over weak denominators.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy import stats

SEED = 20240521
ASSIGNMENT_SEED = 99
N = 10_000
P = 0.5
ALPHA = 0.05
DRAWS = 120_000
CHUNK_SIZE = 500

DENOMINATOR_STRENGTH = 24.0          # both arms; strong, so intervals are bounded
CONTROL_RATIO = 1.0                  # r_{0,n}
TREATMENT_RATIO = 3.0                # r_{1,n}
CORRELATION_GRID = np.linspace(0.0, 1.0, 11)

Z_UNCORRECTED = float(stats.norm.ppf(1.0 - ALPHA / 2.0))   # each arm at 1 - alpha
Z_BONFERRONI = float(stats.norm.ppf(1.0 - ALPHA / 4.0))    # each arm at 1 - alpha/2

UNCORRECTED_COLOR = "#2f2f2f"
BONFERRONI_COLOR = "#c1440e"
NOMINAL_COLOR = "0.5"
NONCOVERAGE_Y_MAX = 0.06


def normalized(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.standard_normal(n)
    x -= x.mean()
    x /= x.std()
    return x


def make_population(correlation: float) -> tuple[np.ndarray, ...]:
    """Symmetric strong denominators; Corr(v_1, v_0) = correlation."""
    rng = np.random.default_rng(SEED)
    eps = normalized(rng, N)          # shared projected residual
    eps_perp = normalized(rng, N)
    eps_perp -= np.mean(eps_perp * eps) * eps
    eps_perp /= eps_perp.std()
    v0 = eps
    v1 = correlation * eps + np.sqrt(max(1.0 - correlation**2, 0.0)) * eps_perp

    zeta0 = normalized(rng, N)         # denominator residuals, independent
    zeta1 = normalized(rng, N)
    denom_mean = DENOMINATOR_STRENGTH / np.sqrt(N)
    z0 = denom_mean + zeta0
    z1 = denom_mean + zeta1
    y0 = CONTROL_RATIO * z0 + v0
    y1 = TREATMENT_RATIO * z1 + v1
    return y0, z0, y1, z1


def arm_moment_columns(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.column_stack((y, y**2, z, z**2, y * z))


def assignment_summaries(
    y0: np.ndarray, z0: np.ndarray, y1: np.ndarray, z1: np.ndarray, draws: int
) -> dict[str, np.ndarray]:
    control_columns = arm_moment_columns(y0, z0)
    treatment_columns = arm_moment_columns(y1, z1)
    control_totals = control_columns.sum(axis=0)

    keys = (
        "dbar", "y1", "z1", "syy1", "szz1", "syz1",
        "y0", "z0", "syy0", "szz0", "syz0",
    )
    out: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    rng = np.random.default_rng(ASSIGNMENT_SEED)
    completed = 0
    while completed < draws:
        batch = min(CHUNK_SIZE, draws - completed)
        treatment = rng.random((batch, N)) < P
        treatment_f = treatment.astype(float)
        treatment_count = treatment_f.sum(axis=1)
        control_count = N - treatment_count

        treatment_sums = treatment_f @ treatment_columns
        control_sums = control_totals - treatment_f @ control_columns

        treatment_means = treatment_sums / treatment_count[:, None]
        control_means = control_sums / control_count[:, None]

        out["dbar"].append(treatment_count / N)
        out["y1"].append(treatment_means[:, 0])
        out["z1"].append(treatment_means[:, 2])
        out["syy1"].append(treatment_means[:, 1] - treatment_means[:, 0] ** 2)
        out["szz1"].append(treatment_means[:, 3] - treatment_means[:, 2] ** 2)
        out["syz1"].append(treatment_means[:, 4] - treatment_means[:, 0] * treatment_means[:, 2])
        out["y0"].append(control_means[:, 0])
        out["z0"].append(control_means[:, 2])
        out["syy0"].append(control_means[:, 1] - control_means[:, 0] ** 2)
        out["szz0"].append(control_means[:, 3] - control_means[:, 2] ** 2)
        out["syz0"].append(control_means[:, 4] - control_means[:, 0] * control_means[:, 2])
        completed += batch

    return {key: np.concatenate(value) for key, value in out.items()}


def fieller_bounds(
    ybar, zbar, syy, szz, syz, design_factor, z_arm
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_y_sq = design_factor * syy
    sigma_z_sq = design_factor * szz
    sigma_yz = design_factor * syz
    a = zbar**2 - z_arm**2 * sigma_z_sq / N
    b = ybar * zbar - z_arm**2 * sigma_yz / N
    c = ybar**2 - z_arm**2 * sigma_y_sq / N
    bounded = a > 0
    radius = np.sqrt(np.maximum(b**2 - a * c, 0.0))
    lower = np.full(ybar.shape, np.nan)
    upper = np.full(ybar.shape, np.nan)
    lower[bounded] = (b[bounded] - radius[bounded]) / a[bounded]
    upper[bounded] = (b[bounded] + radius[bounded]) / a[bounded]
    return bounded, lower, upper


def noncoverage(summaries: dict[str, np.ndarray], z_arm: float) -> float:
    dbar = summaries["dbar"]
    treatment_factor = (1.0 - dbar) / dbar
    control_factor = dbar / (1.0 - dbar)

    bounded1, lower1, upper1 = fieller_bounds(
        summaries["y1"], summaries["z1"], summaries["syy1"],
        summaries["szz1"], summaries["syz1"], treatment_factor, z_arm,
    )
    bounded0, lower0, upper0 = fieller_bounds(
        summaries["y0"], summaries["z0"], summaries["syy0"],
        summaries["szz0"], summaries["syz0"], control_factor, z_arm,
    )

    true_difference = TREATMENT_RATIO - CONTROL_RATIO
    both_bounded = bounded1 & bounded0
    difference_lower = lower1 - upper0
    difference_upper = upper1 - lower0
    misses = np.zeros(dbar.size, dtype=bool)
    misses[both_bounded] = (
        (true_difference < difference_lower[both_bounded])
        | (true_difference > difference_upper[both_bounded])
    )
    return float(misses.mean())


def main() -> None:
    uncorrected = np.empty(CORRELATION_GRID.size)
    bonferroni = np.empty(CORRELATION_GRID.size)
    for index, correlation in enumerate(CORRELATION_GRID):
        summaries = assignment_summaries(*make_population(correlation), DRAWS)
        uncorrected[index] = noncoverage(summaries, Z_UNCORRECTED)
        bonferroni[index] = noncoverage(summaries, Z_BONFERRONI)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axhline(ALPHA, color=NOMINAL_COLOR, lw=1.0, linestyle="--", label="nominal")
    ax.axhline(ALPHA / 2.0, color=NOMINAL_COLOR, lw=0.8, linestyle=":", label=r"$\alpha/2$")
    ax.plot(
        CORRELATION_GRID, uncorrected, color=UNCORRECTED_COLOR, marker="o", ms=4,
        lw=1.5, label=r"each arm at $1-\alpha$",
    )
    ax.plot(
        CORRELATION_GRID, bonferroni, color=BONFERRONI_COLOR, marker="s", ms=4,
        lw=1.5, label=r"each arm at $1-\alpha/2$ (Bonferroni)",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, NONCOVERAGE_Y_MAX)
    ax.set_xlabel(
        r"cross-arm projected-outcome correlation "
        r"$\mathrm{Corr}(v_{i,1}, v_{i,0})$",
        fontsize=11,
    )
    ax.set_ylabel(r"$95\%$ interval non-coverage", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(
        fontsize=9, loc="upper left", frameon=True, framealpha=0.9,
        edgecolor="none", facecolor="white",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    output = Path(__file__).with_name("ratio_difference_level_sharpness.png")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")
    print("corr   each(1-a)  Bonferroni")
    for correlation, unc, bon in zip(CORRELATION_GRID, uncorrected, bonferroni):
        print(f"{correlation:.2f}   {unc:.4f}     {bon:.4f}")


if __name__ == "__main__":
    main()
